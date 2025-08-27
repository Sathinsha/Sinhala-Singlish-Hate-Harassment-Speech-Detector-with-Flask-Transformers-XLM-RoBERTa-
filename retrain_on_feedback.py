import os
import csv
import json
from datetime import datetime
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config_xlm_roberta as config

class FeedbackDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        words = text.split()

        input_ids = [self.tokenizer.cls_token_id]
        attention_mask = [1]
        labels_array = [-100]

        for i, word in enumerate(words):
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(word_ids)
            attention_mask.extend([1] * len(word_ids))
            # Assign the provided label for the word if available; default neutral
            label = labels[i] if i < len(labels) else 0
            labels_array.extend([label] + [-100] * (len(word_ids) - 1))
            if i < len(words) - 1:
                input_ids.append(self.tokenizer.sep_token_id)
                attention_mask.append(1)
                labels_array.append(-100)

        input_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)
        labels_array.append(-100)

        # pad/truncate
        input_ids = input_ids[:config.MAX_LENGTH]
        attention_mask = attention_mask[:config.MAX_LENGTH]
        labels_array = labels_array[:config.MAX_LENGTH]
        while len(input_ids) < config.MAX_LENGTH:
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            labels_array.append(-100)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels_array, dtype=torch.long)
        }

def load_feedback() -> Tuple[List[str], List[List[int]]]:
    texts: List[str] = []
    labels: List[List[int]] = []
    if not os.path.exists(config.FEEDBACK_CSV):
        return texts, labels
    with open(config.FEEDBACK_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get('input_text') or '').strip()
            model_label = (row.get('model_label') or '').strip()
            user_thinks_label = (row.get('user_thinks_label') or '').strip()
            user_words = (row.get('user_words') or '').strip()
            if not text or not user_words:
                continue
            words = text.split()
            word_set = set([w.strip() for w in user_words.split(',') if w.strip()])
            # Build word-level labels: 1 if user marked this word as hate, else 0
            word_labels = [1 if (w in word_set or w.lower() in word_set) and user_thinks_label == 'HATE' else
                           0 if (w in word_set or w.lower() in word_set) and user_thinks_label == 'NOT_HATE' else
                           0 for w in words]
            texts.append(text)
            labels.append(word_labels)
    return texts, labels


def fine_tune_on_feedback(num_epochs: int = 2, lr: float = 2e-5, batch_size: int = 8):
    texts, word_labels = load_feedback()
    if not texts:
        print('No feedback found. Nothing to retrain.')
        return

    print(f'Loaded {len(texts)} feedback samples')
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SAVE_PATH)
    model = AutoModelForTokenClassification.from_pretrained(config.MODEL_SAVE_PATH)

    dataset = FeedbackDataset(texts, word_labels, tokenizer, config.MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f'Fine-tuning (epoch {epoch+1}/{num_epochs})')
        for batch in pbar:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        print(f'Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}')

    # Save to new snapshot
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(config.RETRAIN_OUTPUT_BASE, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f'Fine-tuned model saved to {out_dir}')

if __name__ == '__main__':
    fine_tune_on_feedback()
