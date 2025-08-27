"""
Data preprocessing module for Sinhala/Singlish Hate Speech Detection with XLM-RoBERTa
Uses word-level tokenization by spaces (matching SOLD dataset format)
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class SOLDDatasetXLM(Dataset):
    """Custom Dataset for SOLD hate speech detection with word-level tokens"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text by spaces (word-level)
        words = text.split()
        
        # Create input_ids and attention_mask for the words
        input_ids = []
        attention_mask = []
        
        # Add [CLS] token
        input_ids.append(self.tokenizer.cls_token_id)
        attention_mask.append(1)
        
        # Process each word
        for word in words:
            # Get token IDs for the word
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(word_tokens)
            attention_mask.extend([1] * len(word_tokens))
            
            # Add space token if not last word
            if word != words[-1]:
                input_ids.append(self.tokenizer.sep_token_id)
                attention_mask.append(1)
        
        # Add [SEP] token
        input_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)
        
        # Pad to max_length
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
        
        # Truncate if too long
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        
        # Create labels array (same length as input_ids)
        labels_array = [-100] * self.max_length  # -100 for special tokens and padding
        
        # Map original labels to token positions
        label_idx = 0
        for i, token_id in enumerate(input_ids):
            if token_id == self.tokenizer.cls_token_id or token_id == self.tokenizer.sep_token_id:
                continue
            elif token_id == self.tokenizer.pad_token_id:
                break
            else:
                if label_idx < len(label):
                    labels_array[i] = label[label_idx]
                    label_idx += 1
        
        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels_array, dtype=torch.long)
        }
        
        return item

def load_sold_dataset(file_path: str) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Load SOLD dataset from JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (tokens_list, labels_list)
    """
    tokens_list = []
    labels_list = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            tokens = data['tokens']
            labels = data['labels']
            
            # Ensure tokens and labels have same length
            if len(tokens) == len(labels):
                tokens_list.append(tokens)
                labels_list.append(labels)
    
    print(f"Loaded {len(tokens_list)} samples from {file_path}")
    return tokens_list, labels_list

def create_word_level_dataset(tokens_list: List[List[str]], 
                             labels_list: List[List[int]], 
                             max_length: int = 256) -> Tuple[List[str], List[List[int]]]:
    """
    Create dataset with word-level tokenization (by spaces)
    
    Args:
        tokens_list: List of token lists from SOLD dataset
        labels_list: List of label lists from SOLD dataset
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (processed_texts, processed_labels)
    """
    processed_texts = []
    processed_labels = []
    
    for tokens, labels in zip(tokens_list, labels_list):
        # Join tokens with spaces to create text
        text = " ".join(tokens)
        
        # Create labels array for the text
        text_labels = []
        word_idx = 0
        
        for token in tokens:
            # Each word gets its corresponding label
            if word_idx < len(labels):
                text_labels.append(labels[word_idx])
                word_idx += 1
        
        processed_texts.append(text)
        processed_labels.append(text_labels)
    
    return processed_texts, processed_labels

def load_hate_dictionary(file_path: str) -> set:
    """
    Load hate dictionary from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Set of hate words/phrases
    """
    try:
        df = pd.read_csv(file_path, header=None)
        hate_words = set(df[0].str.strip().tolist())
        print(f"Loaded {len(hate_words)} hate words from dictionary")
        return hate_words
    except Exception as e:
        print(f"Error loading hate dictionary: {e}")
        return set()

def prepare_word_level_loaders(train_texts, train_labels, val_texts, val_labels, 
                              tokenizer, batch_size=16):
    """
    Prepare DataLoaders for training and validation with word-level tokenization
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        tokenizer: Tokenizer
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = SOLDDatasetXLM(train_texts, train_labels, tokenizer, 256)
    val_dataset = SOLDDatasetXLM(val_texts, val_labels, tokenizer, 256)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader

def main():
    """Main function to test word-level data preprocessing"""
    print("Testing word-level data preprocessing...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Load datasets
    train_tokens, train_labels = load_sold_dataset("datsets/SOLD_train_cleaned.json")
    test_tokens, test_labels = load_sold_dataset("datsets/SOLD_test_cleaned.json")
    
    # Load hate dictionary
    hate_dict = load_hate_dictionary("datsets/only_hate.csv")
    
    # Create word-level dataset
    train_texts, train_processed_labels = create_word_level_dataset(
        train_tokens, train_labels, 256
    )
    
    # Split into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_processed_labels,
        test_size=0.1,
        random_state=42
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_tokens)}")
    
    # Test data loader creation
    train_loader, val_loader = prepare_word_level_loaders(
        train_texts, train_labels, val_texts, val_labels, tokenizer, 16
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        break
    
    print("Word-level data preprocessing test completed successfully!")

if __name__ == "__main__":
    main()
