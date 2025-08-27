"""
Inference module for Sinhala/Singlish Hate Speech Detection with XLM-RoBERTa
Uses word-level tokenization by spaces (matching SOLD dataset format)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import Dict, List, Tuple
import pandas as pd
import config_xlm_roberta as config
import os

class HateSpeechDetectorXLM:
    """Main class for hate speech detection with word-level tokens"""
    
    def __init__(self, model_path: str = None, hate_dict_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_path is None:
            model_path = config.MODEL_SAVE_PATH
        if os.path.exists(model_path):
            print(f"Loading trained XLM-RoBERTa model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        else:
            print(f"Trained model not found at {model_path}, using pretrained XLM-RoBERTa")
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            self.model = AutoModelForTokenClassification.from_pretrained(
                config.MODEL_NAME, 
                num_labels=config.NUM_LABELS
            )
        self.model.to(self.device)
        self.model.eval()
        if hate_dict_path is None:
            hate_dict_path = config.HATE_DICT_FILE
        self.hate_dict = self._load_hate_dictionary(hate_dict_path)
        print(f"Hate speech detector initialized on {self.device}")
        print(f"Loaded {len(self.hate_dict)} hate words from dictionary")
        print(f"Using word-level tokenization (by spaces)")
    
    def _load_hate_dictionary(self, file_path: str) -> set:
        try:
            df = pd.read_csv(file_path, header=None)
            hate_words = set(df[0].str.strip().tolist())
            return hate_words
        except Exception as e:
            print(f"Error loading hate dictionary: {e}")
            return set()
    
    def preprocess_text(self, text: str) -> str:
        return " ".join(text.split())
    
    def tokenize_by_words(self, text: str) -> Tuple[List[str], List[int], List[int]]:
        words = text.split()
        word_starts = []
        word_ends = []
        current_pos = 0
        for word in words:
            word_starts.append(current_pos)
            current_pos += len(word)
            word_ends.append(current_pos)
            current_pos += 1
        return words, word_starts, word_ends
    
    def _build_input_with_spans(self, words: List[str]) -> Tuple[List[int], List[int], List[Tuple[int,int]]]:
        """Return input_ids, attention_mask, and word->token spans (inclusive indices) in input space.
        Spans exclude special tokens/padding.
        """
        input_ids: List[int] = []
        attention_mask: List[int] = []
        word_token_spans: List[Tuple[int,int]] = []

        # add CLS
        input_ids.append(self.tokenizer.cls_token_id)
        attention_mask.append(1)

        token_index = 1  # current position in input_ids after CLS
        for i, word in enumerate(words):
            word_subtokens = self.tokenizer.encode(word, add_special_tokens=False)
            start_idx = token_index
            input_ids.extend(word_subtokens)
            attention_mask.extend([1] * len(word_subtokens))
            token_index += len(word_subtokens)
            end_idx = token_index - 1  # inclusive
            word_token_spans.append((start_idx, end_idx))
            if i < len(words) - 1:
                # separator to represent space
                input_ids.append(self.tokenizer.sep_token_id)
                attention_mask.append(1)
                token_index += 1
        # final SEP
        input_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)

        # pad/truncate
        if len(input_ids) > config.MAX_LENGTH:
            input_ids = input_ids[:config.MAX_LENGTH]
            attention_mask = attention_mask[:config.MAX_LENGTH]
            # also need to trim spans that exceed length
            trimmed_spans: List[Tuple[int,int]] = []
            for s,e in word_token_spans:
                if s >= config.MAX_LENGTH:
                    break
                trimmed_spans.append((s, min(e, config.MAX_LENGTH-1)))
            word_token_spans = trimmed_spans
        else:
            while len(input_ids) < config.MAX_LENGTH:
                input_ids.append(self.tokenizer.pad_token_id)
                attention_mask.append(0)

        return input_ids, attention_mask, word_token_spans

    def predict_words(self, text: str) -> Tuple[List[int], List[List[float]]]:
        words = text.split()
        if not words:
            return [], []
        input_ids, attention_mask, word_spans = self._build_input_with_spans(words)
        input_ids_tensor = torch.tensor([input_ids]).to(self.device)
        attention_mask_tensor = torch.tensor([attention_mask]).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor
            )
        logits = outputs.logits[0]  # [seq_len, num_labels]
        probs = torch.softmax(logits, dim=-1)  # [seq_len, 2]

        word_predictions: List[int] = []
        word_probabilities: List[List[float]] = []
        pad_id = self.tokenizer.pad_token_id
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        for (start_idx, end_idx) in word_spans:
            if start_idx >= probs.shape[0]:
                # no tokens for this word due truncation
                word_predictions.append(0)
                word_probabilities.append([1.0, 0.0])
                continue
            valid_indices = []
            for ti in range(start_idx, min(end_idx+1, probs.shape[0])):
                tid = int(input_ids[ti])
                if tid in (pad_id, cls_id, sep_id):
                    continue
                valid_indices.append(ti)
            if not valid_indices:
                word_predictions.append(0)
                word_probabilities.append([1.0, 0.0])
                continue
            # Aggregate probabilities across subtokens: take max p(hate) for confidence and any-vote for label
            sub_probs = probs[valid_indices]  # [k, 2]
            p1_values = sub_probs[:,1]
            p1_max = float(torch.max(p1_values).item())
            p0 = 1.0 - p1_max
            # label = 1 if any subtoken argmax == 1 or if p1_max >= 0.5
            sub_preds = torch.argmax(sub_probs, dim=-1)
            label = 1 if (int(torch.any(sub_preds == 1)) == 1 or p1_max >= 0.5) else 0
            word_predictions.append(label)
            word_probabilities.append([p0, p1_max])

        return word_predictions, word_probabilities
    
    def detect_hate_speech(self, text: str) -> Dict:
        text = self.preprocess_text(text)
        word_predictions, word_probabilities = self.predict_words(text)
        words, word_starts, word_ends = self.tokenize_by_words(text)

        hate_words_entries = []
        hate_probabilities = []
        for i, (word, pred, prob) in enumerate(zip(words, word_predictions, word_probabilities)):
            if pred == 1:
                p1 = prob[1] if isinstance(prob, (list,tuple)) and len(prob) > 1 else (1.0 - prob if isinstance(prob, float) else 0.0)
                hate_words_entries.append({
                    'word': word,
                    'position': (word_starts[i], word_ends[i]),
                    'probability': p1
                })
                hate_probabilities.append(p1)

        dict_hate_words = []
        for word in words:
            wl = word.lower()
            if wl in self.hate_dict or word in self.hate_dict:
                dict_hate_words.append(word)

        sentence_label = "HATE" if (len(hate_words_entries) > 0 or len(dict_hate_words) > 0) else "NOT HATE"

        # Confidence logic: if model predicted hate words, use their mean p1; else if dictionary triggered, set fallback
        if len(hate_probabilities) > 0:
            confidence = float(np.mean(hate_probabilities))
        elif len(dict_hate_words) > 0:
            # dictionary triggered only – provide rule-based confidence
            confidence = 0.8
        else:
            confidence = 0.0

        result = {
            "sentence": text,
            "sentence_label": sentence_label,
            "confidence": confidence,
            "words": words,
            "word_predictions": word_predictions,
            "word_probabilities": word_probabilities,
            "hate_words": hate_words_entries,
            "dictionary_hate_words": dict_hate_words,
            "highlighted_hate_words": [hw['word'] for hw in hate_words_entries] + dict_hate_words
        }
        return result
    
    def batch_detect(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            results.append(self.detect_hate_speech(text))
        return results

if __name__ == "__main__":
    print("🧪 Testing XLM-RoBERTa hate speech detection inference with word-level tokens...")
    detector = HateSpeechDetectorXLM()
    test_sentences = [
        "මම ඔබට උදව් කරන්නම්",
        "ඔබ බොහෝ හොඳයි",
        "මෝඩයා එතන යන්න",
        "ඔබ ඉතා උගත් කෙනෙක්",
        "pakaya mokada",
    ]
    print("\nTesting hate speech detection:")
    print("=" * 50)
    for sentence in test_sentences:
        result = detector.detect_hate_speech(sentence)
        print(f"\nInput: {result['sentence']}")
        print(f"Words: {result['words']}")
        print(f"Classification: {result['sentence_label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        if result['highlighted_hate_words']:
            print(f"Hate words: {', '.join(result['highlighted_hate_words'])}")
        else:
            print("No hate words detected")
        print("-" * 30)
    print("\n✅ XLM-RoBERTa inference test with word-level tokens completed!")
