"""
Training module for Sinhala/Singlish Hate Speech Detection with XLM-RoBERTa
Uses word-level tokenization by spaces (matching SOLD dataset format)
"""

import os
import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import config_xlm_roberta as config
from data_preprocessing_xlm_roberta import (
    load_sold_dataset,
    create_word_level_dataset,
    prepare_word_level_loaders,
    load_hate_dictionary
)

class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class HateSpeechTrainerXLM:
    """Improved trainer class for hate speech detection with word-level tokens"""
    
    def __init__(self, model_name, num_labels, max_length, device, use_focal_loss=True):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device
        self.use_focal_loss = use_focal_loss
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Move to device
        self.model.to(device)
        
        print(f"Model initialized on {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        else:
            self.criterion = None  # Use default cross-entropy
    
    def train_epoch(self, train_loader, optimizer, scheduler, class_weights=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training", 
                           unit="batch", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Use custom loss if specified
            if self.use_focal_loss and self.criterion is not None:
                # Reshape for focal loss
                logits = outputs.logits.view(-1, self.num_labels)
                labels_flat = labels.view(-1)
                
                # Filter out padding tokens (-100)
                valid_mask = labels_flat != -100
                if valid_mask.any():
                    logits_valid = logits[valid_mask]
                    labels_valid = labels_flat[valid_mask]
                    loss = self.criterion(logits_valid, labels_valid)
                else:
                    loss = outputs.loss
            else:
                loss = outputs.loss
            
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (progress_bar.n + 1):.4f}'
            })
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                # Collect predictions and labels (ignore padding tokens)
                for i in range(labels.shape[0]):
                    valid_mask = labels[i] != -100
                    if valid_mask.any():
                        all_predictions.extend(predictions[i][valid_mask].cpu().numpy())
                        all_labels.extend(labels[i][valid_mask].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        
        # Calculate F1 scores
        if len(all_predictions) > 0 and len(all_labels) > 0:
            token_f1 = f1_score(all_labels, all_predictions, average='weighted')
            
            # Calculate per-class F1
            f1_class_0 = f1_score(all_labels, all_predictions, pos_label=0, zero_division=0)
            f1_class_1 = f1_score(all_labels, all_predictions, pos_label=1, zero_division=0)
            
            print(f"Class 0 (Neutral) F1: {f1_class_0:.4f}")
            print(f"Class 1 (Hate) F1: {f1_class_1:.4f}")
        else:
            token_f1 = 0.0
            f1_class_0 = 0.0
            f1_class_1 = 0.0
        
        return {
            'loss': avg_loss,
            'token_f1': token_f1,
            'f1_class_0': f1_class_0,
            'f1_class_1': f1_class_1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def save_model(self, save_path):
        """Save model and tokenizer"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Model and tokenizer saved to {save_path}")
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate, 
              warmup_steps, weight_decay, class_weights=None):
        """Main training loop with improved monitoring"""
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_f1_class_0': [],
            'val_f1_class_1': []
        }
        
        best_f1 = 0
        patience = 5  # Increased patience
        no_improve_count = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Early stopping patience: {patience}")
        print(f"Using focal loss: {self.use_focal_loss}")
        print(f"Using word-level tokenization (by spaces)")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, class_weights)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['token_f1'])
            history['val_f1_class_0'].append(val_metrics['f1_class_0'])
            history['val_f1_class_1'].append(val_metrics['f1_class_1'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Token F1: {val_metrics['token_f1']:.4f}")
            
            # Save best model based on hate class F1 (more important)
            current_f1 = val_metrics['f1_class_1']  # Focus on hate detection
            if current_f1 > best_f1:
                best_f1 = current_f1
                no_improve_count = 0
                print(f"New best hate F1 score: {best_f1:.4f}")
                self.save_model(config.MODEL_SAVE_PATH)
            else:
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} epochs")
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
            print("-" * 50)
        
        # Save training history
        with open(os.path.join(config.RESULTS_SAVE_PATH, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training completed! Best hate F1 score: {best_f1:.4f}")
        return history

def calculate_class_weights(train_labels):
    """Calculate class weights to handle imbalance"""
    all_labels = []
    for labels in train_labels:
        all_labels.extend([l for l in labels if l != -100])
    
    # Count occurrences
    unique, counts = np.unique(all_labels, return_counts=True)
    total = len(all_labels)
    
    # Calculate weights (inverse frequency)
    weights = {}
    for label, count in zip(unique, counts):
        weights[label] = total / (len(unique) * count)
    
    print(f"Class weights: {weights}")
    return weights

def main():
    """Main training function"""
    print("🚀 Starting Sinhala/Singlish Hate Speech Detection Training with XLM-RoBERTa")
    print("📝 Using WORD-LEVEL tokenization by spaces (matching SOLD dataset format)")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading datasets...")
    train_tokens, train_labels = load_sold_dataset(config.TRAIN_FILE)
    test_tokens, test_labels = load_sold_dataset(config.TEST_FILE)
    
    # Load hate dictionary
    hate_dict = load_hate_dictionary(config.HATE_DICT_FILE)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create word-level dataset (by spaces)
    print("Creating word-level dataset (tokenization by spaces)...")
    train_texts, train_processed_labels = create_word_level_dataset(
        train_tokens, train_labels, config.MAX_LENGTH
    )
    
    # Split into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_processed_labels,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED
    )
    
    # Option to use sample data for faster training/testing
    if config.USE_SAMPLE_DATA:
        print(f"⚠️  Using sample data: {config.SAMPLE_SIZE} samples for quick testing")
        # Take only first SAMPLE_SIZE samples
        train_texts = train_texts[:config.SAMPLE_SIZE]
        train_labels = train_labels[:config.SAMPLE_SIZE]
        val_texts = val_texts[:config.SAMPLE_SIZE//10]  # 10% of sample size
        val_labels = val_labels[:config.SAMPLE_SIZE//10]
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_tokens)}")
    
    # Calculate class weights if enabled
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        print("Calculating class weights to handle imbalance...")
        class_weights = calculate_class_weights(train_labels)
    
    # Prepare data loaders with word-level tokenization
    train_loader, val_loader = prepare_word_level_loaders(
        train_texts, train_labels, val_texts, val_labels, 
        tokenizer, config.BATCH_SIZE
    )
    
    # Initialize trainer
    trainer = HateSpeechTrainerXLM(
        model_name=config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        max_length=config.MAX_LENGTH,
        device=device,
        use_focal_loss=config.USE_FOCAL_LOSS
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        class_weights=class_weights
    )
    
    print("✅ Training completed successfully!")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"Results saved to: {config.RESULTS_SAVE_PATH}")

if __name__ == "__main__":
    main()
