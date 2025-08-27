"""
Configuration for Sinhala/Singlish Hate Speech Detection with XLM-RoBERTa
Uses word-level tokenization by spaces (matching SOLD dataset format)
"""

import os
import datetime

# Model Configuration
MODEL_NAME = "xlm-roberta-base"
MODEL_SAVE_PATH = "outputs_xlm_roberta/model"
RESULTS_SAVE_PATH = "outputs_xlm_roberta/results"

# Data Configuration
TRAIN_FILE = "datsets/SOLD_train_cleaned.json"
TEST_FILE = "datsets/SOLD_test_cleaned.json"
HATE_DICT_FILE = "datsets/only_hate.csv"

# Feedback Configuration
FEEDBACK_DIR = "feedback"
FEEDBACK_CSV = os.path.join(FEEDBACK_DIR, "user_flags.csv")
RETRAIN_OUTPUT_BASE = "outputs_xlm_roberta/fine_tuned"

# Training Configuration
MAX_LENGTH = 256  # Maximum sequence length for word-level tokens
BATCH_SIZE = 16   # Reduced batch size for word-level processing
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# Model Architecture
NUM_LABELS = 2  # 0: Neutral, 1: Hate

# Loss Function Configuration
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25  # Weight for positive class
FOCAL_GAMMA = 2.0   # Focusing parameter

# Class Imbalance Handling
USE_CLASS_WEIGHTS = True

# Development/Testing
USE_SAMPLE_DATA = False  # Set to True for quick testing with subset
SAMPLE_SIZE = 1000       # Number of samples to use when USE_SAMPLE_DATA is True

# Tokenization Strategy
TOKENIZATION_TYPE = "word_level_by_spaces"  # New: word-level tokenization by spaces

# Output Configuration
SAVE_BEST_MODEL = True
SAVE_TRAINING_HISTORY = True
SAVE_PREDICTIONS = True

# Logging
VERBOSE = True
SAVE_LOGS = True
LOG_LEVEL = "INFO"

# Performance
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 5

# Validation
EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1_class_1"  # Focus on hate detection F1 score

# Data Processing
REMOVE_SPECIAL_CHARS = False  # Keep special characters for word-level processing
NORMALIZE_TEXT = False        # Keep original text format
LOWERCASE = False             # Keep original case for word matching

# Model Checkpointing
SAVE_TOTAL_LIMIT = 3         # Keep only 3 best checkpoints
SAVE_STEPS = 500             # Save every 500 steps
EVAL_STEPS = 500             # Evaluate every 500 steps

# Hardware
USE_MIXED_PRECISION = False  # Set to True if using GPU with mixed precision support
FP16 = False                 # 16-bit precision training

# Reproducibility
SEED = 42
DETERMINISTIC = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datsets")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_xlm_roberta")

# Ensure output and feedback directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Print configuration summary
if __name__ == "__main__":
    print("🔧 XLM-RoBERTa Configuration Summary")
    print("=" * 50)
    print(f"Model: {MODEL_NAME}")
    print(f"Tokenization: {TOKENIZATION_TYPE}")
    print(f"Max Length: {MAX_LENGTH}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Focal Loss: {USE_FOCAL_LOSS}")
    print(f"Class Weights: {USE_CLASS_WEIGHTS}")
    print(f"Output Path: {MODEL_SAVE_PATH}")
    print(f"Feedback CSV: {FEEDBACK_CSV}")
    print("=" * 50)
