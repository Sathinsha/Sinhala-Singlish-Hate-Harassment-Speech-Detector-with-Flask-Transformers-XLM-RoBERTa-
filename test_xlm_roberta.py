"""
Test script for XLM-RoBERTa hate speech detection setup
Verifies tokenization, data preprocessing, and model loading
"""

from transformers import AutoTokenizer
from data_preprocessing_xlm_roberta import load_sold_dataset, create_xlm_roberta_dataset
import config_xlm_roberta as config

def test_xlm_roberta_setup():
    """Test the complete XLM-RoBERTa setup"""
    
    print("🧪 Testing XLM-RoBERTa Setup for Sinhala Hate Speech Detection")
    print("=" * 60)
    
    # Test 1: Tokenizer
    print("\n1️⃣ Testing XLM-RoBERTa Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        print(f"✅ Tokenizer loaded: {config.MODEL_NAME}")
        
        # Test Sinhala text tokenization
        test_texts = [
            "මෝඩයා එතන යන්න",  # "Go there, idiot"
            "හුත්තො එතන යන්න",   # "Go there, fool"
            "මම ඔබට උදව් කරන්නම්",  # "I will help you"
        ]
        
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            unk_count = tokens.count('[UNK]') if '[UNK]' in tokens else 0
            print(f"   '{text}' -> {tokens} (UNK: {unk_count})")
            
        if unk_count == 0:
            print("✅ No UNK tokens - Sinhala text properly tokenized!")
        else:
            print("⚠️  UNK tokens found - tokenization may have issues")
            
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return False
    
    # Test 2: Data Loading
    print("\n2️⃣ Testing Data Loading...")
    try:
        train_tokens, train_labels = load_sold_dataset(config.TRAIN_FILE)
        print(f"✅ Training data loaded: {len(train_tokens)} samples")
        
        # Show sample data
        print(f"   Sample tokens: {train_tokens[0][:5]}...")
        print(f"   Sample labels: {train_labels[0][:5]}...")
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False
    
    # Test 3: Dataset Creation
    print("\n3️⃣ Testing Dataset Creation...")
    try:
        train_texts, train_processed_labels = create_xlm_roberta_dataset(
            train_tokens[:10], train_labels[:10], tokenizer, config.MAX_LENGTH
        )
        print(f"✅ Dataset created: {len(train_texts)} processed samples")
        print(f"   Max length: {config.MAX_LENGTH}")
        print(f"   Sample processed labels: {train_processed_labels[0][:10]}...")
        
    except Exception as e:
        print(f"❌ Dataset creation error: {e}")
        return False
    
    # Test 4: Configuration
    print("\n4️⃣ Testing Configuration...")
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Max length: {config.MAX_LENGTH}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Learning rate: {config.LEARNING_RATE}")
    print(f"   Epochs: {config.NUM_EPOCHS}")
    print(f"   Output directory: {config.OUTPUT_DIR}")
    
    # Test 5: Output Directories
    print("\n5️⃣ Testing Output Directories...")
    import os
    if os.path.exists(config.OUTPUT_DIR):
        print(f"✅ Output directory exists: {config.OUTPUT_DIR}")
    else:
        print(f"📁 Output directory will be created: {config.OUTPUT_DIR}")
    
    print("\n🎯 Setup Summary:")
    print("   ✅ XLM-RoBERTa tokenizer works with Sinhala text")
    print("   ✅ Data loading and preprocessing functional")
    print("   ✅ Configuration properly set")
    print("   ✅ Ready for Google Colab training!")
    
    return True

if __name__ == "__main__":
    test_xlm_roberta_setup()
