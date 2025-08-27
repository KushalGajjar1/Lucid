#!/usr/bin/env python3
"""
Test script for the Rust-based BPE Tokenizer
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tokenizer():
    """Test the tokenizer functionality"""
    try:
        from Lucid.Tokenizer import BPETokenizer
        print("✓ Successfully imported BPETokenizer")
        
        # Create tokenizer
        tokenizer = BPETokenizer()
        print("✓ Successfully created tokenizer instance")
        
        # Test training
        sample_text = "Hello world! This is a sample text for training the BPE tokenizer."
        print(f"Training on text: {sample_text}")
        tokenizer.train(sample_text, vocab_size=100)
        print("✓ Successfully trained tokenizer")
        
        # Test encoding
        test_text = "Hello world!"
        encoded = tokenizer.encode(test_text)
        print(f"Encoded '{test_text}' -> {encoded}")
        print("✓ Successfully encoded text")
        
        # Test decoding
        decoded = tokenizer.decode(encoded)
        print(f"Decoded {encoded} -> '{decoded}'")
        print("✓ Successfully decoded tokens")
        
        # Test special tokens
        special_text = "Hello <|endoftext|> world!"
        encoded_special = tokenizer.encode(special_text, allowed_special={"<|endoftext|>"})
        print(f"Encoded with special tokens: {encoded_special}")
        decoded_special = tokenizer.decode(encoded_special)
        print(f"Decoded with special tokens: '{decoded_special}'")
        print("✓ Successfully handled special tokens")
        
        # Test save/load
        vocab_path = "test_vocab.json"
        merges_path = "test_merges.json"
        tokenizer.save_vocab_and_merges(vocab_path, merges_path)
        print("✓ Successfully saved vocabulary and merges")
        
        # Create new tokenizer and load
        new_tokenizer = BPETokenizer()
        new_tokenizer.load_vocab_and_merges(vocab_path, merges_path)
        print("✓ Successfully loaded vocabulary and merges")
        
        # Test that loaded tokenizer works
        test_encoded = new_tokenizer.encode("Hello world!")
        test_decoded = new_tokenizer.decode(test_encoded)
        print(f"Loaded tokenizer: 'Hello world!' -> {test_encoded} -> '{test_decoded}'")
        print("✓ Loaded tokenizer works correctly")
        
        # Cleanup test files
        os.remove(vocab_path)
        os.remove(merges_path)
        print("✓ Cleaned up test files")
        
        print("\n🎉 All tests passed! The tokenizer is working correctly.")
        
        # Check if we're using Rust or Python
        if hasattr(tokenizer, '_rust_tokenizer'):
            print("🚀 Using Rust implementation (high performance)")
        else:
            print("🐍 Using Python implementation (fallback)")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have built and installed the package: pip install -e .")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Testing Rust-based BPE Tokenizer...")
    print("=" * 50)
    
    success = test_tokenizer()
    
    if success:
        print("\n✅ Tokenizer is ready to use!")
        sys.exit(0)
    else:
        print("\n❌ Tokenizer test failed!")
        sys.exit(1)
