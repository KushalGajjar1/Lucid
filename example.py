#!/usr/bin/env python3
"""
Example usage of the Rust-based BPE Tokenizer
"""

from Lucid.Tokenizer import BPETokenizer
import time

def benchmark_tokenizer():
    """Benchmark the tokenizer performance"""
    print("🚀 Benchmarking Tokenizer Performance")
    print("=" * 50)
    
    # Create tokenizer
    tokenizer = BPETokenizer()
    
    # Sample text for training
    training_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.
    
    Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. The field has seen significant advances in recent years, particularly with the development of large language models and transformer architectures.
    
    BPE (Byte Pair Encoding) is a subword tokenization algorithm that is commonly used in modern NLP systems. It works by iteratively merging the most frequent pair of adjacent characters or tokens in the training data, building up a vocabulary of subword units.
    """ * 10  # Repeat text to make it longer
    
    print(f"Training text length: {len(training_text)} characters")
    
    # Train the tokenizer
    print("\n🔄 Training tokenizer...")
    start_time = time.time()
    tokenizer.train(training_text, vocab_size=1000)
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.3f} seconds")
    
    # Test encoding performance
    test_text = "Natural language processing with BPE tokenization is very efficient for modern NLP tasks."
    print(f"\n📝 Test text: {test_text}")
    
    print("\n🔄 Encoding text...")
    start_time = time.time()
    encoded = tokenizer.encode(test_text)
    encoding_time = time.time() - start_time
    print(f"✅ Encoding completed in {encoding_time:.6f} seconds")
    print(f"   Encoded to {len(encoded)} tokens: {encoded}")
    
    # Test decoding performance
    print("\n🔄 Decoding tokens...")
    start_time = time.time()
    decoded = tokenizer.decode(encoded)
    decoding_time = time.time() - start_time
    print(f"✅ Decoding completed in {decoding_time:.6f} seconds")
    print(f"   Decoded text: '{decoded}'")
    
    # Test special token handling
    special_text = "Hello <|endoftext|> world! This is a <|endoftext|> test."
    print(f"\n🔤 Test with special tokens: {special_text}")
    
    encoded_special = tokenizer.encode(special_text, allowed_special={"<|endoftext|>"})
    decoded_special = tokenizer.decode(encoded_special)
    print(f"   Encoded: {encoded_special}")
    print(f"   Decoded: '{decoded_special}'")
    
    # Performance summary
    print("\n📊 Performance Summary")
    print("=" * 30)
    print(f"Training time: {training_time:.3f}s")
    print(f"Encoding time: {encoding_time:.6f}s")
    print(f"Decoding time: {decoding_time:.6f}s")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"BPE merges: {len(tokenizer.bpe_merges)}")
    
    # Check implementation type
    if hasattr(tokenizer, '_rust_tokenizer'):
        print("🚀 Implementation: Rust (high performance)")
    else:
        print("🐍 Implementation: Python (fallback)")
    
    return tokenizer

def save_and_load_example(tokenizer):
    """Demonstrate save/load functionality"""
    print("\n💾 Save and Load Example")
    print("=" * 30)
    
    # Save the trained tokenizer
    vocab_path = "example_vocab.json"
    merges_path = "example_merges.json"
    
    print("💾 Saving tokenizer...")
    tokenizer.save_vocab_and_merges(vocab_path, merges_path)
    print("✅ Tokenizer saved successfully")
    
    # Create a new tokenizer and load the saved data
    print("📂 Loading tokenizer...")
    new_tokenizer = BPETokenizer()
    new_tokenizer.load_vocab_and_merges(vocab_path, merges_path)
    print("✅ Tokenizer loaded successfully")
    
    # Test that the loaded tokenizer works
    test_text = "Natural language processing"
    original_encoded = tokenizer.encode(test_text)
    loaded_encoded = new_tokenizer.encode(test_text)
    
    print(f"Original encoding: {original_encoded}")
    print(f"Loaded encoding: {loaded_encoded}")
    print(f"Encodings match: {original_encoded == loaded_encoded}")
    
    # Cleanup
    import os
    os.remove(vocab_path)
    os.remove(merges_path)
    print("🧹 Cleaned up example files")

def main():
    """Main example function"""
    print("🎯 Lucid Tokenizer - Rust Implementation Example")
    print("=" * 60)
    
    try:
        # Run benchmark
        tokenizer = benchmark_tokenizer()
        
        # Demonstrate save/load
        save_and_load_example(tokenizer)
        
        print("\n🎉 Example completed successfully!")
        print("\nYou can now use the tokenizer in your own projects:")
        print("  from Lucid.Tokenizer import BPETokenizer")
        print("  tokenizer = BPETokenizer()")
        print("  tokenizer.train(your_text, vocab_size=1000)")
        print("  tokens = tokenizer.encode('Your text here')")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
