use lucid_tokenizer::BPETokenizer;
use std::collections::HashSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BPE Tokenizer Example ===\n");

    // Create a new tokenizer
    let mut tokenizer = BPETokenizer::new();
    
    // Training text
    let training_text = "The quick brown fox jumps over the lazy dog. This is a sample text for training the BPE tokenizer. We will use this to demonstrate encoding and decoding capabilities.";
    
    println!("Training text: {}", training_text);
    println!("Training tokenizer with vocab size 200...\n");
    
    // Train the tokenizer
    tokenizer.train(training_text, 200, None)?;
    
    println!("Training completed!");
    println!("Vocabulary size: {}", tokenizer.vocab_size());
    println!("BPE merges count: {}\n", tokenizer.merges_count());
    
    // Test encoding and decoding
    let test_text = "The quick brown fox jumps over the lazy dog.";
    println!("Test text: {}", test_text);
    
    // Encode the text
    let encoded = tokenizer.encode(test_text, None)?;
    println!("Encoded tokens: {:?}", encoded);
    println!("Number of tokens: {}\n", encoded.len());
    
    // Decode the tokens back to text
    let decoded = tokenizer.decode(&encoded)?;
    println!("Decoded text: {}", decoded);
    println!("Text matches: {}\n", test_text == decoded);
    
    // Test with special tokens
    println!("=== Testing with Special Tokens ===\n");
    
    let mut tokenizer_with_special = BPETokenizer::new();
    let special_tokens: HashSet<String> = [
        "<|endoftext|>".to_string(),
        "<|startoftext|>".to_string(),
        "<|pad|>".to_string(),
    ].into_iter().collect();
    
    let special_text = "Hello <|startoftext|> world <|endoftext|> goodbye <|pad|>";
    println!("Text with special tokens: {}", special_text);
    
    tokenizer_with_special.train(special_text, 150, Some(special_tokens.clone()))?;
    
    let encoded_special = tokenizer_with_special.encode(special_text, Some(&special_tokens))?;
    println!("Encoded with special tokens: {:?}", encoded_special);
    
    let decoded_special = tokenizer_with_special.decode(&encoded_special)?;
    println!("Decoded with special tokens: {}", decoded_special);
    println!("Text matches: {}\n", special_text == decoded_special);
    
    // Test saving and loading
    println!("=== Testing Save/Load Functionality ===\n");
    
    let vocab_path = "vocab.json";
    let merges_path = "bpe_merges.json";
    
    // Save the trained tokenizer
    tokenizer.save_vocab_and_merges(vocab_path, merges_path)?;
    println!("Tokenizer saved to {} and {}", vocab_path, merges_path);
    
    // Create a new tokenizer and load the saved data
    let mut loaded_tokenizer = BPETokenizer::new();
    loaded_tokenizer.load_vocab_and_merges(vocab_path, merges_path)?;
    
    println!("Tokenizer loaded from files");
    println!("Loaded vocabulary size: {}", loaded_tokenizer.vocab_size());
    println!("Loaded BPE merges count: {}\n", loaded_tokenizer.merges_count());
    
    // Test that the loaded tokenizer works the same
    let test_loaded = "The quick brown fox";
    let encoded_loaded = loaded_tokenizer.encode(test_loaded, None)?;
    let decoded_loaded = loaded_tokenizer.decode(&encoded_loaded)?;
    
    println!("Test with loaded tokenizer:");
    println!("Input: {}", test_loaded);
    println!("Output: {}", decoded_loaded);
    println!("Matches: {}\n", test_loaded == decoded_loaded);
    
    // Clean up temporary files
    std::fs::remove_file(vocab_path)?;
    std::fs::remove_file(merges_path)?;
    println!("Temporary files cleaned up");
    
    println!("=== Example completed successfully! ===");
    Ok(())
} 