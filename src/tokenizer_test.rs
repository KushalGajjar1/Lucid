#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;

    #[test]
    fn test_new_tokenizer() {
        let tokenizer = BPETokenizer::new();
        assert_eq!(tokenizer.vocab_size(), 0);
        assert_eq!(tokenizer.merges_count(), 0);
    }

    #[test]
    fn test_basic_training() {
        let mut tokenizer = BPETokenizer::new();
        let text = "hello world";
        
        tokenizer.train(text, 100, None).unwrap();
        assert!(tokenizer.vocab_size() > 0);
        assert!(tokenizer.merges_count() > 0);
    }

    #[test]
    fn test_training_with_special_tokens() {
        let mut tokenizer = BPETokenizer::new();
        let text = "hello <|endoftext|> world";
        let special_tokens: HashSet<String> = ["<|endoftext|>".to_string()].into_iter().collect();
        
        tokenizer.train(text, 100, Some(special_tokens.clone())).unwrap();
        assert!(tokenizer.vocab_size() > 0);
        
        // Check that special token is in vocabulary
        assert!(tokenizer.inverse_vocab.contains_key("<|endoftext|>"));
    }

    #[test]
    fn test_encoding_decoding_roundtrip() {
        let mut tokenizer = BPETokenizer::new();
        let text = "The quick brown fox jumps over the lazy dog.";
        
        tokenizer.train(text, 200, None).unwrap();
        
        let encoded = tokenizer.encode(text, None).unwrap();
        assert!(!encoded.is_empty());
        
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encoding_with_special_tokens() {
        let mut tokenizer = BPETokenizer::new();
        let text = "Hello <|startoftext|> world <|endoftext|> goodbye";
        let special_tokens: HashSet<String> = [
            "<|startoftext|>".to_string(),
            "<|endoftext|>".to_string(),
        ].into_iter().collect();
        
        tokenizer.train(text, 150, Some(special_tokens.clone())).unwrap();
        
        let encoded = tokenizer.encode(text, Some(&special_tokens)).unwrap();
        assert!(!encoded.is_empty());
        
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encoding_without_special_tokens() {
        let mut tokenizer = BPETokenizer::new();
        let text = "Hello <|startoftext|> world <|endoftext|> goodbye";
        let special_tokens: HashSet<String> = [
            "<|startoftext|>".to_string(),
            "<|endoftext|>".to_string(),
        ].into_iter().collect();
        
        tokenizer.train(text, 150, Some(special_tokens.clone())).unwrap();
        
        // Encode without special token handling - should treat special tokens as regular text
        let encoded = tokenizer.encode(text, None).unwrap();
        assert!(!encoded.is_empty());
        
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_special_token_not_found() {
        let mut tokenizer = BPETokenizer::new();
        let text = "Hello world";
        tokenizer.train(text, 100, None).unwrap();
        
        let special_tokens: HashSet<String> = ["<|nonexistent|>".to_string()].into_iter().collect();
        
        let result = tokenizer.encode(text, Some(&special_tokens));
        assert!(result.is_err());
        
        if let Err(TokenizerError::SpecialTokenNotFound(token)) = result {
            assert_eq!(token, "<|nonexistent|>");
        } else {
            panic!("Expected SpecialTokenNotFound error");
        }
    }

    #[test]
    fn test_disallowed_special_tokens() {
        let mut tokenizer = BPETokenizer::new();
        let text = "Hello <|startoftext|> world <|endoftext|> goodbye";
        let special_tokens: HashSet<String> = [
            "<|startoftext|>".to_string(),
        ].into_iter().collect();
        
        tokenizer.train(text, 150, Some(special_tokens.clone())).unwrap();
        
        // Try to encode with only startoftext allowed, but text contains endoftext
        let result = tokenizer.encode(text, Some(&special_tokens));
        assert!(result.is_err());
        
        if let Err(TokenizerError::DisallowedSpecialTokens(tokens)) = result {
            assert!(tokens.contains(&"<|endoftext|>".to_string()));
        } else {
            panic!("Expected DisallowedSpecialTokens error");
        }
    }

    #[test]
    fn test_character_not_found() {
        let mut tokenizer = BPETokenizer::new();
        let text = "hello world";
        tokenizer.train(text, 100, None).unwrap();
        
        // Try to encode text with a character not in the vocabulary
        let text_with_unknown = "hello world 🚀";
        let result = tokenizer.encode(text_with_unknown, None);
        assert!(result.is_err());
        
        if let Err(TokenizerError::CharacterNotFound(chars)) = result {
            assert!(chars.contains(&'🚀'));
        } else {
            panic!("Expected CharacterNotFound error");
        }
    }

    #[test]
    fn test_token_id_not_found() {
        let mut tokenizer = BPETokenizer::new();
        let text = "hello world";
        tokenizer.train(text, 100, None).unwrap();
        
        // Try to decode with a token ID not in the vocabulary
        let invalid_token_ids = vec![9999];
        let result = tokenizer.decode(&invalid_token_ids);
        assert!(result.is_err());
        
        if let Err(TokenizerError::TokenIdNotFound(id)) = result {
            assert_eq!(id, 9999);
        } else {
            panic!("Expected TokenIdNotFound error");
        }
    }

    #[test]
    fn test_save_and_load() {
        let mut tokenizer = BPETokenizer::new();
        let text = "The quick brown fox jumps over the lazy dog.";
        
        tokenizer.train(text, 200, None).unwrap();
        let original_vocab_size = tokenizer.vocab_size();
        let original_merges_count = tokenizer.merges_count();
        
        // Save tokenizer
        let vocab_path = "test_vocab.json";
        let merges_path = "test_merges.json";
        tokenizer.save_vocab_and_merges(vocab_path, merges_path).unwrap();
        
        // Create new tokenizer and load
        let mut loaded_tokenizer = BPETokenizer::new();
        loaded_tokenizer.load_vocab_and_merges(vocab_path, merges_path).unwrap();
        
        // Verify loaded data
        assert_eq!(loaded_tokenizer.vocab_size(), original_vocab_size);
        assert_eq!(loaded_tokenizer.merges_count(), original_merges_count);
        
        // Test functionality
        let test_text = "The quick brown fox";
        let original_encoded = tokenizer.encode(test_text, None).unwrap();
        let loaded_encoded = loaded_tokenizer.encode(test_text, None).unwrap();
        assert_eq!(original_encoded, loaded_encoded);
        
        // Clean up
        fs::remove_file(vocab_path).unwrap();
        fs::remove_file(merges_path).unwrap();
    }

    #[test]
    fn test_find_freq_pair() {
        let token_ids = vec![1, 2, 1, 2, 1, 3, 2, 1];
        
        // Test most frequent
        let most_freq = BPETokenizer::find_freq_pair(&token_ids, "most").unwrap();
        assert_eq!(most_freq, Some((1, 2))); // (1,2) appears 3 times
        
        // Test least frequent
        let least_freq = BPETokenizer::find_freq_pair(&token_ids, "least").unwrap();
        assert_eq!(least_freq, Some((3, 2))); // (3,2) appears 1 time
        
        // Test invalid mode
        let result = BPETokenizer::find_freq_pair(&token_ids, "invalid");
        assert!(result.is_err());
        
        if let Err(TokenizerError::InvalidMode) = result {
            // Expected error
        } else {
            panic!("Expected InvalidMode error");
        }
    }

    #[test]
    fn test_replace_pair() {
        let token_ids = vec![1, 2, 1, 2, 3];
        let pair = (1, 2);
        let new_id = 99;
        
        let replaced = BPETokenizer::replace_pair(&token_ids, pair, new_id);
        assert_eq!(replaced, vec![99, 99, 3]);
    }

    #[test]
    fn test_empty_text() {
        let mut tokenizer = BPETokenizer::new();
        let text = "";
        
        tokenizer.train(text, 100, None).unwrap();
        let encoded = tokenizer.encode(text, None).unwrap();
        assert_eq!(encoded, vec![]);
        
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_single_character() {
        let mut tokenizer = BPETokenizer::new();
        let text = "a";
        
        tokenizer.train(text, 100, None).unwrap();
        let encoded = tokenizer.encode(text, None).unwrap();
        assert_eq!(encoded.len(), 1);
        
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, "a");
    }

    #[test]
    fn test_multiline_text() {
        let mut tokenizer = BPETokenizer::new();
        let text = "Hello\nworld\nthis is a test";
        
        tokenizer.train(text, 200, None).unwrap();
        let encoded = tokenizer.encode(text, None).unwrap();
        assert!(!encoded.is_empty());
        
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_unicode_characters() {
        let mut tokenizer = BPETokenizer::new();
        let text = "Hello 世界 🌍";
        
        tokenizer.train(text, 200, None).unwrap();
        let encoded = tokenizer.encode(text, None).unwrap();
        assert!(!encoded.is_empty());
        
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_vocab_size_limit() {
        let mut tokenizer = BPETokenizer::new();
        let text = "a b c d e f g h i j k l m n o p q r s t u v w x y z";
        
        // Train with very small vocab size
        tokenizer.train(text, 10, None).unwrap();
        assert!(tokenizer.vocab_size() <= 10);
    }

    #[test]
    fn test_default_implementation() {
        let tokenizer = BPETokenizer::default();
        assert_eq!(tokenizer.vocab_size(), 0);
        assert_eq!(tokenizer.merges_count(), 0);
    }
} 