use std::collections::{HashMap, HashSet, VecDeque};
use std::collections::hash_map::Entry;
use serde::{Deserialize, Serialize};
use regex::Regex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Character not found in vocabulary: {0:?}")]
    CharacterNotFound(Vec<char>),
    #[error("Token ID not found in vocabulary: {0}")]
    TokenIdNotFound(usize),
    #[error("Special token not found in vocabulary: {0}")]
    SpecialTokenNotFound(String),
    #[error("Disallowed special tokens encountered in text: {0:?}")]
    DisallowedSpecialTokens(Vec<String>),
    #[error("Invalid mode. Choose 'most' or 'least'")]
    InvalidMode,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BPETokenizer {
    /// Maps token id to token string
    vocab: HashMap<usize, String>,
    /// Maps token string to token id
    inverse_vocab: HashMap<String, usize>,
    /// Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
    bpe_merges: HashMap<(usize, usize), usize>,
}

impl BPETokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            inverse_vocab: HashMap::new(),
            bpe_merges: HashMap::new(),
        }
    }

    /// Train BPE Tokenizer
    /// 
    /// # Arguments
    /// * `text` - The text used to train the tokenizer
    /// * `vocab_size` - The vocabulary size
    /// * `allowed_special` - A set of included special tokens
    pub fn train(
        &mut self,
        text: &str,
        vocab_size: usize,
        allowed_special: Option<HashSet<String>>,
    ) -> Result<()> {
        let allowed_special = allowed_special.unwrap_or_else(|| {
            let mut set = HashSet::new();
            set.insert("<|endoftext|>".to_string());
            set
        });

        // Replace space with "Ġ"
        let mut processed_text = String::new();
        let mut prev_char = '\0';
        for (i, ch) in text.chars().enumerate() {
            if ch == ' ' && i != 0 {
                processed_text.push('Ġ');
            }
            if ch != ' ' {
                processed_text.push(ch);
            }
            prev_char = ch;
        }

        // Initialize vocab with unique characters
        let mut unique_chars: Vec<char> = (0..256).map(|i| i as u8 as char).collect();
        let text_chars: HashSet<char> = processed_text.chars().collect();
        for &ch in &text_chars {
            if !unique_chars.contains(&ch) {
                unique_chars.push(ch);
            }
        }
        if !unique_chars.contains(&'Ġ') {
            unique_chars.push('Ġ');
        }

        // Build vocabulary
        for (i, &ch) in unique_chars.iter().enumerate() {
            self.vocab.insert(i, ch.to_string());
            self.inverse_vocab.insert(ch.to_string(), i);
        }

        // Add special tokens
        for token in &allowed_special {
            if !self.inverse_vocab.contains_key(token) {
                let new_id = self.vocab.len();
                self.vocab.insert(new_id, token.clone());
                self.inverse_vocab.insert(token.clone(), new_id);
            }
        }

        // Tokenize the text
        let mut token_ids: Vec<usize> = processed_text
            .chars()
            .map(|ch| self.inverse_vocab[&ch.to_string()])
            .collect();

        // Find and Replace frequent pairs
        for new_id in self.vocab.len()..vocab_size {
            if let Some(pair_id) = Self::find_freq_pair(&token_ids, "most")? {
                token_ids = Self::replace_pair(&token_ids, pair_id, new_id);
                self.bpe_merges.insert(pair_id, new_id);
            } else {
                break;
            }
        }

        // Build the vocabulary with the merged tokens
        for (&(p0, p1), &new_id) in &self.bpe_merges {
            let merged_token = format!("{}{}", self.vocab[&p0], self.vocab[&p1]);
            self.vocab.insert(new_id, merged_token.clone());
            self.inverse_vocab.insert(merged_token, new_id);
        }

        Ok(())
    }

    /// Encode the input text into a list of token IDs
    /// 
    /// # Arguments
    /// * `text` - The input text to encode
    /// * `allowed_special` - Special tokens to allow passthrough
    /// 
    /// # Returns
    /// List of token IDs
    pub fn encode(
        &self,
        text: &str,
        allowed_special: Option<&HashSet<String>>,
    ) -> Result<Vec<usize>> {
        let mut token_ids = Vec::new();

        if let Some(allowed_special) = allowed_special {
            if !allowed_special.is_empty() {
                // Build regex to match allowed special tokens
                let special_pattern = format!(
                    "({})",
                    allowed_special
                        .iter()
                        .map(|tok| regex::escape(tok))
                        .collect::<Vec<_>>()
                        .join("|")
                );
                let regex = Regex::new(&special_pattern).unwrap();

                let mut last_index = 0;
                for cap in regex.find_iter(text) {
                    let prefix = &text[last_index..cap.start()];
                    // Encode prefix without special handling
                    token_ids.extend(self.encode(prefix, None)?);

                    let special_token = cap.as_str();
                    if let Some(&token_id) = self.inverse_vocab.get(special_token) {
                        token_ids.push(token_id);
                    } else {
                        return Err(TokenizerError::SpecialTokenNotFound(special_token.to_string()));
                    }
                    last_index = cap.end();
                }

                // Remaining part to process normally
                let remaining_text = &text[last_index..];

                // Check if any disallowed special tokens are in the remainder
                let disallowed: Vec<String> = self
                    .inverse_vocab
                    .keys()
                    .filter(|tok| {
                        tok.starts_with("<|") && tok.ends_with("|>") && 
                        remaining_text.contains(tok) && !allowed_special.contains(*tok)
                    })
                    .cloned()
                    .collect();

                if !disallowed.is_empty() {
                    return Err(TokenizerError::DisallowedSpecialTokens(disallowed));
                }

                // Process remaining text
                let remaining_tokens = self.tokenize_text(remaining_text);
                for token in remaining_tokens {
                    if let Some(&token_id) = self.inverse_vocab.get(&token) {
                        token_ids.push(token_id);
                    } else {
                        token_ids.extend(self.tokenize_with_bpe(&token)?);
                    }
                }

                return Ok(token_ids);
            }
        }

        // If no special tokens or remaining text after special token split
        let tokens = self.tokenize_text(text);
        for token in tokens {
            if let Some(&token_id) = self.inverse_vocab.get(&token) {
                token_ids.push(token_id);
            } else {
                token_ids.extend(self.tokenize_with_bpe(&token)?);
            }
        }

        Ok(token_ids)
    }

    /// Tokenize text into words with proper spacing
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let lines: Vec<&str> = text.split('\n').collect();
        
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                tokens.push("\n".to_string());
            }
            let words: Vec<&str> = line.split_whitespace().collect();
            for (j, word) in words.iter().enumerate() {
                if j == 0 && i > 0 {
                    tokens.push(format!("Ġ{}", word));
                } else if j == 0 {
                    tokens.push(word.to_string());
                } else {
                    tokens.push(format!("Ġ{}", word));
                }
            }
        }
        tokens
    }

    /// Tokenize a single token using BPE merges
    /// 
    /// # Arguments
    /// * `token` - The token to tokenize
    /// 
    /// # Returns
    /// The list of token IDs after applying BPE
    fn tokenize_with_bpe(&self, token: &str) -> Result<Vec<usize>> {
        // Tokenize the token into individual characters
        let mut token_ids: Vec<usize> = Vec::new();
        for ch in token.chars() {
            if let Some(&token_id) = self.inverse_vocab.get(&ch.to_string()) {
                token_ids.push(token_id);
            } else {
                return Err(TokenizerError::CharacterNotFound(vec![ch]));
            }
        }

        let mut can_merge = true;
        while can_merge && token_ids.len() > 1 {
            can_merge = false;
            let mut new_tokens = Vec::new();
            let mut i = 0;
            
            while i < token_ids.len() - 1 {
                let pair = (token_ids[i], token_ids[i + 1]);
                if let Some(&merged_token_id) = self.bpe_merges.get(&pair) {
                    new_tokens.push(merged_token_id);
                    i += 2;
                    can_merge = true;
                } else {
                    new_tokens.push(token_ids[i]);
                    i += 1;
                }
            }
            
            if i < token_ids.len() {
                new_tokens.push(token_ids[i]);
            }
            token_ids = new_tokens;
        }

        Ok(token_ids)
    }

    /// Decode a list of token IDs back into a string
    /// 
    /// # Arguments
    /// * `token_ids` - The list of token IDs to decode
    /// 
    /// # Returns
    /// The decoded string
    pub fn decode(&self, token_ids: &[usize]) -> Result<String> {
        let mut decoded_string = String::new();
        
        for (i, &token_id) in token_ids.iter().enumerate() {
            let token = self.vocab.get(&token_id)
                .ok_or(TokenizerError::TokenIdNotFound(token_id))?;
            
            if token == "\n" {
                if !decoded_string.is_empty() && !decoded_string.ends_with(' ') {
                    decoded_string.push(' ');
                }
                decoded_string.push_str(token);
            } else if token.starts_with('Ġ') {
                decoded_string.push(' ');
                decoded_string.push_str(&token[1..]);
            } else {
                decoded_string.push_str(token);
            }
        }

        Ok(decoded_string)
    }

    /// Save the vocabulary and BPE merges to JSON files
    /// 
    /// # Arguments
    /// * `vocab_path` - Path to save vocabulary
    /// * `bpe_merges_path` - Path to save the BPE merges
    pub fn save_vocab_and_merges(&self, vocab_path: &str, bpe_merges_path: &str) -> Result<()> {
        // Save vocabulary
        let vocab_json = serde_json::to_string_pretty(&self.vocab)?;
        std::fs::write(vocab_path, vocab_json)?;

        // Save BPE merges
        let merges_list: Vec<MergeEntry> = self
            .bpe_merges
            .iter()
            .map(|(&(p0, p1), &new_id)| MergeEntry {
                pair: vec![p0, p1],
                new_id,
            })
            .collect();
        let merges_json = serde_json::to_string_pretty(&merges_list)?;
        std::fs::write(bpe_merges_path, merges_json)?;

        Ok(())
    }

    /// Load the vocabulary and BPE merges from JSON files
    /// 
    /// # Arguments
    /// * `vocab_path` - Path to the vocabulary file
    /// * `bpe_merges_path` - Path to the BPE merges file
    pub fn load_vocab_and_merges(&mut self, vocab_path: &str, bpe_merges_path: &str) -> Result<()> {
        // Load vocabulary
        let vocab_content = std::fs::read_to_string(vocab_path)?;
        let loaded_vocab: HashMap<String, String> = serde_json::from_str(&vocab_content)?;
        self.vocab = loaded_vocab
            .into_iter()
            .map(|(k, v)| (k.parse::<usize>().unwrap(), v))
            .collect();
        self.inverse_vocab = self.vocab
            .iter()
            .map(|(&k, v)| (v.clone(), k))
            .collect();

        // Load BPE merges
        let merges_content = std::fs::read_to_string(bpe_merges_path)?;
        let merges_list: Vec<MergeEntry> = serde_json::from_str(&merges_content)?;
        for merge in merges_list {
            if merge.pair.len() == 2 {
                let pair = (merge.pair[0], merge.pair[1]);
                self.bpe_merges.insert(pair, merge.new_id);
            }
        }

        Ok(())
    }

    /// Find the most or least frequent pair in token IDs
    /// 
    /// # Arguments
    /// * `token_ids` - List of token IDs
    /// * `mode` - "most" or "least" frequent
    /// 
    /// # Returns
    /// The most/least frequent pair or None if no pairs exist
    fn find_freq_pair(token_ids: &[usize], mode: &str) -> Result<Option<(usize, usize)>> {
        if token_ids.len() < 2 {
            return Ok(None);
        }

        let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for window in token_ids.windows(2) {
            let pair = (window[0], window[1]);
            *pair_counts.entry(pair).or_insert(0) += 1;
        }

        if pair_counts.is_empty() {
            return Ok(None);
        }

        let result = match mode {
            "most" => pair_counts.into_iter().max_by_key(|&(_, count)| count),
            "least" => pair_counts.into_iter().min_by_key(|&(_, count)| count),
            _ => return Err(TokenizerError::InvalidMode),
        };

        Ok(result.map(|(pair, _)| pair))
    }

    /// Replace all occurrences of a pair with a new token ID
    /// 
    /// # Arguments
    /// * `token_ids` - List of token IDs
    /// * `pair_id` - The pair to replace
    /// * `new_id` - The new token ID to insert
    /// 
    /// # Returns
    /// New list with pairs replaced
    fn replace_pair(token_ids: &[usize], pair_id: (usize, usize), new_id: usize) -> Vec<usize> {
        let mut dq: VecDeque<usize> = token_ids.iter().cloned().collect();
        let mut replaced = Vec::new();

        while let Some(current) = dq.pop_front() {
            if let Some(&next) = dq.front() {
                if (current, next) == pair_id {
                    replaced.push(new_id);
                    dq.pop_front(); // Remove the next element
                } else {
                    replaced.push(current);
                }
            } else {
                replaced.push(current);
            }
        }

        replaced
    }

    /// Get the current vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the current BPE merges count
    pub fn merges_count(&self) -> usize {
        self.bpe_merges.len()
    }
}

impl Default for BPETokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct MergeEntry {
    pair: Vec<usize>,
    new_id: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenizer() {
        let mut tokenizer = BPETokenizer::new();
        let text = "hello world";
        
        tokenizer.train(text, 100, None).unwrap();
        assert!(tokenizer.vocab_size() > 0);
        
        let encoded = tokenizer.encode(text, None).unwrap();
        assert!(!encoded.is_empty());
        
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_special_tokens() {
        let mut tokenizer = BPETokenizer::new();
        let text = "hello <|endoftext|> world";
        let special_tokens: HashSet<String> = ["<|endoftext|>".to_string()].into_iter().collect();
        
        tokenizer.train(text, 100, Some(special_tokens.clone())).unwrap();
        
        let encoded = tokenizer.encode(text, Some(&special_tokens)).unwrap();
        assert!(!encoded.is_empty());
        
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, "hello <|endoftext|> world");
    }
} 