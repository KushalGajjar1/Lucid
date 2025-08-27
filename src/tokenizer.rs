use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyDict};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use regex::Regex;
use crate::error::{Result, TokenizerError};

#[derive(Serialize, Deserialize)]
struct BpeMerge {
    pair: Vec<u32>,
    new_id: u32,
}

#[pyclass]
pub struct BPETokenizer {
    #[pyo3(get)]
    vocab: HashMap<u32, String>,
    #[pyo3(get)]
    inverse_vocab: HashMap<String, u32>,
    #[pyo3(get)]
    bpe_merges: HashMap<(u32, u32), u32>,
}

#[pymethods]
impl BPETokenizer {
    #[new]
    fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            inverse_vocab: HashMap::new(),
            bpe_merges: HashMap::new(),
        }
    }

    fn train(&mut self, text: &str, vocab_size: usize, allowed_special: Option<Vec<String>>) -> PyResult<()> {
        let allowed_special = allowed_special.unwrap_or_else(|| vec!["<|endoftext|>".to_string()]);
        
        // Replace space with "Ġ"
        let mut processed_text = String::new();
        let chars: Vec<char> = text.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            if ch == ' ' && i != 0 {
                processed_text.push('Ġ');
            }
            if ch != ' ' {
                processed_text.push(ch);
            }
        }

        // Initialize vocab with unique characters
        let mut unique_chars: Vec<char> = (0..256).map(|i| char::from_u32(i).unwrap()).collect();
        for ch in processed_text.chars() {
            if !unique_chars.contains(&ch) {
                unique_chars.push(ch);
            }
        }
        if !unique_chars.contains(&'Ġ') {
            unique_chars.push('Ġ');
        }

        self.vocab.clear();
        self.inverse_vocab.clear();
        for (i, &ch) in unique_chars.iter().enumerate() {
            let id = i as u32;
            self.vocab.insert(id, ch.to_string());
            self.inverse_vocab.insert(ch.to_string(), id);
        }

        // Add special tokens
        for token in &allowed_special {
            if !self.inverse_vocab.contains_key(token) {
                let new_id = self.vocab.len() as u32;
                self.vocab.insert(new_id, token.clone());
                self.inverse_vocab.insert(token.clone(), new_id);
            }
        }

        // Tokenize the text
        let mut token_ids: Vec<u32> = processed_text
            .chars()
            .map(|ch| self.inverse_vocab[&ch.to_string()])
            .collect();

        // Find and replace frequent pairs
        for new_id in self.vocab.len()..vocab_size {
            if let Some(pair_id) = self.find_freq_pair(&token_ids, "most")? {
                token_ids = self.replace_pair(&token_ids, pair_id, new_id as u32);
                self.bpe_merges.insert(pair_id, new_id as u32);
            } else {
                break;
            }
        }

        // Build the vocabulary with the merged tokens
        for ((p0, p1), new_id) in &self.bpe_merges {
            let merged_token = format!("{}{}", self.vocab[p0], self.vocab[p1]);
            self.vocab.insert(*new_id, merged_token.clone());
            self.inverse_vocab.insert(merged_token, *new_id);
        }

        Ok(())
    }

    fn encode(&self, text: &str, allowed_special: Option<Vec<String>>) -> PyResult<Vec<u32>> {
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
                    token_ids.extend(self.encode_internal(prefix)?);
                    
                    let special_token = cap.as_str();
                    if let Some(&id) = self.inverse_vocab.get(special_token) {
                        token_ids.push(id);
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "Special token {} not found in vocabulary.",
                            special_token
                        )));
                    }
                    last_index = cap.end();
                }
                
                // Remaining part to process normally
                let remaining_text = &text[last_index..];
                
                // Check if any disallowed special tokens are in the remainder
                let disallowed: Vec<String> = self.inverse_vocab
                    .keys()
                    .filter(|tok| tok.starts_with("<|") && tok.ends_with("|>"))
                    .filter(|tok| remaining_text.contains(tok.as_str()))
                    .filter(|tok| !allowed_special.contains(tok))
                    .cloned()
                    .collect();
                
                if !disallowed.is_empty() {
                    return Err(PyValueError::new_err(format!(
                        "Disallowed special tokens encountered in text: {:?}",
                        disallowed
                    )));
                }
                
                token_ids.extend(self.encode_internal(remaining_text)?);
                return Ok(token_ids);
            }
        }
        
        // If no special tokens or remaining text after special token split
        token_ids.extend(self.encode_internal(text)?);
        Ok(token_ids)
    }

    fn decode(&self, token_ids: Vec<u32>) -> PyResult<String> {
        let mut decoded_string = String::new();
        
        for (i, &token_id) in token_ids.iter().enumerate() {
            if let Some(token) = self.vocab.get(&token_id) {
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
            } else {
                return Err(PyValueError::new_err(format!(
                    "Token ID {} not found in vocab",
                    token_id
                )));
            }
        }
        
        Ok(decoded_string)
    }

    fn save_vocab_and_merges(&self, vocab_path: &str, bpe_merges_path: &str) -> PyResult<()> {
        // Save vocabulary
        let vocab_json = serde_json::to_string_pretty(&self.vocab)
            .map_err(|e| PyValueError::new_err(format!("JSON serialization error: {}", e)))?;
        std::fs::write(vocab_path, vocab_json)
            .map_err(|e| PyValueError::new_err(format!("IO error: {}", e)))?;

        // Save BPE merges
        let merges_list: Vec<BpeMerge> = self.bpe_merges
            .iter()
            .map(|((p0, p1), &new_id)| BpeMerge {
                pair: vec![*p0, *p1],
                new_id,
            })
            .collect();
        
        let merges_json = serde_json::to_string_pretty(&merges_list)
            .map_err(|e| PyValueError::new_err(format!("JSON serialization error: {}", e)))?;
        std::fs::write(bpe_merges_path, merges_json)
            .map_err(|e| PyValueError::new_err(format!("IO error: {}", e)))?;

        Ok(())
    }

    fn load_vocab_and_merges(&mut self, vocab_path: &str, bpe_merges_path: &str) -> PyResult<()> {
        // Load vocabulary
        let vocab_content = std::fs::read_to_string(vocab_path)
            .map_err(|e| PyValueError::new_err(format!("IO error: {}", e)))?;
        let loaded_vocab: HashMap<String, String> = serde_json::from_str(&vocab_content)
            .map_err(|e| PyValueError::new_err(format!("JSON deserialization error: {}", e)))?;
        
        self.vocab.clear();
        self.inverse_vocab.clear();
        for (k, v) in loaded_vocab {
            let id: u32 = k.parse()
                .map_err(|e| PyValueError::new_err(format!("Invalid token ID: {}", e)))?;
            self.vocab.insert(id, v.clone());
            self.inverse_vocab.insert(v, id);
        }

        // Load BPE merges
        let merges_content = std::fs::read_to_string(bpe_merges_path)
            .map_err(|e| PyValueError::new_err(format!("IO error: {}", e)))?;
        let merges_list: Vec<BpeMerge> = serde_json::from_str(&merges_content)
            .map_err(|e| PyValueError::new_err(format!("JSON deserialization error: {}", e)))?;
        
        self.bpe_merges.clear();
        for merge in merges_list {
            if merge.pair.len() == 2 {
                let pair = (merge.pair[0], merge.pair[1]);
                self.bpe_merges.insert(pair, merge.new_id);
            }
        }

        Ok(())
    }

    #[staticmethod]
    fn find_freq_pair(token_ids: Vec<u32>, mode: &str) -> PyResult<Option<(u32, u32)>> {
        let mut pairs: HashMap<(u32, u32), usize> = HashMap::new();
        
        for window in token_ids.windows(2) {
            if window.len() == 2 {
                let pair = (window[0], window[1]);
                *pairs.entry(pair).or_insert(0) += 1;
            }
        }
        
        if pairs.is_empty() {
            return Ok(None);
        }
        
        match mode {
            "most" => {
                let max_pair = pairs.iter().max_by_key(|&(_, count)| count);
                Ok(max_pair.map(|(&pair, _)| pair))
            }
            "least" => {
                let min_pair = pairs.iter().min_by_key(|&(_, count)| count);
                Ok(min_pair.map(|(&pair, _)| pair))
            }
            _ => Err(PyValueError::new_err("Invalid mode. Choose 'most' or 'least'")),
        }
    }

    #[staticmethod]
    fn replace_pair(token_ids: Vec<u32>, pair_id: (u32, u32), new_id: u32) -> Vec<u32> {
        let mut replaced = Vec::new();
        let mut i = 0;
        
        while i < token_ids.len() {
            if i + 1 < token_ids.len() && (token_ids[i], token_ids[i + 1]) == pair_id {
                replaced.push(new_id);
                i += 2;
            } else {
                replaced.push(token_ids[i]);
                i += 1;
            }
        }
        
        replaced
    }
}

impl BPETokenizer {
    fn encode_internal(&self, text: &str) -> Result<Vec<u32>> {
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
        
        let mut token_ids = Vec::new();
        for token in tokens {
            if let Some(&id) = self.inverse_vocab.get(&token) {
                token_ids.push(id);
            } else {
                token_ids.extend(self.tokenize_with_bpe(&token)?);
            }
        }
        
        Ok(token_ids)
    }
    
    fn tokenize_with_bpe(&self, token: &str) -> Result<Vec<u32>> {
        let mut token_ids: Vec<u32> = token
            .chars()
            .map(|ch| {
                self.inverse_vocab
                    .get(&ch.to_string())
                    .copied()
                    .ok_or_else(|| TokenizerError::CharacterNotFound(ch.to_string()))
            })
            .collect::<Result<Vec<u32>>>()?;
        
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
}
