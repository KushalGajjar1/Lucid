use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Character not found in vocabulary: {0}")]
    CharacterNotFound(String),
    
    #[error("Token ID not found in vocabulary: {0}")]
    TokenIdNotFound(u32),
    
    #[error("Special token not found in vocabulary: {0}")]
    SpecialTokenNotFound(String),
    
    #[error("Disallowed special tokens encountered: {0:?}")]
    DisallowedSpecialTokens(Vec<String>),
    
    #[error("Invalid mode. Choose 'most' or 'least'")]
    InvalidMode,
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, TokenizerError>;
