"""
Rust-based BPE Tokenizer wrapper for Python.
This module provides the same interface as the original Python implementation
but uses the high-performance Rust core underneath.
"""

try:
    from lucid_tokenizer import BPETokenizer as RustBPETokenizer
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust implementation not available, falling back to Python implementation")

from collections import Counter, deque
import json
import re
from typing import List, Set, Optional, Union

class BPETokenizer:
    """
    BPE Tokenizer with Rust core implementation.
    
    This class provides the same interface as the original Python implementation
    but delegates the core functionality to a high-performance Rust implementation.
    """

    def __init__(self):
        if RUST_AVAILABLE:
            self._rust_tokenizer = RustBPETokenizer()
            self._use_rust = True
        else:
            # Fallback to Python implementation
            self._use_rust = False
            self._python_tokenizer = _PythonBPETokenizer()
        
        # Expose the same attributes for compatibility
        if self._use_rust:
            self.vocab = self._rust_tokenizer.vocab
            self.inverse_vocab = self._rust_tokenizer.inverse_vocab
            self.bpe_merges = self._rust_tokenizer.bpe_merges
        else:
            self.vocab = self._python_tokenizer.vocab
            self.inverse_vocab = self._python_tokenizer.inverse_vocab
            self.bpe_merges = self._python_tokenizer.bpe_merges

    def train(self, text: str, vocab_size: int, allowed_special: Set[str] = {"<|endoftext|>"}) -> None:
        """
        Train BPE Tokenizer

        Args:
            text (str): The text used to train the tokenizer
            vocab_size (int): The vocabulary size
            allowed_special (set): A set of included special tokens
        """
        if self._use_rust:
            self._rust_tokenizer.train(text, vocab_size, list(allowed_special))
            # Update the exposed attributes
            self.vocab = self._rust_tokenizer.vocab
            self.inverse_vocab = self._rust_tokenizer.inverse_vocab
            self.bpe_merges = self._rust_tokenizer.bpe_merges
        else:
            self._python_tokenizer.train(text, vocab_size, allowed_special)
            self.vocab = self._python_tokenizer.vocab
            self.inverse_vocab = self._python_tokenizer.inverse_vocab
            self.bpe_merges = self._python_tokenizer.bpe_merges

    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """
        Encode the input text into a list of token IDs

        Args:
            text (str): The input text to encode
            allowed_special (set or None): Special tokens to allow passthrough

        Returns:
            List of token IDs.
        """
        if self._use_rust:
            allowed_special_list = list(allowed_special) if allowed_special else None
            return self._rust_tokenizer.encode(text, allowed_special_list)
        else:
            return self._python_tokenizer.encode(text, allowed_special)

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a string

        Args:
            token_ids (list[int]): The list of token IDs to decode

        Returns:
            str: The decoded string
        """
        if self._use_rust:
            return self._rust_tokenizer.decode(token_ids)
        else:
            return self._python_tokenizer.decode(token_ids)

    def save_vocab_and_merges(self, vocab_path: str, bpe_merges_path: str) -> None:
        """
        Saves the vocabulary and BPE merges to JSON files

        Args:
            vocab_path (str): Path to save vocabulary
            bpe_merges_path (str): Path to save the BPE merges
        """
        if self._use_rust:
            self._rust_tokenizer.save_vocab_and_merges(vocab_path, bpe_merges_path)
        else:
            self._python_tokenizer.save_vocab_and_merges(vocab_path, bpe_merges_path)

    def load_vocab_and_merges(self, vocab_path: str, bpe_merges_path: str) -> None:
        """
        Load the vocabulary and BPE merges from JSON files

        Args:
            vocab_path (str): Path to the vocabulary file
            bpe_merges_path (str): Path to the BPE merges file
        """
        if self._use_rust:
            self._rust_tokenizer.load_vocab_and_merges(vocab_path, bpe_merges_path)
            # Update the exposed attributes
            self.vocab = self._rust_tokenizer.vocab
            self.inverse_vocab = self._rust_tokenizer.inverse_vocab
            self.bpe_merges = self._rust_tokenizer.bpe_merges
        else:
            self._python_tokenizer.load_vocab_and_merges(vocab_path, bpe_merges_path)
            self.vocab = self._python_tokenizer.vocab
            self.inverse_vocab = self._python_tokenizer.inverse_vocab
            self.bpe_merges = self._python_tokenizer.bpe_merges

    @staticmethod
    def find_freq_pair(token_ids: List[int], mode: str = "most") -> Optional[tuple[int, int]]:
        """
        Find the most or least frequent pair of consecutive tokens
        
        Args:
            token_ids (list[int]): List of token IDs
            mode (str): 'most' or 'least' frequent
            
        Returns:
            tuple[int, int] or None: The most/least frequent pair
        """
        if RUST_AVAILABLE:
            return RustBPETokenizer.find_freq_pair(token_ids, mode)
        else:
            return _PythonBPETokenizer.find_freq_pair(token_ids, mode)

    @staticmethod
    def replace_pair(token_ids: List[int], pair_id: tuple[int, int], new_id: int) -> List[int]:
        """
        Replace all occurrences of a pair with a new token ID
        
        Args:
            token_ids (list[int]): List of token IDs
            pair_id (tuple[int, int]): The pair to replace
            new_id (int): The new token ID to insert
            
        Returns:
            list[int]: New list with replacements
        """
        if RUST_AVAILABLE:
            return RustBPETokenizer.replace_pair(token_ids, pair_id, new_id)
        else:
            return _PythonBPETokenizer.replace_pair(token_ids, pair_id, new_id)


class _PythonBPETokenizer:
    """
    Fallback Python implementation of BPE Tokenizer.
    This is used when the Rust implementation is not available.
    """
    
    def __init__(self):
        # Maps token id to token string
        self.vocab = {}
        # Maps token string to token id
        self.inverse_vocab = {}
        # Dictionary of BPE merges : {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

    def train(self, text: str, vocab_size: int, allowed_special: Set[str] = {"<|endoftext|>"}) -> None:
        """
        Train BPE Tokenizer

        Args:
            text (str) : The text used to train the tokenizer
            vocab_size (int) : The vocabulary size
            allowed_special (set) : A set of included special tokens
        """

        # Replace space with "Ġ"
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with unique characters
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")
        
        self.vocab = {i : char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char : i for i, char in self.vocab.items()}

        # Add special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the text
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # Find and Replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id
        
        # Build the vocabulary with the merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """
        Encode the input text into a list of token IDs

        Args:
            text (str) : The input text to encode
            allowed_special (set or None) : Special tokens to allow passthrough

        Returns:
            List of token IDs.
        """

        token_ids = []
        if allowed_special is not None and len(allowed_special) > 0:
            # Build regex to match allowed special tokens
            special_pattern = ("(" + "|".join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ")")

            last_index = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_index:match.start()]
                # Encode prefix without special handling
                token_ids.extend(self.encode(prefix, allowed_special=None))

                special_token = match.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocabulary.")
                last_index = match.end()
            
            # Remaining part to process normally
            text = text[last_index:]

            # Check if any disallowed special tokens are in the remainder
            disallowed = [
                tok for tok in self.inverse_vocab 
                if tok.startswith("<|") and tok.endswith("|>") and tok in text and tok not in allowed_special
            ]

            if disallowed:
                raise ValueError(f"Disallowed special tokens encounterd in text : {disallowed}")

        # If no special tokens or remaining text after special token split
        tokens = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")
            words = line.split()
            for j, word in enumerate(words):
                if j == 0 and i > 0:
                    tokens.append("Ġ" + word)
                elif j == 0:
                    tokens.append(word)
                else:
                    tokens.append("Ġ" + word)

        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))

        return token_ids

    def tokenize_with_bpe(self, token: str) -> List[int]:
        """
        Tokenize a single token using BPE merges

        Args:
            token (str) : The token to tokenize

        Return:
            list[int] : The list of token IDs after applying BPE 
        """

        # Tokenize the token into individual characters
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab : {missing_chars}")
        
        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i+1])
                if pair in self.bpe_merges:
                    merged_token_id = self.bpe_merges[pair]
                    new_tokens.append(merged_token_id)
                    i += 2
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a string

        Args:
            token_ids (list[int]) : The list of token IDs to decode

        Returns:
            str: The decoded string
        """

        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab")
            token = self.vocab[token_id]
            if token == "\n":
                if decoded_string and not decoded_string.endswith(" "):
                    decoded_string += " "
                decoded_string += token
            elif token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token

        return decoded_string

    def save_vocab_and_merges(self, vocab_path: str, bpe_merges_path: str) -> None:
        """
        Saves the vocabulary and BPE merges to JSON files

        Args:
            vocab_path (str) : Path to save vocabulary
            bpe_merges_path (str) : Path to save the BPE merges
        """

        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=4)

        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair" : list(pair), "new_id" : new_id} for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=4)

    def load_vocab_and_merges(self, vocab_path: str, bpe_merges_path: str) -> None:
        """
        Load the vocabulary and BPE merges from JSON files

        Args:
            vocab_path (str) : Path to the vocabulary file
            bpe_merges_path (str) : Path to the BPE merges file
        """

        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k) : v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v : int(k) for k, v in loaded_vocab.items()}

        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

    @staticmethod
    def find_freq_pair(token_ids: List[int], mode: str = "most") -> Optional[tuple[int, int]]:
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if not pairs:
            return None
        
        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'")
        
    @staticmethod
    def replace_pair(token_ids: List[int], pair_id: tuple[int, int], new_id: int) -> List[int]:
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(current)

        return replaced
