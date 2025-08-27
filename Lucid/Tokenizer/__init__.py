"""
Lucid Tokenizer Module

This module provides a high-performance BPE (Byte Pair Encoding) tokenizer
with a Rust core implementation and Python bindings.
"""

try:
    from .rust_tokenizer import BPETokenizer
except ImportError:
    # Fallback to original Python implementation if Rust version is not available
    from .tokenizer import BPETokenizer

__all__ = ["BPETokenizer"]