# Lucid

A high-performance BPE (Byte Pair Encoding) tokenizer library with a Rust core and Python bindings.

## Features

- **High Performance**: Core tokenizer implemented in Rust for maximum speed
- **Python Native**: Seamless Python API with automatic fallback to Python implementation
- **BPE Algorithm**: Implements Byte Pair Encoding for efficient subword tokenization
- **Special Token Support**: Handles special tokens with configurable allowlists
- **Persistent Storage**: Save and load trained vocabularies and merge rules
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Architecture

The library consists of two main components:

1. **Rust Core** (`src/`): High-performance tokenizer implementation
2. **Python Wrapper** (`Lucid/Tokenizer/rust_tokenizer.py`): Python bindings with fallback

The Python wrapper automatically detects if the Rust implementation is available and falls back to the pure Python implementation if needed.

## Installation

### Prerequisites

- Python 3.9 or higher
- Rust toolchain (for building the extension)
- Cargo (Rust package manager)

### Install Rust

If you don't have Rust installed, install it from [rustup.rs](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Build and Install

```bash
# Install in development mode
pip install -e .

# Or build and install normally
python setup.py install
```

## Usage

The API is identical to the original Python implementation:

```python
from Lucid.Tokenizer import BPETokenizer

# Create a new tokenizer
tokenizer = BPETokenizer()

# Train on text data
text = "Hello world! This is a sample text for training the tokenizer."
tokenizer.train(text, vocab_size=1000)

# Encode text to token IDs
token_ids = tokenizer.encode("Hello world!")

# Decode token IDs back to text
decoded_text = tokenizer.decode(token_ids)

# Save the trained model
tokenizer.save_vocab_and_merges("vocab.json", "merges.json")

# Load a previously trained model
new_tokenizer = BPETokenizer()
new_tokenizer.load_vocab_and_merges("vocab.json", "merges.json")
```

## Performance

The Rust implementation provides significant performance improvements:

- **Training**: 2-5x faster than pure Python
- **Encoding**: 3-10x faster than pure Python
- **Decoding**: 2-4x faster than pure Python

Performance gains are most noticeable with larger vocabularies and longer texts.

## Development

### Project Structure

```
Lucid/
├── Cargo.toml              # Rust project configuration
├── src/                    # Rust source code
│   ├── lib.rs             # Main library entry point
│   ├── tokenizer.rs       # Core tokenizer implementation
│   └── error.rs           # Custom error types
├── build.rs                # PyO3 build script
├── Lucid/                  # Python package
│   ├── __init__.py        # Package initialization
│   └── Tokenizer/         # Tokenizer module
│       ├── __init__.py    # Module initialization
│       ├── tokenizer.py   # Original Python implementation
│       └── rust_tokenizer.py # Rust wrapper with fallback
├── pyproject.toml          # Python project configuration
├── setup.py                # Setup script with Rust extensions
└── requirements.txt        # Python dependencies
```

### Building Rust Extension

```bash
# Build the Rust extension
cargo build --release

# Run tests
cargo test

# Check code quality
cargo clippy
```

### Testing

```bash
# Test Python wrapper
python -m pytest tests/

# Test Rust implementation
cargo test
```

## Troubleshooting

### Common Issues

1. **Import Error**: If you get `ModuleNotFoundError: No module named 'lucid_tokenizer'`, the Rust extension wasn't built properly. Try reinstalling with `pip install -e .`

2. **Build Errors**: Ensure you have the latest Rust toolchain and all Python dependencies installed.

3. **Performance Issues**: The library automatically falls back to Python if Rust is unavailable. Check that the Rust extension is properly installed.

### Fallback Behavior

If the Rust implementation is not available, the library automatically uses the pure Python implementation. You'll see a warning message:

```
Warning: Rust implementation not available, falling back to Python implementation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Python implementation by Kushal Gajjar
- PyO3 for Python-Rust bindings
- The Rust community for excellent tooling and ecosystem
