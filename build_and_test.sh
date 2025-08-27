#!/bin/bash

echo "=== Building and Testing Rust BPE Tokenizer ==="

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo is not installed. Please install Rust first."
    echo "Visit https://rustup.rs/ for installation instructions."
    exit 1
fi

echo "✓ Rust/Cargo found"

# Clean any previous builds
echo "Cleaning previous builds..."
cargo clean

# Build the project
echo "Building project..."
if cargo build; then
    echo "✓ Build successful"
else
    echo "✗ Build failed"
    exit 1
fi

# Run tests
echo "Running tests..."
if cargo test; then
    echo "✓ All tests passed"
else
    echo "✗ Tests failed"
    exit 1
fi

# Run example
echo "Running example..."
if cargo run --example basic_usage; then
    echo "✓ Example ran successfully"
else
    echo "✗ Example failed"
    exit 1
fi

# Check if criterion is available for benchmarks
if cargo bench --no-run &> /dev/null; then
    echo "Running benchmarks..."
    if cargo bench; then
        echo "✓ Benchmarks completed"
    else
        echo "⚠ Benchmarks failed (this is optional)"
    fi
else
    echo "⚠ Criterion not available, skipping benchmarks"
fi

# Check code quality
echo "Checking code quality..."
if cargo clippy -- -D warnings &> /dev/null; then
    echo "✓ Code quality checks passed"
else
    echo "⚠ Code quality issues found (run 'cargo clippy' for details)"
fi

echo ""
echo "=== Build and Test Summary ==="
echo "✓ Project built successfully"
echo "✓ All tests passed"
echo "✓ Example ran successfully"
echo "✓ Ready for use!"

echo ""
echo "To use the tokenizer in your project:"
echo "1. Add to Cargo.toml: lucid-tokenizer = \"0.1.0\""
echo "2. Import: use lucid_tokenizer::BPETokenizer;"
echo ""
echo "To run benchmarks: cargo bench"
echo "To check code quality: cargo clippy"
echo "To view documentation: cargo doc --open" 