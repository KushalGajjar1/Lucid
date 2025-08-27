@echo off
echo === Building and Testing Rust BPE Tokenizer ===

REM Check if Rust is installed
cargo --version >nul 2>&1
if errorlevel 1 (
    echo Error: Rust/Cargo is not installed. Please install Rust first.
    echo Visit https://rustup.rs/ for installation instructions.
    pause
    exit /b 1
)

echo ✓ Rust/Cargo found

REM Clean any previous builds
echo Cleaning previous builds...
cargo clean

REM Build the project
echo Building project...
cargo build
if errorlevel 1 (
    echo ✗ Build failed
    pause
    exit /b 1
)
echo ✓ Build successful

REM Run tests
echo Running tests...
cargo test
if errorlevel 1 (
    echo ✗ Tests failed
    pause
    exit /b 1
)
echo ✓ All tests passed

REM Run example
echo Running example...
cargo run --example basic_usage
if errorlevel 1 (
    echo ✗ Example failed
    pause
    exit /b 1
)
echo ✓ Example ran successfully

REM Check if criterion is available for benchmarks
echo Checking for benchmark support...
cargo bench --no-run >nul 2>&1
if errorlevel 1 (
    echo ⚠ Criterion not available, skipping benchmarks
) else (
    echo Running benchmarks...
    cargo bench
    if errorlevel 1 (
        echo ⚠ Benchmarks failed (this is optional)
    ) else (
        echo ✓ Benchmarks completed
    )
)

REM Check code quality
echo Checking code quality...
cargo clippy -- -D warnings >nul 2>&1
if errorlevel 1 (
    echo ⚠ Code quality issues found (run 'cargo clippy' for details)
) else (
    echo ✓ Code quality checks passed
)

echo.
echo === Build and Test Summary ===
echo ✓ Project built successfully
echo ✓ All tests passed
echo ✓ Example ran successfully
echo ✓ Ready for use!

echo.
echo To use the tokenizer in your project:
echo 1. Add to Cargo.toml: lucid-tokenizer = "0.1.0"
echo 2. Import: use lucid_tokenizer::BPETokenizer;
echo.
echo To run benchmarks: cargo bench
echo To check code quality: cargo clippy
echo To view documentation: cargo doc --open

pause 