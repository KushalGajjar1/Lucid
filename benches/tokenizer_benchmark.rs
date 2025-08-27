use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucid_tokenizer::BPETokenizer;
use std::collections::HashSet;

fn bench_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("Training");
    
    let training_text = "The quick brown fox jumps over the lazy dog. ".repeat(1000);
    
    group.bench_function("train_vocab_1000", |b| {
        b.iter(|| {
            let mut tokenizer = BPETokenizer::new();
            tokenizer.train(black_box(&training_text), 1000, None).unwrap();
        });
    });
    
    group.bench_function("train_vocab_5000", |b| {
        b.iter(|| {
            let mut tokenizer = BPETokenizer::new();
            tokenizer.train(black_box(&training_text), 5000, None).unwrap();
        });
    });
    
    group.finish();
}

fn bench_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("Encoding");
    
    // Pre-train a tokenizer
    let mut tokenizer = BPETokenizer::new();
    let training_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    tokenizer.train(&training_text, 2000, None).unwrap();
    
    let test_texts = vec![
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "This is a longer text that will test the encoding performance of our BPE tokenizer implementation",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
    ];
    
    for (i, text) in test_texts.iter().enumerate() {
        group.bench_function(&format!("encode_text_{}", i), |b| {
            b.iter(|| {
                tokenizer.encode(black_box(text), None).unwrap();
            });
        });
    }
    
    group.finish();
}

fn bench_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decoding");
    
    // Pre-train a tokenizer and encode some text
    let mut tokenizer = BPETokenizer::new();
    let training_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    tokenizer.train(&training_text, 2000, None).unwrap();
    
    let test_texts = vec![
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "This is a longer text that will test the decoding performance",
    ];
    
    let encoded_texts: Vec<Vec<usize>> = test_texts
        .iter()
        .map(|text| tokenizer.encode(text, None).unwrap())
        .collect();
    
    for (i, encoded) in encoded_texts.iter().enumerate() {
        group.bench_function(&format!("decode_text_{}", i), |b| {
            b.iter(|| {
                tokenizer.decode(black_box(encoded)).unwrap();
            });
        });
    }
    
    group.finish();
}

fn bench_special_tokens(c: &mut Criterion) {
    let mut group = c.benchmark_group("Special Tokens");
    
    // Pre-train a tokenizer with special tokens
    let mut tokenizer = BPETokenizer::new();
    let special_tokens: HashSet<String> = [
        "<|endoftext|>".to_string(),
        "<|startoftext|>".to_string(),
        "<|pad|>".to_string(),
        "<|unk|>".to_string(),
    ].into_iter().collect();
    
    let training_text = "Hello <|startoftext|> world <|endoftext|> goodbye <|pad|> ".repeat(100);
    tokenizer.train(&training_text, 2000, Some(special_tokens.clone())).unwrap();
    
    let test_text = "Hello <|startoftext|> world <|endoftext|> goodbye <|pad|>";
    
    group.bench_function("encode_with_special_tokens", |b| {
        b.iter(|| {
            tokenizer.encode(black_box(test_text), Some(&special_tokens)).unwrap();
        });
    });
    
    group.bench_function("encode_without_special_tokens", |b| {
        b.iter(|| {
            tokenizer.encode(black_box(test_text), None).unwrap();
        });
    });
    
    group.finish();
}

fn bench_save_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("Save/Load");
    
    // Pre-train a tokenizer
    let mut tokenizer = BPETokenizer::new();
    let training_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    tokenizer.train(&training_text, 2000, None).unwrap();
    
    group.bench_function("save_tokenizer", |b| {
        b.iter(|| {
            tokenizer.save_vocab_and_merges("temp_vocab.json", "temp_merges.json").unwrap();
            // Clean up immediately
            std::fs::remove_file("temp_vocab.json").unwrap();
            std::fs::remove_file("temp_merges.json").unwrap();
        });
    });
    
    // Prepare files for loading benchmark
    tokenizer.save_vocab_and_merges("temp_vocab.json", "temp_merges.json").unwrap();
    
    group.bench_function("load_tokenizer", |b| {
        b.iter(|| {
            let mut loaded_tokenizer = BPETokenizer::new();
            loaded_tokenizer.load_vocab_and_merges("temp_vocab.json", "temp_merges.json").unwrap();
        });
    });
    
    // Clean up
    std::fs::remove_file("temp_vocab.json").unwrap();
    std::fs::remove_file("temp_merges.json").unwrap();
    
    group.finish();
}

criterion_group!(
    benches,
    bench_training,
    bench_encoding,
    bench_decoding,
    bench_special_tokens,
    bench_save_load
);
criterion_main!(benches); 