use pyo3::prelude::*;

mod tokenizer;
mod error;

use tokenizer::BPETokenizer;

/// A Python module implemented in Rust.
#[pymodule]
fn lucid_tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BPETokenizer>()?;
    Ok(())
}
