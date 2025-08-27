from setuptools import setup, find_packages
from setuptools_rust import RustExtension

setup(
    name="Lucid",
    version="0.1.0",
    description="Library to simplify Training with high-performance Rust core",
    author="Kushal Gajjar",
    author_email="kushalgajjar1@gmail.com",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.9",
    url="https://github.com/KushalGajjar1/Lucid",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    rust_extensions=[
        RustExtension(
            "lucid_tokenizer",
            path="Cargo.toml",
            binding=setuptools_rust.Binding.PyO3,
        )
    ],
    zip_safe=False,
)