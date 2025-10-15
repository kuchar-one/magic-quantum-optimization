#!/usr/bin/env python3
"""
Setup script for Magic Quantum Sequence Optimization
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="magic-quantum-optimization",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive quantum optimization framework for preparing target superposition states",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/magic-quantum-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "magic-optimize=quo:main",
            "magic-app=run_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
    keywords="quantum computing, optimization, multi-objective, GPU, CUDA, streamlit",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/magic-quantum-optimization/issues",
        "Source": "https://github.com/yourusername/magic-quantum-optimization",
        "Documentation": "https://github.com/yourusername/magic-quantum-optimization#readme",
    },
)
