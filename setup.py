#!/usr/bin/env python3
"""
Setup script for the Advanced RAG System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    # Core packages
    packages = [
        "torch",
        "transformers", 
        "sentence-transformers",
        "faiss-cpu",
        "spacy",
        "scikit-learn",
        "PyMuPDF",
        "python-docx",
        "PyPDF2",
        "langchain",
        "pandas",
        "numpy"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
    
    # Download spaCy model
    try:
        print("Downloading spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    except subprocess.CalledProcessError as e:
        print(f"Failed to download spaCy model: {e}")

def test_imports():
    """Test if all imports work"""
    print("\nTesting imports...")
    
    test_imports = [
        "torch",
        "transformers",
        "sentence_transformers",
        "faiss",
        "spacy",
        "sklearn",
        "fitz",
        "docx",
        "langchain"
    ]
    
    failed_imports = []
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        print("Please install missing packages manually.")
    else:
        print("\nAll imports successful!")

if __name__ == "__main__":
    print("Advanced RAG System Setup")
    print("=" * 30)
    
    install_requirements()
    test_imports()
    
    print("\nSetup complete!")
    print("You can now run: python rag.py") 