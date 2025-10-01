#!/usr/bin/env python3
"""
Create a tiny dataset file that the CLI can use
"""

from datasets import Dataset
import os

def main():
    # Create minimal text data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test sentence.",
        "Machine learning is fascinating and complex.",
        "Sparse autoencoders help with interpretability.",
        "Python is a great programming language.",
    ] * 40  # Repeat to get enough samples

    # Create dataset
    dataset = Dataset.from_dict({"text": texts})

    # Save to disk
    os.makedirs("./tiny_dataset", exist_ok=True)
    dataset.save_to_disk("./tiny_dataset")

    print(f"✅ Created tiny dataset with {len(dataset)} samples")
    print(f"📁 Saved to: ./tiny_dataset")

if __name__ == "__main__":
    main()