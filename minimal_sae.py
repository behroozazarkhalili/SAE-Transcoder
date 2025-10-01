#!/usr/bin/env python3
"""
Minimal SAE Training Example
Using synthetic data to avoid downloads and tensor dimension issues
Based on EleutherAI/sparsify documentation and working patterns
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import SaeConfig, Trainer, TrainConfig

def create_synthetic_dataset(tokenizer, num_samples=50, seq_length=64):
    """
    Create a small synthetic dataset that mimics the expected format
    from chunk_and_tokenize to avoid the tensor dimension bug
    """
    print(f"Creating synthetic dataset: {num_samples} samples, {seq_length} tokens each")

    # Create random token sequences within vocabulary range
    vocab_size = len(tokenizer)

    # Generate random input_ids ensuring they're within vocab range
    # Use smaller numbers to avoid potential OOV issues
    input_ids = []
    for _ in range(num_samples):
        # Generate valid token IDs (avoid special tokens at the edges)
        sample = torch.randint(1, min(vocab_size-1, 1000), (seq_length,)).tolist()
        input_ids.append(sample)

    # Create dataset in the format expected by sparsify
    dataset = Dataset.from_dict({
        'input_ids': input_ids
    })

    # Set format to torch as expected by the trainer
    dataset = dataset.with_format('torch', columns=['input_ids'])

    print(f"✅ Synthetic dataset created: {len(dataset)} samples")
    print(f"   Sample shape: {dataset[0]['input_ids'].shape}")
    print(f"   Token range: {min(dataset[0]['input_ids'])} - {max(dataset[0]['input_ids'])}")

    return dataset

def main():
    """
    Minimal SAE training with synthetic data
    """
    print("🚀 Starting Minimal SAE Training")

    # Use proven working model from documentation
    MODEL = "EleutherAI/pythia-160m"

    print(f"📥 Loading model: {MODEL}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with minimal configuration
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": "cuda:1" if torch.cuda.is_available() else "cpu"},
        torch_dtype=torch.bfloat16,
    )

    print(f"✅ Model loaded on: {next(model.parameters()).device}")

    # Create synthetic dataset that matches expected format
    dataset = create_synthetic_dataset(tokenizer, num_samples=100, seq_length=128)

    print("🏗️ Creating SAE configuration")

    # Use exact minimal configuration from documentation
    # This should avoid the tensor dimension bug by using defaults
    sae_config = SaeConfig()
    train_config = TrainConfig(
        sae=sae_config,
        batch_size=4,  # Very small batch to minimize memory and potential issues
        save_dir="./minimal_sae_results",
        run_name="minimal_test",
        log_to_wandb=False,  # Disable wandb for simplicity
    )

    print("📋 Configuration:")
    print(f"   - Batch size: {train_config.batch_size}")
    print(f"   - Expansion factor: {sae_config.expansion_factor}")
    print(f"   - Top-k sparsity: {sae_config.k}")
    print(f"   - Transcode mode: {sae_config.transcode}")

    print("🏋️ Creating trainer")
    trainer = Trainer(train_config, dataset, model)

    print("🚀 Starting training (this may take a few minutes)")
    try:
        trainer.fit()
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to: {train_config.save_dir}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"💡 This helps us understand the library's behavior")
        raise

if __name__ == "__main__":
    main()