#!/usr/bin/env python3
"""
Debug script to understand the mask tensor dimension issue
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import SaeConfig, Trainer, TrainConfig

def debug_mask_dimensions():
    """Debug the exact mask dimensions that cause the error"""

    MODEL = "EleutherAI/pythia-160m"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
        torch_dtype=torch.bfloat16,
    )

    # Create very minimal dataset - just one sample
    input_ids = [[1, 2, 3, 4, 5]]  # Single sample, 5 tokens
    dataset = Dataset.from_dict({'input_ids': input_ids})
    dataset = dataset.with_format('torch', columns=['input_ids'])

    print(f"Dataset sample shape: {dataset[0]['input_ids'].shape}")

    # Create minimal config
    sae_config = SaeConfig()
    train_config = TrainConfig(
        sae=sae_config,
        batch_size=1,  # Single batch to simplify
        save_dir="./debug_results",
        run_name="debug_test",
        log_to_wandb=False,
    )

    print(f"Creating trainer...")
    trainer = Trainer(train_config, dataset, model)

    # Let's patch the hook to debug mask dimensions
    original_hook = None

    def debug_hook(module, args, kwargs, outputs):
        print(f"\n=== DEBUG HOOK ===")
        print(f"Module: {module}")
        print(f"Outputs type: {type(outputs)}")

        if isinstance(outputs, tuple):
            print(f"Outputs is tuple with {len(outputs)} elements")
            for i, item in enumerate(outputs):
                if hasattr(item, 'shape'):
                    print(f"  Element {i} shape: {item.shape}")
                    print(f"  Element {i} dtype: {item.dtype}")
                else:
                    print(f"  Element {i} type: {type(item)}")
        elif hasattr(outputs, 'shape'):
            print(f"Outputs shape: {outputs.shape}")
            print(f"Outputs dtype: {outputs.dtype}")
        else:
            print(f"Outputs has no shape attribute")

        print(f"==================\n")
        return outputs

    # Add debug hook to first layer to see tensor shapes
    first_layer = None
    for name, module in model.named_modules():
        if 'layers.0' in name and 'mlp' not in name and 'attention' not in name:
            first_layer = module
            break

    if first_layer:
        print(f"Adding debug hook to: {first_layer}")
        first_layer.register_forward_hook(debug_hook, with_kwargs=True)

    try:
        print("Starting training to trigger the error...")
        trainer.fit()
    except Exception as e:
        print(f"Expected error occurred: {e}")
        print(f"Error type: {type(e)}")

        # Let's try to understand what went wrong
        print("\nDebugging the trainer state...")

if __name__ == "__main__":
    debug_mask_dimensions()