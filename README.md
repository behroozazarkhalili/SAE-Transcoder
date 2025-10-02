# SAE-Transcoder

Unified training framework for Sparse Autoencoders (SAE) and Transcoders using [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify).

## Overview

This project provides a clean, modular implementation for training both SAE and Transcoder models with comprehensive JSON-based configuration management.

### Key Features

- **Unified Interface**: Single codebase for both SAE and Transcoder training
- **JSON Configuration**: Easy-to-use configuration with meaningful defaults
- **Custom Hookpoints**: Unix pattern matching for flexible layer selection
- **Streaming Datasets**: Memory-efficient dataset loading
- **Flexible Experiment Tracking**: Support for both W&B and Trackio (local-first alternative)
- **Modular Design**: Separated config, trainer, and CLI components

### SAE vs Transcoder

**Sparse Autoencoder (SAE)**
- Purpose: Learn interpretable sparse representations
- Architecture: Input → Encoder → Sparse Latents → Decoder → Input
- Training: Reconstructs input activations (`transcode=False`)
- Loss: Typically FVU (Fraction of Variance Unexplained)
- Use Case: Interpretability research, feature discovery

**Transcoder**
- Purpose: End-to-end component replacement
- Architecture: Input → Encoder → Sparse Latents → Decoder → Output
- Training: Predicts module output from input (`transcode=True`)
- Loss: Typically CE/KL for task performance
- Use Case: Model component replacement, performance maintenance

## Installation

```bash
# Clone repository
git clone https://github.com/behroozazarkhalili/SAE-Transcoder.git
cd SAE-Transcoder

# Install dependencies
pip install sparsify transformers datasets wandb torch
```

## Quick Start

### Minimal Config Example

Create a `config.json` with just the model type:

```json
{
  "model_type": "sae"
}
```

### Train

```bash
# Train with minimal config (uses all defaults)
python -m src.main --config config.json

# Train with custom run name
python -m src.main --config config.json --run-name my_experiment

# Train with W&B logging
python -m src.main --config config.json --wandb
```

## Configuration

### Available Parameters

All parameters have sensible defaults - only specify what you want to override.

#### Model Type Selection
- `model_type`: `"sae"` or `"transcoder"` (required)

#### Model and Dataset
- `model_name`: HuggingFace model identifier (default: `"HuggingFaceTB/SmolLM2-135M"`)
- `dataset_name`: HuggingFace dataset identifier (default: `"EleutherAI/SmolLM2-135M-10B"`)
- `dataset_split`: Dataset split to use (default: `"train"`)
- `max_samples`: Maximum samples to use (default: `1000`, `null` = use all)
- `max_seq_length`: Maximum sequence length (default: `256`)

#### Architecture
- `expansion_factor`: Dictionary size multiplier (default: `8`)
- `num_latents`: Explicit latent count (overrides expansion_factor if > 0, default: `0`)
- `k`: Number of top-k active features (default: `8`)
- `activation`: `"topk"` or `"groupmax"` (default: `"topk"`)
- `normalize_decoder`: Normalize decoder weights (default: `true`)
- `multi_topk`: Use multi-scale top-k (default: `false`)
- `skip_connection`: Add skip connection (default: `false`)

#### Training Parameters
- `batch_size`: Samples per batch (default: `2`)
- `grad_acc_steps`: Gradient accumulation steps (default: `8`)
- `micro_acc_steps`: Micro-batching for memory (default: `1`)
- `loss_fn`: `"fvu"`, `"ce"`, or `"kl"` (default: `"fvu"`)

#### Optimization
- `optimizer`: `"adam"`, `"muon"`, or `"signum"` (default: `"adam"`)
- `lr`: Learning rate (default: `null` = auto-select)
- `lr_warmup_steps`: Warmup steps (default: `1000`)
- `weight_decay`: L2 regularization (default: `0.0`)

#### Sparsity and Regularization
- `auxk_alpha`: Auxiliary loss weight (default: `0.03125`)
- `dead_feature_threshold`: Tokens before feature considered dead (default: `5000000`)
- `k_decay_steps`: Steps to decay k value (default: `20000`)

#### Layer Selection
- `layers`: Specific layer indices (e.g., `[6, 7, 8]`, default: `null`)
- `layer_stride`: Stride for layer selection (default: `1`)
- `hookpoints`: Custom hookpoints with Unix patterns (default: `null`)

#### Training Control
- `max_steps`: Maximum training steps (default: `null` = full dataset)
- `init_seeds`: Random seeds for initialization (default: `[42]`)
- `finetune`: Finetune from checkpoint (default: `false`)
- `exclude_tokens`: Token IDs to exclude (default: `[]`)

#### Distributed Training
- `distribute_modules`: Distribute across GPUs (default: `false`)

#### Logging and Saving
- `save_dir`: Checkpoint directory (default: `"./checkpoints"`)
- `run_name`: Experiment name (default: `"unified_experiment"`)
- `save_every`: Save every N steps (default: `1000`)
- `save_best`: Save best checkpoint (default: `true`)

#### Experiment Tracking (W&B or Trackio)
- `experiment_tracker`: `"wandb"` or `"trackio"` (default: `"wandb"`)
  - **wandb**: Weights & Biases (cloud-based experiment tracking)
  - **trackio**: HuggingFace Trackio (local-first, open-source alternative)
- `log_to_wandb`: Enable experiment tracking (default: `false`)
- `wandb_project`: Project name for both wandb and trackio (default: `"sae-transcoder-unified"`)
- `wandb_entity`: W&B entity/team or HF username (default: `null`)
- `wandb_log_frequency`: Log every N steps (default: `100`)

#### Hardware
- `device`: Device for training (default: `"cuda:1"` if available, else `"cpu"`)
- `dtype`: Floating point precision (default: `"bfloat16"`)

### Custom Hookpoints

Hookpoints support Unix pattern matching for flexible layer selection:

```json
{
  "model_type": "sae",
  "hookpoints": ["h.*.attn", "h.*.mlp.act"]
}
```

**Pattern Examples:**
- `"h.*.attn"` - All attention layers
- `"h.*.mlp.act"` - All MLP activations
- `"h.[012].attn"` - Attention layers 0, 1, 2
- `"h.[0-5].attn"` - Attention layers 0 through 5
- `"layers.*"` - All layers (wildcard)

### Example Configs

See `config_examples/` directory for complete examples:
- `sae_basic.json` - Basic SAE configuration
- `transcoder_layers.json` - Transcoder with specific layers
- `custom_hookpoints.json` - Custom hookpoints with patterns
- `advanced_full.json` - All available parameters

## Project Structure

```
SAE-Transcoder/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── trainer.py           # Unified trainer
│   └── main.py              # CLI entry point
├── config_examples/         # Example configurations
├── archive/                 # Archived development files
├── train_sae_and_transcoder.py  # Legacy unified script
└── README.md
```

## Usage Examples

### Basic SAE Training

```json
{
  "model_type": "sae",
  "max_samples": 1000,
  "expansion_factor": 8,
  "k": 8,
  "run_name": "my_sae_experiment"
}
```

### Transcoder with Specific Layers

```json
{
  "model_type": "transcoder",
  "layers": [6, 7, 8],
  "expansion_factor": 16,
  "k": 16,
  "batch_size": 2,
  "run_name": "transcoder_layers_6_7_8"
}
```

### Custom Hookpoints

```json
{
  "model_type": "sae",
  "model_name": "gpt2",
  "hookpoints": ["h.*.attn", "h.*.mlp.act"],
  "expansion_factor": 32,
  "k": 32,
  "run_name": "custom_hookpoints"
}
```

### Using Trackio (Local-First Experiment Tracking)

```json
{
  "model_type": "sae",
  "experiment_tracker": "trackio",
  "log_to_wandb": true,
  "wandb_project": "my_local_experiments",
  "wandb_entity": "your_hf_username",
  "run_name": "trackio_experiment"
}
```

**Note**: Install trackio with `pip install trackio`. Trackio stores experiments locally by default and can optionally sync to HuggingFace Spaces.

## Development

### Running Tests

```bash
# Test with minimal config
python -m src.main --config /tmp/test_minimal.json
```

### Module Usage

```python
from src import UnifiedTrainingConfig, ConfigLoader, UnifiedTrainer

# Load config
config = ConfigLoader.load_from_json("config.json")

# Create trainer
trainer = UnifiedTrainer(config)

# Train
trained_model = trainer.train()
```

## Contributing

Contributions are welcome! Please ensure code follows the existing structure and includes appropriate documentation.

## License

This project uses the [EleutherAI/sparsify](https://github.com/EleutherAI/sparsify) library. Please refer to their repository for licensing information.

## Citation

If you use this code, please cite the EleutherAI sparsify library:

```bibtex
@software{eleutherai_sparsify,
  title = {Sparsify: Sparse Autoencoder Training Library},
  author = {EleutherAI},
  url = {https://github.com/EleutherAI/sparsify},
  year = {2024}
}
```

## Acknowledgments

- [EleutherAI](https://www.eleuther.ai/) for the sparsify library
- [HuggingFace](https://huggingface.co/) for transformers and datasets
