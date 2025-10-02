#!/usr/bin/env python3
"""
Unified SAE and Transcoder Training Script
Using EleutherAI/sparsify library with JSON-based configuration

This script provides a unified interface for training both Sparse Autoencoders (SAE)
and Transcoders with comprehensive JSON-based configuration management.

Key Differences:
- SAE (transcode=False): Reconstructs input activations for interpretability
- Transcoder (transcode=True): Predicts module output from input for component replacement

Custom Hookpoints:
- Supports Unix pattern matching for flexible layer selection
- Examples: "h.*.attn", "h.*.mlp.act", "layers.[012]", "h.[0-5].attn"
"""

from __future__ import annotations

import os
import sys
import torch
import json
import logging
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any
from dataclasses import dataclass, asdict, field
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from sparsify import SaeConfig, TranscoderConfig, Trainer, TrainConfig
from sparsify.config import SparseCoderConfig
from sparsify.data import chunk_and_tokenize

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION SYSTEM
# ==============================================================================

@dataclass
class UnifiedTrainingConfig:
    """Unified configuration for both SAE and Transcoder training"""

    # ==== MODEL TYPE SELECTION ====
    model_type: Literal["sae", "transcoder"] = "sae"  # KEY PARAMETER

    # Model and Dataset
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    dataset_name: str = "EleutherAI/SmolLM2-135M-10B"
    dataset_split: str = "train"
    max_samples: Optional[int] = 1_000
    max_seq_length: int = 256

    # ==== CORE ARCHITECTURE ====
    expansion_factor: int = 8
    num_latents: int = 0  # If > 0, overrides expansion_factor
    k: int = 8  # Number of top-k active features
    activation: Literal["topk", "groupmax"] = "topk"
    normalize_decoder: bool = True
    multi_topk: bool = False
    skip_connection: bool = False

    # ==== TRAINING PARAMETERS ====
    batch_size: int = 2
    grad_acc_steps: int = 8
    micro_acc_steps: int = 1
    loss_fn: Literal["fvu", "ce", "kl"] = "fvu"

    # ==== OPTIMIZATION ====
    optimizer: Literal["adam", "muon", "signum"] = "adam"
    lr: Optional[float] = None
    lr_warmup_steps: int = 1000
    weight_decay: float = 0.0

    # ==== SPARSITY AND REGULARIZATION ====
    auxk_alpha: float = 1/32
    dead_feature_threshold: int = 5_000_000
    k_decay_steps: int = 20_000

    # ==== LAYER SELECTION ====
    # Option 1: Specific layer numbers
    layers: Optional[List[int]] = None
    layer_stride: int = 1

    # Option 2: Custom hookpoints with Unix pattern matching
    # Examples: ["h.*.attn", "h.*.mlp.act"], ["layers.[012]"], ["h.[0-5].attn"]
    hookpoints: Optional[List[str]] = None

    # ==== TRAINING CONTROL ====
    max_steps: Optional[int] = None
    init_seeds: Optional[List[int]] = None
    finetune: bool = False
    exclude_tokens: Optional[List[int]] = None

    # ==== DISTRIBUTED TRAINING ====
    distribute_modules: bool = False

    # ==== LOGGING AND SAVING ====
    save_dir: str = "./checkpoints"
    run_name: str = "unified_experiment"
    save_every: int = 1000
    save_best: bool = True

    # ==== WEIGHTS & BIASES ====
    log_to_wandb: bool = False
    wandb_project: str = "sae-transcoder-unified"
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 100

    # ==== HARDWARE ====
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"

    def __post_init__(self):
        """Validate and adjust configuration"""
        if self.init_seeds is None:
            self.init_seeds = [42]
        if self.exclude_tokens is None:
            self.exclude_tokens = []

        # Convert dtype string to torch.dtype
        if isinstance(self.dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            self.dtype = dtype_map.get(self.dtype, torch.bfloat16)

        # Validate layer selection
        if self.layers is not None and self.hookpoints is not None:
            logger.warning("⚠️  Both 'layers' and 'hookpoints' specified. 'hookpoints' will take precedence.")

        # Auto-adjust parameters based on model type
        if self.model_type == "transcoder":
            # Note: Using FVU loss for transcoder to avoid Identity.forward() issue with Llama models
            if self.loss_fn == "ce":
                self.loss_fn = "fvu"
                logger.info("🔄 Auto-adjusted loss_fn to 'fvu' for transcoder (Llama compatibility)")

            if self.grad_acc_steps < 2:
                self.grad_acc_steps = 4
                logger.info("🔄 Auto-adjusted grad_acc_steps to 4 for transcoder")

            if self.batch_size > 16:
                self.batch_size = 16
                logger.info("🔄 Auto-adjusted batch_size to 16 for transcoder")

        # Update save directory and run name
        self.save_dir = f"./{self.model_type}_checkpoints"
        if self.run_name == "unified_experiment":
            self.run_name = f"{self.model_type}_experiment"

class ConfigLoader:
    """Load and validate training configurations from JSON files"""

    @staticmethod
    def load_from_json(config_path: str) -> UnifiedTrainingConfig:
        """Load configuration from JSON file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"📄 Loading configuration from: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Validate required fields
        if "model_type" not in config_dict:
            raise ValueError("'model_type' is required in config (must be 'sae' or 'transcoder')")

        # Create config object
        try:
            config = UnifiedTrainingConfig(**config_dict)
            logger.info("✅ Configuration loaded successfully")
            return config
        except TypeError as e:
            raise ValueError(f"Invalid configuration: {e}")

    @staticmethod
    def save_to_json(config: UnifiedTrainingConfig, output_path: str):
        """Save configuration to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dict and handle torch.dtype
        config_dict = asdict(config)
        if isinstance(config_dict.get('dtype'), torch.dtype):
            dtype_str_map = {
                torch.float32: "float32",
                torch.float16: "float16",
                torch.bfloat16: "bfloat16",
            }
            config_dict['dtype'] = dtype_str_map.get(config_dict['dtype'], "bfloat16")

        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"💾 Configuration saved to: {output_path}")

    @staticmethod
    def create_example_configs():
        """Create example configuration files"""
        examples_dir = Path("./config_examples")
        examples_dir.mkdir(exist_ok=True)

        # Example 1: Basic SAE configuration
        sae_config = {
            "model_type": "sae",
            "model_name": "HuggingFaceTB/SmolLM2-135M",
            "dataset_name": "EleutherAI/SmolLM2-135M-10B",
            "max_samples": 1000,
            "max_seq_length": 256,
            "expansion_factor": 8,
            "k": 8,
            "batch_size": 2,
            "grad_acc_steps": 8,
            "loss_fn": "fvu",
            "optimizer": "adam",
            "device": "cuda:1",
            "log_to_wandb": False,
            "run_name": "basic_sae"
        }

        # Example 2: Transcoder with specific layers
        transcoder_config = {
            "model_type": "transcoder",
            "model_name": "HuggingFaceTB/SmolLM2-135M",
            "dataset_name": "EleutherAI/SmolLM2-135M-10B",
            "max_samples": 5000,
            "max_seq_length": 256,
            "expansion_factor": 16,
            "k": 16,
            "batch_size": 2,
            "grad_acc_steps": 16,
            "layers": [6, 7, 8],
            "loss_fn": "fvu",
            "optimizer": "adam",
            "lr": 0.0001,
            "device": "cuda:1",
            "log_to_wandb": True,
            "wandb_project": "transcoder-training",
            "run_name": "transcoder_layers_6_7_8"
        }

        # Example 3: Custom hookpoints (attention layers)
        custom_hookpoints_config = {
            "model_type": "sae",
            "model_name": "gpt2",
            "dataset_name": "EleutherAI/SmolLM2-135M-10B",
            "max_samples": 2000,
            "max_seq_length": 512,
            "expansion_factor": 32,
            "k": 32,
            "batch_size": 4,
            "grad_acc_steps": 4,
            "hookpoints": ["h.*.attn", "h.*.mlp.act"],
            "loss_fn": "fvu",
            "optimizer": "adam",
            "device": "cuda:1",
            "run_name": "custom_hookpoints_attn_mlp"
        }

        # Example 4: Advanced training with all options
        advanced_config = {
            "model_type": "sae",
            "model_name": "HuggingFaceTB/SmolLM2-135M",
            "dataset_name": "EleutherAI/SmolLM2-135M-10B",
            "dataset_split": "train",
            "max_samples": 10000,
            "max_seq_length": 1024,
            "expansion_factor": 64,
            "k": 64,
            "activation": "topk",
            "normalize_decoder": True,
            "multi_topk": True,
            "batch_size": 8,
            "grad_acc_steps": 4,
            "micro_acc_steps": 1,
            "loss_fn": "fvu",
            "optimizer": "adam",
            "lr": 0.0001,
            "lr_warmup_steps": 2000,
            "weight_decay": 0.0,
            "auxk_alpha": 0.03125,
            "dead_feature_threshold": 5000000,
            "k_decay_steps": 20000,
            "layer_stride": 2,
            "save_every": 500,
            "save_best": True,
            "log_to_wandb": True,
            "wandb_project": "advanced-sae-training",
            "wandb_log_frequency": 50,
            "device": "cuda:1",
            "dtype": "bfloat16",
            "run_name": "advanced_sae_full_config"
        }

        # Save example configs
        ConfigLoader.save_to_json(UnifiedTrainingConfig(**sae_config), examples_dir / "sae_basic.json")
        ConfigLoader.save_to_json(UnifiedTrainingConfig(**transcoder_config), examples_dir / "transcoder_layers.json")
        ConfigLoader.save_to_json(UnifiedTrainingConfig(**custom_hookpoints_config), examples_dir / "custom_hookpoints.json")
        ConfigLoader.save_to_json(UnifiedTrainingConfig(**advanced_config), examples_dir / "advanced_full.json")

        logger.info(f"✅ Example configurations created in {examples_dir}/")

# ==============================================================================
# TRAINER CLASS
# ==============================================================================

class UnifiedTrainer:
    """Unified trainer for both SAE and Transcoder"""

    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None

        # Display configuration
        self._print_header()

    def _print_header(self):
        """Print training header with key differences"""
        model_type_upper = self.config.model_type.upper()

        logger.info("="*70)
        logger.info(f"🚀 UNIFIED TRAINING - {model_type_upper} MODE")
        logger.info("="*70)

        if self.config.model_type == "sae":
            logger.info("📋 SPARSE AUTOENCODER (SAE) Configuration:")
            logger.info("   • Purpose: Learn interpretable sparse representations")
            logger.info("   • Architecture: Input → Encoder → Sparse Latents → Decoder → Input")
            logger.info("   • Training: Reconstructs input activations (transcode=False)")
            logger.info("   • Loss: Typically FVU (Fraction of Variance Unexplained)")
            logger.info("   • Use Case: Interpretability research, feature discovery")
        else:
            logger.info("📋 TRANSCODER Configuration:")
            logger.info("   • Purpose: End-to-end component replacement")
            logger.info("   • Architecture: Input → Encoder → Sparse Latents → Decoder → Output")
            logger.info("   • Training: Predicts module output from input (transcode=True)")
            logger.info("   • Loss: Typically CE/KL for task performance")
            logger.info("   • Use Case: Model component replacement, performance maintenance")

        logger.info(f"   • Model: {self.config.model_name}")
        logger.info(f"   • Dataset: {self.config.dataset_name}")
        logger.info(f"   • Architecture: {self.config.expansion_factor}x expansion, k={self.config.k}")
        logger.info(f"   • Training: {self.config.batch_size} batch, {self.config.loss_fn} loss")

        # Display hookpoint information
        if self.config.hookpoints:
            logger.info(f"   • Custom Hookpoints: {', '.join(self.config.hookpoints)}")
        elif self.config.layers:
            logger.info(f"   • Layers: {self.config.layers}")
        else:
            logger.info("   • Layers: All (default residual stream)")

        logger.info("="*70)

    def setup_model_and_data(self):
        """Load model, tokenizer, and dataset"""
        logger.info("📥 Loading model and tokenizer...")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map={"": self.config.device},
            torch_dtype=self.config.dtype,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("📚 Loading and processing dataset...")

        # Load dataset with streaming to avoid downloading entire dataset
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            streaming=True
        )

        # Limit dataset size if specified
        if self.config.max_samples:
            dataset = dataset.take(self.config.max_samples)
            logger.info(f"📋 Limited dataset to {self.config.max_samples:,} samples")

        # Convert IterableDataset to Dataset for chunk_and_tokenize compatibility
        dataset = Dataset.from_dict({"text": [item["text"] for item in dataset]})

        # Tokenize dataset
        self.dataset = chunk_and_tokenize(
            dataset,
            self.tokenizer,
            max_seq_len=self.config.max_seq_length
        )

        logger.info(f"✅ Dataset ready: {len(self.dataset):,} samples")

    def create_model_config(self) -> SparseCoderConfig:
        """Create the appropriate model configuration based on type"""
        base_config = {
            "expansion_factor": self.config.expansion_factor,
            "num_latents": self.config.num_latents,
            "k": self.config.k,
            "activation": self.config.activation,
            "normalize_decoder": self.config.normalize_decoder,
            "multi_topk": self.config.multi_topk,
            "skip_connection": self.config.skip_connection,
        }

        if self.config.model_type == "sae":
            # Sparse Autoencoder: transcode=False (default)
            return SaeConfig(**base_config)
        else:
            # Transcoder: transcode=True (automatic in TranscoderConfig)
            return TranscoderConfig(**base_config)

    def create_train_config(self, model_config) -> TrainConfig:
        """Create training configuration"""
        return TrainConfig(
            sae=model_config,

            # Training parameters
            batch_size=self.config.batch_size,
            grad_acc_steps=self.config.grad_acc_steps,
            micro_acc_steps=self.config.micro_acc_steps,
            loss_fn=self.config.loss_fn,

            # Optimization
            optimizer=self.config.optimizer,
            lr=self.config.lr,
            lr_warmup_steps=self.config.lr_warmup_steps,

            # Sparsity
            auxk_alpha=self.config.auxk_alpha,
            dead_feature_threshold=self.config.dead_feature_threshold,
            k_decay_steps=self.config.k_decay_steps,

            # Layer selection (hookpoints takes precedence over layers)
            layers=self.config.layers if self.config.hookpoints is None else None,
            layer_stride=self.config.layer_stride,
            hookpoints=self.config.hookpoints,

            # Training control
            init_seeds=self.config.init_seeds,
            finetune=self.config.finetune,
            exclude_tokens=self.config.exclude_tokens,

            # Distributed
            distribute_modules=self.config.distribute_modules,

            # Saving and logging
            save_dir=self.config.save_dir,
            run_name=self.config.run_name,
            save_every=self.config.save_every,
            save_best=self.config.save_best,
            log_to_wandb=self.config.log_to_wandb,
            wandb_log_frequency=self.config.wandb_log_frequency,
        )

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        if self.config.log_to_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_name,
                config=asdict(self.config),
                tags=[self.config.model_type, "unified_training"],
            )
            logger.info("📊 Weights & Biases logging enabled")

    def train(self) -> Trainer:
        """Execute training based on model type"""
        try:
            # Setup
            self.setup_model_and_data()
            self.setup_wandb()

            # Create configurations
            model_config = self.create_model_config()
            train_config = self.create_train_config(model_config)

            # Log the key difference
            transcode_mode = getattr(model_config, 'transcode', False)
            logger.info(f"🔧 Model configuration: transcode={transcode_mode}")

            # Save configuration
            os.makedirs(self.config.save_dir, exist_ok=True)
            config_path = Path(self.config.save_dir) / f"{self.config.run_name}_config.json"
            ConfigLoader.save_to_json(self.config, str(config_path))

            # Create and run trainer
            logger.info("🏋️ Creating trainer...")
            trainer = Trainer(train_config, self.dataset, self.model)

            logger.info(f"🚀 Starting {self.config.model_type.upper()} training...")
            trainer.fit()

            logger.info(f"✅ {self.config.model_type.upper()} training completed successfully!")

            if self.config.log_to_wandb:
                wandb.finish()

            return trainer

        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main function with JSON-based configuration"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified SAE and Transcoder Training with JSON Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with JSON config
  python train_sae_and_transcoder.py --config config.json

  # Create example configs
  python train_sae_and_transcoder.py --create-examples

  # Train with config and override run name
  python train_sae_and_transcoder.py --config config.json --run-name my_experiment
        """
    )

    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--create-examples", action="store_true", help="Create example configuration files")
    parser.add_argument("--run-name", type=str, help="Override run name from config")
    parser.add_argument("--device", type=str, help="Override device from config")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging (override config)")

    args = parser.parse_args()

    # Create example configs if requested
    if args.create_examples:
        ConfigLoader.create_example_configs()
        logger.info("\n📚 Example configurations created! Check ./config_examples/ directory")
        logger.info("\nYou can now run:")
        logger.info("  python train_sae_and_transcoder.py --config config_examples/sae_basic.json")
        return

    # Require config file for training
    if not args.config:
        parser.print_help()
        logger.error("\n❌ Error: --config is required for training")
        logger.info("💡 Tip: Use --create-examples to generate example configuration files")
        sys.exit(1)

    # Load configuration
    config = ConfigLoader.load_from_json(args.config)

    # Apply command line overrides
    if args.run_name:
        config.run_name = args.run_name
        logger.info(f"🔄 Override: run_name = {args.run_name}")

    if args.device:
        config.device = args.device
        logger.info(f"🔄 Override: device = {args.device}")

    if args.wandb:
        config.log_to_wandb = True
        logger.info("🔄 Override: log_to_wandb = True")

    # Create trainer and run
    trainer = UnifiedTrainer(config)
    trained_model = trainer.train()

    logger.info(f"🎉 Training complete! Checkpoints saved to: {config.save_dir}")

    return trained_model

if __name__ == "__main__":
    main()

# ==============================================================================
# DOCUMENTATION
# ==============================================================================

"""
JSON CONFIGURATION FORMAT:

Basic SAE Example:
{
  "model_type": "sae",
  "model_name": "HuggingFaceTB/SmolLM2-135M",
  "dataset_name": "EleutherAI/SmolLM2-135M-10B",
  "max_samples": 1000,
  "max_seq_length": 256,
  "expansion_factor": 8,
  "k": 8,
  "batch_size": 2,
  "grad_acc_steps": 8,
  "loss_fn": "fvu",
  "device": "cuda:1",
  "run_name": "basic_sae"
}

Transcoder with Specific Layers:
{
  "model_type": "transcoder",
  "layers": [6, 7, 8],
  "expansion_factor": 16,
  "k": 16,
  "batch_size": 2,
  "grad_acc_steps": 16,
  "loss_fn": "fvu",
  "log_to_wandb": true,
  "run_name": "transcoder_layers_6_7_8"
}

Custom Hookpoints (Unix Pattern Matching):
{
  "model_type": "sae",
  "model_name": "gpt2",
  "hookpoints": ["h.*.attn", "h.*.mlp.act"],
  "expansion_factor": 32,
  "k": 32,
  "batch_size": 4,
  "run_name": "custom_hookpoints"
}

Hookpoint Pattern Examples:
- "h.*.attn"         : All attention layers
- "h.*.mlp.act"      : All MLP activations
- "h.[012].attn"     : Attention layers 0, 1, 2
- "h.[0-5].attn"     : Attention layers 0 through 5
- "layers.*"         : All layers (wildcard)
- "embed_tokens"     : Embedding layer

USAGE:

1. Create example configs:
   python train_sae_and_transcoder.py --create-examples

2. Train with config:
   python train_sae_and_transcoder.py --config config.json

3. Train with overrides:
   python train_sae_and_transcoder.py --config config.json --run-name my_exp --wandb

4. Custom config file:
   python train_sae_and_transcoder.py --config my_experiments/custom_config.json
"""
