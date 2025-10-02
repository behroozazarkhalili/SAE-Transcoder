"""
Configuration management for SAE and Transcoder training.

This module provides:
- UnifiedTrainingConfig: Comprehensive dataclass with all training parameters
- ConfigLoader: JSON-based configuration loading and saving
"""

from __future__ import annotations

import json
import logging
import torch
from pathlib import Path
from typing import Optional, List, Literal
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTrainingConfig:
    """Unified configuration for both SAE and Transcoder training with comprehensive parameters"""

    # ==== MODEL TYPE SELECTION ====
    model_type: Literal["sae", "transcoder"] = "sae"  # Training mode: SAE (reconstruction) or Transcoder (end-to-end)

    # ==== MODEL AND DATASET ====
    model_name: str = "HuggingFaceTB/SmolLM2-135M"  # HuggingFace model identifier
    dataset_name: str = "EleutherAI/SmolLM2-135M-10B"  # HuggingFace dataset identifier
    dataset_split: str = "train"  # Dataset split to use (train/validation/test)
    max_samples: Optional[int] = 1_000  # Maximum number of samples to use (None = use all)
    max_seq_length: int = 256  # Maximum sequence length for tokenization

    # ==== CORE ARCHITECTURE ====
    expansion_factor: int = 8  # Dictionary size multiplier (d_model * expansion_factor)
    num_latents: int = 0  # Explicit number of latent features (overrides expansion_factor if > 0)
    k: int = 8  # Number of top-k active features in sparse representation
    activation: Literal["topk", "groupmax"] = "topk"  # Activation function type
    normalize_decoder: bool = True  # Normalize decoder weights to unit norm
    multi_topk: bool = False  # Use multi-scale top-k activation
    skip_connection: bool = False  # Add skip connection from input to output

    # ==== TRAINING PARAMETERS ====
    batch_size: int = 2  # Number of samples per batch
    grad_acc_steps: int = 8  # Gradient accumulation steps (effective batch = batch_size * grad_acc_steps)
    micro_acc_steps: int = 1  # Micro-batching for memory efficiency
    loss_fn: Literal["fvu", "ce", "kl"] = "fvu"  # Loss function: FVU (variance) / CE (cross-entropy) / KL (divergence)

    # ==== OPTIMIZATION ====
    optimizer: Literal["adam", "muon", "signum"] = "adam"  # Optimizer algorithm
    lr: Optional[float] = None  # Learning rate (None = auto-select based on model)
    lr_warmup_steps: int = 1000  # Number of warmup steps for learning rate schedule
    weight_decay: float = 0.0  # L2 regularization weight decay

    # ==== SPARSITY AND REGULARIZATION ====
    auxk_alpha: float = 1/32  # Auxiliary loss weight for encouraging sparsity (default: 0.03125)
    dead_feature_threshold: int = 5_000_000  # Number of tokens before considering feature "dead"
    k_decay_steps: int = 20_000  # Steps over which to decay k value (for curriculum learning)

    # ==== LAYER SELECTION ====
    layers: Optional[List[int]] = None  # Specific layer indices to train on (e.g., [6, 7, 8])
    layer_stride: int = 1  # Stride for layer selection (e.g., 2 = every other layer)
    hookpoints: Optional[List[str]] = None  # Custom hookpoints with Unix patterns (e.g., ["h.*.attn", "h.*.mlp.act"])

    # ==== TRAINING CONTROL ====
    max_steps: Optional[int] = None  # Maximum training steps (None = train on full dataset)
    init_seeds: Optional[List[int]] = None  # Random seeds for initialization (supports multi-seed training)
    finetune: bool = False  # Whether to finetune from existing checkpoint
    exclude_tokens: Optional[List[int]] = None  # Token IDs to exclude from training (e.g., padding tokens)

    # ==== DISTRIBUTED TRAINING ====
    distribute_modules: bool = False  # Distribute SAE modules across multiple GPUs

    # ==== LOGGING AND SAVING ====
    save_dir: str = "./checkpoints"  # Directory to save model checkpoints
    run_name: str = "unified_experiment"  # Experiment name for logging and checkpoints
    save_every: int = 1000  # Save checkpoint every N steps
    save_best: bool = True  # Save best checkpoint based on validation loss

    # ==== WEIGHTS & BIASES ====
    log_to_wandb: bool = False  # Enable Weights & Biases logging
    wandb_project: str = "sae-transcoder-unified"  # W&B project name
    wandb_entity: Optional[str] = None  # W&B entity (username or team name)
    wandb_log_frequency: int = 100  # Log metrics to W&B every N steps

    # ==== HARDWARE ====
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"  # Device for training (cuda:0, cuda:1, cpu)
    dtype: str = "bfloat16"  # Floating point precision (float32, float16, bfloat16)

    def __post_init__(self):
        """Validate and adjust configuration after initialization"""
        # Set default values for mutable fields
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
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            UnifiedTrainingConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid or missing required fields
        """
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
        """
        Save configuration to JSON file.

        Args:
            config: Configuration object to save
            output_path: Path where to save JSON file
        """
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
