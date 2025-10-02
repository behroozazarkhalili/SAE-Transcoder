"""
Unified trainer for SAE and Transcoder models.

This module provides the UnifiedTrainer class that handles:
- Model and dataset setup
- Training configuration creation
- Weights & Biases integration
- Training execution and checkpoint management
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import asdict

import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify import SaeConfig, TranscoderConfig, Trainer, TrainConfig
from sparsify.config import SparseCoderConfig
from sparsify.data import chunk_and_tokenize

from .config import UnifiedTrainingConfig, ConfigLoader

logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """Unified trainer for both SAE and Transcoder models"""

    def __init__(self, config: UnifiedTrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: UnifiedTrainingConfig instance with all training parameters
        """
        self.config = config
        self.model = None  # Will be loaded in setup_model_and_data()
        self.tokenizer = None  # Will be loaded in setup_model_and_data()
        self.dataset = None  # Will be loaded in setup_model_and_data()

        # Display configuration
        self._print_header()

    def _print_header(self):
        """Print training header with key differences between SAE and Transcoder"""
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
        """Load model, tokenizer, and dataset with streaming support"""
        logger.info("📥 Loading model and tokenizer...")

        # Load model with specified device and dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map={"": self.config.device},
            torch_dtype=self.config.dtype,
        )

        # Load tokenizer and handle padding token
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
        """
        Create the appropriate model configuration based on type.

        Returns:
            SparseCoderConfig instance (either SaeConfig or TranscoderConfig)
        """
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

    def create_train_config(self, model_config: SparseCoderConfig) -> TrainConfig:
        """
        Create training configuration from unified config.

        Args:
            model_config: Model architecture configuration (SaeConfig or TranscoderConfig)

        Returns:
            TrainConfig instance for sparsify Trainer
        """
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

            # Sparsity and regularization
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

            # Distributed training
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
        """Setup Weights & Biases logging if enabled"""
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
        """
        Execute training based on model type.

        Returns:
            Trained sparsify Trainer instance

        Raises:
            Exception: If training fails
        """
        try:
            # Setup model, data, and logging
            self.setup_model_and_data()
            self.setup_wandb()

            # Create configurations
            model_config = self.create_model_config()
            train_config = self.create_train_config(model_config)

            # Log the key difference
            transcode_mode = getattr(model_config, 'transcode', False)
            logger.info(f"🔧 Model configuration: transcode={transcode_mode}")

            # Save configuration for reproducibility
            os.makedirs(self.config.save_dir, exist_ok=True)
            config_path = Path(self.config.save_dir) / f"{self.config.run_name}_config.json"
            ConfigLoader.save_to_json(self.config, str(config_path))

            # Create and run trainer
            logger.info("🏋️ Creating trainer...")
            trainer = Trainer(train_config, self.dataset, self.model)

            logger.info(f"🚀 Starting {self.config.model_type.upper()} training...")
            trainer.fit()

            logger.info(f"✅ {self.config.model_type.upper()} training completed successfully!")

            # Cleanup W&B
            if self.config.log_to_wandb:
                wandb.finish()

            return trainer

        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise e
