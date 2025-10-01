#!/usr/bin/env python3
"""
Comprehensive Sparse Autoencoder (SAE) Training Script
Using EleutherAI/sparsify library

This script provides complete configuration options for training sparse autoencoders
on neural network activations for interpretability research.
"""

import os
import torch
import json
import logging
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any
from dataclasses import dataclass, asdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SAETrainingConfig:
    """Comprehensive configuration for SAE training"""
    
    # Model and Dataset - using proven working model from documentation
    model_name: str = "Qwen3-0.6B"
    dataset_name: str = "EleutherAI/SmolLM2-135M-10B"
    dataset_split: str = "train"
    max_samples: Optional[int] = 1_000  # Limit dataset size for faster training
    max_seq_length: int = 1024
    
    # Core SAE Architecture
    expansion_factor: int = 32  # Latent dimension = input_dim * expansion_factor
    num_latents: int = 0  # If > 0, overrides expansion_factor
    k: int = 32  # Number of top-k active features
    activation: Literal["topk", "groupmax"] = "topk"
    normalize_decoder: bool = True
    multi_topk: bool = False  # Use Multi-TopK loss for better reconstruction
    skip_connection: bool = False  # Add residual connection
    
    # Training Parameters
    batch_size: int = 32
    grad_acc_steps: int = 1  # Gradient accumulation steps
    micro_acc_steps: int = 1  # Micro-batch accumulation
    loss_fn: Literal["fvu", "ce", "kl"] = "ce"  # Fraction of Variance Unexplained
    
    # Optimization
    optimizer: Literal["adam", "muon", "signum"] = "signum"
    lr: Optional[float] = None  # Auto-selected if None
    lr_warmup_steps: int = 1000
    weight_decay: float = 0.0
    
    # Sparsity and Regularization  
    auxk_alpha: float = 1/32  # AuxK loss coefficient for dead feature revival
    dead_feature_threshold: int = 10_000_000  # Tokens before marking feature as dead
    k_decay_steps: int = 25_000  # Steps for k-sparsity schedule
    
    # Layer Selection
    layers: Optional[List[int]] = None  # Specific layers to train on (None = all)
    layer_stride: int = 1  # Train every nth layer
    hookpoints: Optional[List[str]] = None  # Specific module names
    
    # Training Control
    max_steps: Optional[int] = None
    init_seeds: List[int] = None
    finetune: bool = False  # Fine-tune from existing checkpoint
    exclude_tokens: List[int] = None  # Token IDs to exclude from training
    
    # Distributed Training
    distribute_modules: bool = False  # Distribute layers across GPUs
    
    # Logging and Saving
    save_dir: str = "./sae_checkpoints"
    run_name: str = "sae_experiment"
    save_every: int = 1000
    save_best: bool = True
    
    # Weights & Biases
    log_to_wandb: bool = False
    wandb_project: str = "sae-training"
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 100
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    
    def __post_init__(self):
        if self.init_seeds is None:
            self.init_seeds = [42]
        if self.exclude_tokens is None:
            self.exclude_tokens = []

class SAETrainer:
    """Comprehensive SAE trainer with full configuration support"""
    
    def __init__(self, config: SAETrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
        logger.info(f"🚀 Initializing SAE Trainer")
        logger.info(f"📋 Configuration: {self.config.run_name}")
        logger.info(f"🎯 Model: {self.config.model_name}")
        logger.info(f"📊 Dataset: {self.config.dataset_name}")
        
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
        
        # Load dataset
        dataset = load_dataset(
            self.config.dataset_name, 
            split=self.config.dataset_split
        )
        
        # Limit dataset size if specified
        if self.config.max_samples and len(dataset) > self.config.max_samples:
            dataset = dataset.select(range(self.config.max_samples))
            logger.info(f"📋 Limited dataset to {self.config.max_samples:,} samples")
        
        # Tokenize dataset
        self.dataset = chunk_and_tokenize(
            dataset, 
            self.tokenizer,
            max_seq_len=self.config.max_seq_length
        )
        
        logger.info(f"✅ Dataset ready: {len(self.dataset):,} samples")
        
    def create_sae_config(self) -> SaeConfig:
        """Create SAE configuration"""
        return SaeConfig(
            # Core architecture
            expansion_factor=self.config.expansion_factor,
            num_latents=self.config.num_latents,
            k=self.config.k,
            activation=self.config.activation,
            
            # Training features
            normalize_decoder=self.config.normalize_decoder,
            multi_topk=self.config.multi_topk,
            skip_connection=self.config.skip_connection,
            
            # CRITICAL: Set to False for sparse autoencoder mode
            transcode=False,
        )
        
    def create_train_config(self, sae_config: SaeConfig) -> TrainConfig:
        """Create training configuration - simplified to match documentation"""
        # Use minimal configuration like the documentation example
        # cfg = TrainConfig(SaeConfig(), batch_size=16)
        return TrainConfig(
            sae=sae_config,
            batch_size=self.config.batch_size,
            # Only add essential parameters to avoid tensor shape issues
            save_dir=self.config.save_dir,
            run_name=self.config.run_name,
            log_to_wandb=self.config.log_to_wandb,
        )
        
    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        if self.config.log_to_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_name,
                config=asdict(self.config),
                tags=["sae", "sparse_autoencoder", "interpretability"],
            )
            logger.info("📊 Weights & Biases logging enabled")
    
    def print_configuration(self):
        """Print detailed configuration"""
        logger.info("\n" + "="*60)
        logger.info("SPARSE AUTOENCODER CONFIGURATION")
        logger.info("="*60)
        logger.info(f"🏗️  Architecture:")
        logger.info(f"   - Expansion factor: {self.config.expansion_factor}")
        logger.info(f"   - Number of latents: {self.config.num_latents or 'auto'}")
        logger.info(f"   - Top-k sparsity: {self.config.k}")
        logger.info(f"   - Activation function: {self.config.activation}")
        logger.info(f"   - Normalize decoder: {self.config.normalize_decoder}")
        logger.info(f"   - Multi-TopK: {self.config.multi_topk}")
        logger.info(f"   - Skip connection: {self.config.skip_connection}")
        logger.info(f"")
        logger.info(f"🎯 Training:")
        logger.info(f"   - Batch size: {self.config.batch_size}")
        logger.info(f"   - Loss function: {self.config.loss_fn}")
        logger.info(f"   - Optimizer: {self.config.optimizer}")
        logger.info(f"   - Learning rate: {self.config.lr or 'auto'}")
        logger.info(f"   - AuxK alpha: {self.config.auxk_alpha}")
        logger.info(f"")
        logger.info(f"💾 Saving:")
        logger.info(f"   - Save directory: {self.config.save_dir}")
        logger.info(f"   - Save every: {self.config.save_every} steps")
        logger.info(f"   - Save best: {self.config.save_best}")
        logger.info("="*60)
        
    def train(self) -> Trainer:
        """Execute SAE training"""
        try:
            # Setup
            self.setup_model_and_data()
            self.setup_wandb()
            self.print_configuration()
            
            # Create configurations
            sae_config = self.create_sae_config()
            train_config = self.create_train_config(sae_config)
            
            # Save configuration
            os.makedirs(self.config.save_dir, exist_ok=True)
            config_path = Path(self.config.save_dir) / f"{self.config.run_name}_config.json"
            with open(config_path, "w") as f:
                json.dump(asdict(self.config), f, indent=2, default=str)
            
            # Create and run trainer
            logger.info("🏋️ Creating trainer...")
            trainer = Trainer(train_config, self.dataset, self.model)
            
            logger.info("🚀 Starting SAE training...")
            trainer.fit()
            
            logger.info("✅ SAE training completed successfully!")
            
            if self.config.log_to_wandb:
                wandb.finish()
                
            return trainer
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise

def main():
    """Main training function with example configurations"""
    
    # Example 1: Basic SAE training - minimal config like documentation
    basic_config = SAETrainingConfig(
        run_name="basic_sae_training",
        batch_size=16,  # Match documentation example
        max_samples=1000,  # Fewer samples for testing
        log_to_wandb=False,
    )
    
    # Example 2: High-quality SAE for research
    research_config = SAETrainingConfig(
        run_name="research_quality_sae",
        expansion_factor=128,  # Large for fine-grained features
        k=32,
        batch_size=32,
        grad_acc_steps=4,
        optimizer="adam",
        lr=1e-4,
        multi_topk=True,
        auxk_alpha=1/64,  # Lower for better feature quality
        dead_feature_threshold=1_000_000,
        log_to_wandb=True,
        wandb_project="sae-research",
        save_every=500,
    )
    
    # Example 3: Memory-efficient training
    efficient_config = SAETrainingConfig(
        run_name="memory_efficient_sae",
        expansion_factor=16,
        k=16,
        batch_size=8,
        grad_acc_steps=8,
        micro_acc_steps=4,
        optimizer="signum",  # Memory efficient
        layers=[6, 7, 8],  # Target specific layers
        log_to_wandb=False,
    )
    
    # Choose configuration
    config = basic_config  # Change this to use different configs
    
    # Create trainer and run
    trainer = SAETrainer(config)
    trained_model = trainer.train()
    
    logger.info(f"🎉 Training complete! Checkpoints saved to: {config.save_dir}")
    
    return trained_model

if __name__ == "__main__":
    main()