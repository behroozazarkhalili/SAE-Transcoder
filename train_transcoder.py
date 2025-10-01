#!/usr/bin/env python3
"""
Comprehensive Transcoder Training Script
Using EleutherAI/sparsify library

This script provides complete configuration options for training transcoders
for end-to-end neural network component replacement.
"""

import os
import torch
import json
import logging
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any
from dataclasses import dataclass, asdict
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from sparsify import TranscoderConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TranscoderTrainingConfig:
    """Comprehensive configuration for Transcoder training"""
    
    # Model and Dataset
    model_name: str = "HuggingFaceTB/SmolLM2-135M"
    dataset_name: str = "EleutherAI/SmolLM2-135M-10B"
    dataset_split: str = "train"
    max_samples: Optional[int] = 1_000  # Smaller for end-to-end training
    max_seq_length: int = 1024
    
    # Core Transcoder Architecture
    expansion_factor: int = 32
    num_latents: int = 0  # If > 0, overrides expansion_factor
    k: int = 32  # Number of top-k active features
    activation: Literal["topk", "groupmax"] = "topk"
    normalize_decoder: bool = True
    multi_topk: bool = False
    skip_connection: bool = False
    
    # Training Parameters
    batch_size: int = 16  # Smaller for end-to-end training
    grad_acc_steps: int = 4  # Higher accumulation for stability
    micro_acc_steps: int = 1
    loss_fn: Literal["fvu", "ce", "kl"] = "ce"  # Cross-entropy for end-to-end
    
    # Optimization
    optimizer: Literal["adam", "muon", "signum"] = "adam"
    lr: Optional[float] = None  # Auto-selected if None
    lr_warmup_steps: int = 2000  # Longer warmup for end-to-end
    weight_decay: float = 0.01  # Some regularization for end-to-end
    
    # Sparsity and Regularization
    auxk_alpha: float = 1/32
    dead_feature_threshold: int = 5_000_000  # Stricter for end-to-end
    k_decay_steps: int = 15_000  # Faster decay for end-to-end
    
    # Layer Selection - Typically target specific layers for transcoders
    layers: Optional[List[int]] = None  # e.g., [6, 7, 8] for middle layers
    layer_stride: int = 1
    hookpoints: Optional[List[str]] = None
    
    # Training Control
    max_steps: Optional[int] = None
    init_seeds: List[int] = None
    finetune: bool = False
    exclude_tokens: List[int] = None
    
    # Distributed Training
    distribute_modules: bool = False
    
    # Logging and Saving
    save_dir: str = "./transcoder_checkpoints"
    run_name: str = "transcoder_experiment"
    save_every: int = 2000  # Less frequent saves for end-to-end
    save_best: bool = True
    
    # Weights & Biases
    log_to_wandb: bool = False
    wandb_project: str = "transcoder-training"
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 50  # More frequent logging for end-to-end
    
    # Hardware
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        if self.init_seeds is None:
            self.init_seeds = [42]
        if self.exclude_tokens is None:
            self.exclude_tokens = []

class TranscoderTrainer:
    """Comprehensive Transcoder trainer for end-to-end optimization"""
    
    def __init__(self, config: TranscoderTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
        logger.info(f"🚀 Initializing Transcoder Trainer")
        logger.info(f"📋 Configuration: {self.config.run_name}")
        logger.info(f"🎯 Model: {self.config.model_name}")
        logger.info(f"📊 Dataset: {self.config.dataset_name}")
        
    def setup_model_and_data(self):
        """Load model, tokenizer, and dataset"""
        logger.info("📥 Loading model and tokenizer...")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map={"": "cuda:1"},
            torch_dtype=torch.bfloat16,
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
        
    def create_transcoder_config(self) -> TranscoderConfig:
        """Create Transcoder configuration"""
        return TranscoderConfig(  # Automatically sets transcode=True
            # Core architecture
            expansion_factor=self.config.expansion_factor,
            num_latents=self.config.num_latents,
            k=self.config.k,
            activation=self.config.activation,
            
            # Training features
            normalize_decoder=self.config.normalize_decoder,
            multi_topk=self.config.multi_topk,
            skip_connection=self.config.skip_connection,
            
            # transcode=True is automatically set by TranscoderConfig
        )
        
    def create_train_config(self, transcoder_config: TranscoderConfig) -> TrainConfig:
        """Create training configuration for end-to-end training"""
        return TrainConfig(
            sae=transcoder_config,
            
            # Training parameters
            batch_size=self.config.batch_size,
            grad_acc_steps=self.config.grad_acc_steps,
            micro_acc_steps=self.config.micro_acc_steps,
            loss_fn=self.config.loss_fn,  # Typically 'ce' or 'kl' for transcoders
            
            # Optimization
            optimizer=self.config.optimizer,
            lr=self.config.lr,
            lr_warmup_steps=self.config.lr_warmup_steps,
            
            # Sparsity
            auxk_alpha=self.config.auxk_alpha,
            dead_feature_threshold=self.config.dead_feature_threshold,
            k_decay_steps=self.config.k_decay_steps,
            
            # Layer selection
            layers=self.config.layers,
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
                tags=["transcoder", "end_to_end", "component_replacement"],
            )
            logger.info("📊 Weights & Biases logging enabled")
    
    def print_configuration(self):
        """Print detailed configuration"""
        logger.info("\n" + "="*60)
        logger.info("TRANSCODER CONFIGURATION")
        logger.info("="*60)
        logger.info(f"🏗️  Architecture:")
        logger.info(f"   - Mode: END-TO-END TRANSCODER")
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
        logger.info(f"   - Gradient accumulation: {self.config.grad_acc_steps}")
        logger.info(f"   - Loss function: {self.config.loss_fn} (end-to-end)")
        logger.info(f"   - Optimizer: {self.config.optimizer}")
        logger.info(f"   - Learning rate: {self.config.lr or 'auto'}")
        logger.info(f"   - AuxK alpha: {self.config.auxk_alpha}")
        logger.info(f"   - Target layers: {self.config.layers or 'all'}")
        logger.info(f"")
        logger.info(f"💾 Saving:")
        logger.info(f"   - Save directory: {self.config.save_dir}")
        logger.info(f"   - Save every: {self.config.save_every} steps")
        logger.info(f"   - Save best: {self.config.save_best}")
        logger.info("="*60)
        
    def evaluate_baseline(self):
        """Evaluate baseline model performance before training"""
        logger.info("📊 Evaluating baseline model performance...")
        
        # Get a small sample for evaluation
        sample_size = min(100, len(self.dataset))
        eval_dataset = torch.utils.data.Subset(self.dataset, range(sample_size))
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False)
        
        total_loss = 0.0
        num_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.config.device)
                outputs = self.model(input_ids, labels=input_ids)
                total_loss += outputs.loss.item()
                num_batches += 1
                
                if num_batches >= 50:  # Limit evaluation
                    break
        
        baseline_loss = total_loss / num_batches
        logger.info(f"📈 Baseline model loss: {baseline_loss:.4f}")
        return baseline_loss
        
    def train(self) -> Trainer:
        """Execute Transcoder training"""
        try:
            # Setup
            self.setup_model_and_data()
            self.setup_wandb()
            self.print_configuration()
            
            # Evaluate baseline
            baseline_loss = self.evaluate_baseline()
            
            # Create configurations
            transcoder_config = self.create_transcoder_config()
            train_config = self.create_train_config(transcoder_config)
            
            # Save configuration
            os.makedirs(self.config.save_dir, exist_ok=True)
            config_path = Path(self.config.save_dir) / f"{self.config.run_name}_config.json"
            with open(config_path, "w") as f:
                json.dump(asdict(self.config), f, indent=2, default=str)
            
            # Save baseline performance
            baseline_path = Path(self.config.save_dir) / f"{self.config.run_name}_baseline.json"
            with open(baseline_path, "w") as f:
                json.dump({"baseline_loss": baseline_loss}, f, indent=2)
            
            # Create and run trainer
            logger.info("🏋️ Creating trainer...")
            trainer = Trainer(train_config, self.dataset, self.model)
            
            logger.info("🚀 Starting end-to-end Transcoder training...")
            trainer.fit()
            
            logger.info("✅ Transcoder training completed successfully!")
            
            if self.config.log_to_wandb:
                wandb.finish()
                
            return trainer
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise

def main():
    """Main training function with example configurations"""
    
    # Example 1: Basic transcoder for middle layers
    basic_config = TranscoderTrainingConfig(
        run_name="basic_transcoder",
        expansion_factor=32,
        k=32,
        batch_size=8,
        grad_acc_steps=4,
        layers=[6, 7, 8],  # Target middle layers
        loss_fn="fvu",
        optimizer="adam",
        log_to_wandb=False,
    )
    
    # Example 2: High-performance transcoder
    performance_config = TranscoderTrainingConfig(
        run_name="performance_transcoder",
        expansion_factor=64,
        k=64,
        batch_size=16,
        grad_acc_steps=8,
        layers=[4, 5, 6, 7, 8, 9],  # Multiple layers
        loss_fn="fvu",
        optimizer="muon",  # Better for end-to-end
        lr=2e-3,
        multi_topk=True,
        auxk_alpha=1/64,
        log_to_wandb=True,
        wandb_project="transcoder-performance",
        save_every=1000,
    )
    
    # Example 3: Efficiency-focused transcoder
    efficient_config = TranscoderTrainingConfig(
        run_name="efficient_transcoder",
        expansion_factor=16,
        k=16,
        batch_size=4,
        grad_acc_steps=16,
        layers=[8],  # Single target layer
        loss_fn="kl",  # KL divergence
        optimizer="signum",
        max_samples=25_000,  # Smaller dataset
        log_to_wandb=False,
    )
    
    # Choose configuration
    config = basic_config  # Change this to use different configs
    
    # Create trainer and run
    trainer = TranscoderTrainer(config)
    trained_model = trainer.train()
    
    logger.info(f"🎉 Training complete! Checkpoints saved to: {config.save_dir}")
    
    return trained_model

if __name__ == "__main__":
    main()