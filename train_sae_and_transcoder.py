#!/usr/bin/env python3
"""
Unified SAE and Transcoder Training Script
Using EleutherAI/sparsify library

This script provides a unified interface for training both Sparse Autoencoders (SAE)
and Transcoders with comprehensive configuration options.

Key Differences:
- SAE (transcode=False): Reconstructs input activations for interpretability
- Transcoder (transcode=True): Predicts module output from input for component replacement
"""

from __future__ import annotations

import os
import torch
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any, Union
from dataclasses import dataclass, asdict
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from sparsify import SaeConfig, TranscoderConfig, Trainer, TrainConfig
from sparsify.config import SparseCoderConfig
from sparsify.data import chunk_and_tokenize

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    max_seq_length: int = 1024
    
    # ==== CORE ARCHITECTURE ====
    expansion_factor: int = 32  # Latent dimension = input_dim * expansion_factor
    num_latents: int = 0  # If > 0, overrides expansion_factor
    k: int = 32  # Number of top-k active features
    activation: Literal["topk", "groupmax"] = "topk"
    normalize_decoder: bool = True
    multi_topk: bool = False  # Multi-TopK loss for better reconstruction
    skip_connection: bool = False  # Add residual connection
    
    # ==== TRAINING PARAMETERS ====
    batch_size: int = 16
    grad_acc_steps: int = 2  # Gradient accumulation steps
    micro_acc_steps: int = 1  # Micro-batch accumulation
    loss_fn: Literal["fvu", "ce", "kl"] = "fvu"  # Loss function
    
    # ==== OPTIMIZATION ====
    optimizer: Literal["adam", "muon", "signum"] = "adam"
    lr: Optional[float] = None  # Auto-selected if None
    lr_warmup_steps: int = 1000
    weight_decay: float = 0.0
    
    # ==== SPARSITY AND REGULARIZATION ====
    auxk_alpha: float = 1/32  # AuxK loss coefficient for dead feature revival
    dead_feature_threshold: int = 5_000_000  # Tokens before marking feature as dead
    k_decay_steps: int = 20_000  # Steps for k-sparsity schedule
    
    # ==== LAYER SELECTION ====
    layers: Optional[List[int]] = None  # Specific layers (None = all)
    layer_stride: int = 1  # Train every nth layer
    hookpoints: Optional[List[str]] = None  # Specific module names
    
    # ==== TRAINING CONTROL ====
    max_steps: Optional[int] = None
    init_seeds: List[int] = None
    finetune: bool = False
    exclude_tokens: List[int] = None
    
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        """Set defaults and model-type specific adjustments"""
        if self.init_seeds is None:
            self.init_seeds = [42]
        if self.exclude_tokens is None:
            self.exclude_tokens = []
            
        # Auto-adjust parameters based on model type
        if self.model_type == "transcoder":
            # Transcoder-specific adjustments
            # Note: Using FVU loss for transcoder to avoid Identity.forward() issue with decoder models
            if self.loss_fn == "ce":
                self.loss_fn = "fvu" 
                logger.info("🔄 Auto-adjusted loss_fn to 'fvu' for transcoder")

            if self.grad_acc_steps < 2:
                self.grad_acc_steps = 4  # More accumulation for stability
                logger.info("🔄 Auto-adjusted grad_acc_steps to 4 for transcoder")

            if self.batch_size > 16:
                self.batch_size = 16  # Smaller batches for end-to-end
                logger.info("🔄 Auto-adjusted batch_size to 16 for transcoder")
                
        # Update save directory and run name
        self.save_dir = f"./{self.model_type}_checkpoints"
        if self.run_name == "unified_experiment":
            self.run_name = f"{self.model_type}_experiment"

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
        logger.info("="*70)
        
    def setup_model_and_data(self):
        """Load model, tokenizer, and dataset"""
        logger.info("📥 Loading model and tokenizer...")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map={"": self.config.device},
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
            with open(config_path, "w") as f:
                config_dict = asdict(self.config)
                config_dict['transcode_mode'] = transcode_mode
                json.dump(config_dict, f, indent=2, default=str)
            
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
# COMMAND LINE INTERFACE
# ==============================================================================

def create_parser():
    """Create comprehensive command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Unified SAE and Transcoder Training with Comprehensive Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ==== MODEL TYPE ====
    parser.add_argument("--model-type", choices=["sae", "transcoder"],
                       default="transcoder", help="Model type to train")

    # ==== MODEL AND DATASET ====
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-135M",
                       help="Model name from HuggingFace")
    parser.add_argument("--dataset-name", default="EleutherAI/SmolLM2-135M-10B",
                       help="Dataset name from HuggingFace")
    parser.add_argument("--dataset-split", default="train",
                       help="Dataset split to use")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum training samples")
    parser.add_argument("--max-seq-length", type=int, default=256,
                       help="Maximum sequence length")

    # ==== ARCHITECTURE ====
    parser.add_argument("--expansion-factor", type=int, default=4,
                       help="Latent expansion factor")
    parser.add_argument("--num-latents", type=int, default=0,
                       help="Number of latents (overrides expansion-factor if > 0)")
    parser.add_argument("--k", type=int, default=4,
                       help="Top-k sparsity")
    parser.add_argument("--activation", choices=["topk", "groupmax"], default="topk",
                       help="Activation function")
    parser.add_argument("--normalize-decoder", action="store_true", default=True,
                       help="Normalize decoder weights")
    parser.add_argument("--multi-topk", action="store_true",
                       help="Use multi-TopK loss")
    parser.add_argument("--skip-connection", action="store_true",
                       help="Add skip/residual connection")

    # ==== TRAINING PARAMETERS ====
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--grad-acc-steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--micro-acc-steps", type=int, default=1,
                       help="Micro-batch accumulation steps")
    parser.add_argument("--loss-fn", choices=["fvu", "ce", "kl"], default="fvu",
                       help="Loss function")

    # ==== OPTIMIZATION ====
    parser.add_argument("--optimizer", choices=["adam", "muon", "signum"],
                       default="adam", help="Optimizer")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (auto-selected if None)")
    parser.add_argument("--lr-warmup-steps", type=int, default=1000,
                       help="Learning rate warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                       help="Weight decay")

    # ==== SPARSITY AND REGULARIZATION ====
    parser.add_argument("--auxk-alpha", type=float, default=1/32,
                       help="AuxK loss coefficient")
    parser.add_argument("--dead-feature-threshold", type=int, default=5_000_000,
                       help="Dead feature threshold in tokens")
    parser.add_argument("--k-decay-steps", type=int, default=20_000,
                       help="K-sparsity decay schedule steps")

    # ==== LAYER SELECTION ====
    parser.add_argument("--layers", nargs="+", type=int,
                       help="Specific layers to train (e.g., --layers 6 7 8)")
    parser.add_argument("--layer-stride", type=int, default=1,
                       help="Train every nth layer")
    parser.add_argument("--hookpoints", nargs="+", type=str,
                       help="Specific module names to hook")

    # ==== TRAINING CONTROL ====
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum training steps")
    parser.add_argument("--finetune", action="store_true",
                       help="Finetune from checkpoint")

    # ==== DISTRIBUTED TRAINING ====
    parser.add_argument("--distribute-modules", action="store_true",
                       help="Distribute modules across devices")

    # ==== SAVING ====
    parser.add_argument("--save-dir", default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--run-name", default="unified_experiment",
                       help="Run name for saving and logging")
    parser.add_argument("--save-every", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--save-best", action="store_true", default=True,
                       help="Save best model")

    # ==== WEIGHTS & BIASES ====
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="sae-transcoder-unified",
                       help="W&B project name")
    parser.add_argument("--wandb-entity", default=None,
                       help="W&B entity name")
    parser.add_argument("--wandb-log-frequency", type=int, default=100,
                       help="W&B logging frequency")

    # ==== HARDWARE ====
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")

    return parser

def main():
    """Main function with comprehensive CLI support"""
    parser = create_parser()
    args = parser.parse_args()

    # Create comprehensive config from CLI arguments
    config = UnifiedTrainingConfig(
        # Model type
        model_type=args.model_type,

        # Model and dataset
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,

        # Architecture
        expansion_factor=args.expansion_factor,
        num_latents=args.num_latents,
        k=args.k,
        activation=args.activation,
        normalize_decoder=args.normalize_decoder,
        multi_topk=args.multi_topk,
        skip_connection=args.skip_connection,

        # Training parameters
        batch_size=args.batch_size,
        grad_acc_steps=args.grad_acc_steps,
        micro_acc_steps=args.micro_acc_steps,
        loss_fn=args.loss_fn,

        # Optimization
        optimizer=args.optimizer,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        weight_decay=args.weight_decay,

        # Sparsity and regularization
        auxk_alpha=args.auxk_alpha,
        dead_feature_threshold=args.dead_feature_threshold,
        k_decay_steps=args.k_decay_steps,

        # Layer selection
        layers=args.layers,
        layer_stride=args.layer_stride,
        hookpoints=args.hookpoints,

        # Training control
        max_steps=args.max_steps,
        finetune=args.finetune,

        # Distributed
        distribute_modules=args.distribute_modules,

        # Saving
        save_dir=args.save_dir,
        run_name=args.run_name,
        save_every=args.save_every,
        save_best=args.save_best,

        # Weights & Biases
        log_to_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_log_frequency=args.wandb_log_frequency,

        # Hardware
        device=args.device,
    )

    # Create trainer and run
    trainer = UnifiedTrainer(config)
    trained_model = trainer.train()

    logger.info(f"🎉 Training complete! Checkpoints saved to: {config.save_dir}")

    return trained_model

if __name__ == "__main__":
    main()

# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

"""
COMMAND LINE USAGE EXAMPLES:

1. Basic SAE training (default configuration):
   python train_sae_and_transcoder.py

2. SAE with custom parameters:
   python train_sae_and_transcoder.py --model-type sae --expansion-factor 16 --k 16 --batch-size 4

3. SAE with W&B logging:
   python train_sae_and_transcoder.py --model-type sae --wandb --run-name my_sae_experiment

4. Transcoder training with specific layers:
   python train_sae_and_transcoder.py --model-type transcoder --layers 6 7 8 --loss-fn ce

5. Transcoder with all options:
   python train_sae_and_transcoder.py \
       --model-type transcoder \
       --expansion-factor 32 \
       --k 32 \
       --batch-size 2 \
       --grad-acc-steps 16 \
       --layers 6 7 8 \
       --loss-fn ce \
       --optimizer adam \
       --lr 1e-4 \
       --max-samples 10000 \
       --wandb \
       --run-name comprehensive_transcoder

6. Full configuration with all supported parameters:
   python train_sae_and_transcoder.py \
       --model-type sae \
       --model-name HuggingFaceTB/SmolLM2-135M \
       --dataset-name EleutherAI/SmolLM2-135M-10B \
       --max-samples 5000 \
       --max-seq-length 256 \
       --expansion-factor 8 \
       --k 8 \
       --activation topk \
       --batch-size 2 \
       --grad-acc-steps 8 \
       --loss-fn fvu \
       --optimizer adam \
       --lr-warmup-steps 1000 \
       --auxk-alpha 0.03125 \
       --save-every 500 \
       --wandb \
       --wandb-project my_project \
       --run-name full_config_test \
       --device cuda:1

PROGRAMMATIC USAGE:

# SAE Training
from train_sae_and_transcoder import UnifiedTrainingConfig, UnifiedTrainer

sae_config = UnifiedTrainingConfig(
    model_type="sae",
    expansion_factor=8,
    k=8,
    batch_size=2,
    loss_fn="fvu",
    max_samples=1000,
)
trainer = UnifiedTrainer(sae_config)
trainer.train()

# Transcoder Training with comprehensive options
transcoder_config = UnifiedTrainingConfig(
    model_type="transcoder",
    expansion_factor=16,
    k=16,
    batch_size=2,
    grad_acc_steps=16,
    loss_fn="ce",
    layers=[6, 7, 8],
    max_samples=5000,
    optimizer="adam",
    lr=1e-4,
    multi_topk=True,
    log_to_wandb=True,
    run_name="my_transcoder",
)
trainer = UnifiedTrainer(transcoder_config)
trainer.train()

# Quick help to see all available parameters:
#   python train_sae_and_transcoder.py --help
"""