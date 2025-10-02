#!/usr/bin/env python3
"""
Unified SAE and Transcoder Training - Main Entry Point

This script provides a unified interface for training both Sparse Autoencoders (SAE)
and Transcoders with comprehensive JSON-based configuration management.

Key Differences:
- SAE (transcode=False): Reconstructs input activations for interpretability
- Transcoder (transcode=True): Predicts module output from input for component replacement

Custom Hookpoints:
- Supports Unix pattern matching for flexible layer selection
- Examples: "h.*.attn", "h.*.mlp.act", "layers.[012]", "h.[0-5].attn"

Usage:
    python -m src.main --config config.json
    python -m src.main --config config.json --run-name my_experiment --wandb
"""

from __future__ import annotations

import argparse
import logging

from .config import ConfigLoader
from .trainer import UnifiedTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function with JSON-based configuration and CLI overrides"""
    parser = argparse.ArgumentParser(
        description="Unified SAE and Transcoder Training with JSON Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with JSON config
  python -m src.main --config config.json

  # Train with config and override run name
  python -m src.main --config config.json --run-name my_experiment

  # Train with W&B logging enabled
  python -m src.main --config config.json --wandb

  # Override device
  python -m src.main --config config.json --device cuda:0

Minimal Config Example (config.json):
  {
    "model_type": "sae"
  }

See config_examples/ directory for more examples.
        """
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file"
    )

    # Optional overrides
    parser.add_argument(
        "--run-name",
        type=str,
        help="Override run name from config"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Override device from config (e.g., cuda:0, cuda:1, cpu)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging (override config)"
    )

    args = parser.parse_args()

    # Load configuration from JSON
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

    # Create trainer and execute training
    trainer = UnifiedTrainer(config)
    trained_model = trainer.train()

    logger.info(f"🎉 Training complete! Checkpoints saved to: {config.save_dir}")

    return trained_model


if __name__ == "__main__":
    main()
