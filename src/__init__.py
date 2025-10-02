"""
SAE and Transcoder Training Package

This package provides a unified interface for training Sparse Autoencoders (SAE)
and Transcoders using the EleutherAI/sparsify library.

Main components:
- config: Configuration management (UnifiedTrainingConfig, ConfigLoader)
- trainer: Training orchestration (UnifiedTrainer)
- main: CLI entry point
"""

from .config import UnifiedTrainingConfig, ConfigLoader
from .trainer import UnifiedTrainer

__all__ = ["UnifiedTrainingConfig", "ConfigLoader", "UnifiedTrainer"]
