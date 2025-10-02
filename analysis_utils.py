#!/usr/bin/env python3
"""
Comprehensive analysis utilities for trained SAEs and Transcoders
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass

from sparsify import SparseCoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResults:
    """Container for analysis results"""
    model_type: str  # 'sae' or 'transcoder'
    model_path: str
    sparsity_metrics: Dict[str, float]
    reconstruction_metrics: Dict[str, float]
    feature_metrics: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None

class ModelAnalyzer:
    """Comprehensive analyzer for SAE/Transcoder models"""
    
    def __init__(self, model_path: str, model_type: str = "auto"):
        """
        Initialize analyzer
        
        Args:
            model_path: Path to saved model directory
            model_type: 'sae', 'transcoder', or 'auto' to detect from config
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model = None
        self.config = None
        
        self._load_model()
        
    def _load_model(self):
        """Load model and configuration"""
        try:
            # Load model
            self.model = SparseCoder.from_disk(self.model_path)
            logger.info(f"✅ Loaded model from {self.model_path}")
            
            # Load configuration
            config_path = self.model_path / "cfg.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                    
                # Auto-detect model type
                if self.model_type == "auto":
                    self.model_type = "transcoder" if self.config.get("transcode", False) else "sae"
                    
            logger.info(f"📋 Model type: {self.model_type.upper()}")
            logger.info(f"🏗️  Architecture: {self.model.num_latents:,} latents, k={self.model.cfg.k}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
            
    def analyze_sparsity(self, activations: torch.Tensor) -> Dict[str, float]:
        """Analyze sparsity patterns of the model"""
        logger.info("📊 Analyzing sparsity patterns...")
        
        with torch.no_grad():
            # Encode activations
            encoded = self.model.encode(activations)
            
            # Basic sparsity metrics
            total_features = self.model.num_latents
            batch_size = activations.shape[0]
            
            # Active features per sample
            active_per_sample = encoded.top_acts.shape[1]  # Should equal k
            
            # Unique active features across batch
            unique_active = torch.unique(encoded.top_indices).numel()
            
            # Feature activation frequency
            feature_counts = torch.zeros(total_features)
            feature_counts.index_add_(0, encoded.top_indices.flatten(), 
                                    torch.ones_like(encoded.top_indices.flatten(), dtype=torch.float))
            
            # Dead features (never activated)
            dead_features = (feature_counts == 0).sum().item()
            
            # Frequently activated features (>10% of samples)
            frequent_threshold = batch_size * 0.1
            frequent_features = (feature_counts > frequent_threshold).sum().item()
            
            # Sparsity statistics
            sparsity_ratio = active_per_sample / total_features
            feature_utilization = unique_active / total_features
            
            # L0 norm statistics
            l0_norms = (encoded.top_acts != 0).sum(dim=1).float()
            
            metrics = {
                "total_features": total_features,
                "active_per_sample": active_per_sample,
                "sparsity_ratio": sparsity_ratio,
                "unique_active_features": unique_active,
                "feature_utilization": feature_utilization,
                "dead_features": dead_features,
                "dead_feature_percentage": dead_features / total_features * 100,
                "frequent_features": frequent_features,
                "frequent_feature_percentage": frequent_features / total_features * 100,
                "mean_l0": l0_norms.mean().item(),
                "std_l0": l0_norms.std().item(),
                "max_activation": encoded.top_acts.max().item(),
                "mean_activation": encoded.top_acts.mean().item(),
            }
            
        logger.info(f"✅ Sparsity analysis complete")
        logger.info(f"   Active features per sample: {active_per_sample}/{total_features} ({sparsity_ratio:.2%})")
        logger.info(f"   Feature utilization: {unique_active}/{total_features} ({feature_utilization:.2%})")
        logger.info(f"   Dead features: {dead_features} ({dead_features/total_features:.2%})")
        
        return metrics
        
    def analyze_reconstruction(self, activations: torch.Tensor) -> Dict[str, float]:
        """Analyze reconstruction quality"""
        logger.info("🔄 Analyzing reconstruction quality...")
        
        with torch.no_grad():
            # Forward pass
            if self.model_type == "sae":
                # For SAE: input = target
                reconstruction = self.model(activations).sae_out
                target = activations
            else:
                # For transcoder: would need both input and target
                # For this analysis, we'll use the same approach
                reconstruction = self.model(activations).sae_out  
                target = activations
            
            # Reconstruction metrics
            mse = torch.nn.functional.mse_loss(reconstruction, target)
            mae = torch.nn.functional.l1_loss(reconstruction, target)
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                reconstruction.flatten(), target.flatten(), dim=0
            )
            
            # Fraction of variance explained (1 - FVU)
            target_var = target.var()
            residual_var = (target - reconstruction).var()
            fve = 1 - (residual_var / target_var)
            
            # Pearson correlation
            reconstruction_flat = reconstruction.flatten()
            target_flat = target.flatten()
            correlation = torch.corrcoef(torch.stack([reconstruction_flat, target_flat]))[0, 1]
            
            # Relative error norms
            relative_l2_error = (reconstruction - target).norm() / target.norm()
            
            metrics = {
                "mse": mse.item(),
                "mae": mae.item(),
                "rmse": torch.sqrt(mse).item(),
                "cosine_similarity": cos_sim.item(),
                "fraction_variance_explained": fve.item(),
                "pearson_correlation": correlation.item(),
                "relative_l2_error": relative_l2_error.item(),
                "target_norm": target.norm().item(),
                "reconstruction_norm": reconstruction.norm().item(),
                "residual_norm": (target - reconstruction).norm().item(),
            }
            
        logger.info(f"✅ Reconstruction analysis complete")
        logger.info(f"   MSE: {mse:.6f}")
        logger.info(f"   Cosine similarity: {cos_sim:.4f}")
        logger.info(f"   Fraction variance explained: {fve:.4f}")
        
        return metrics
        
    def analyze_features(self, activations: torch.Tensor, top_k: int = 20) -> Dict[str, Any]:
        """Analyze individual feature properties"""
        logger.info(f"🔍 Analyzing top {top_k} features...")
        
        with torch.no_grad():
            encoded = self.model.encode(activations)
            
            # Feature activation statistics
            feature_counts = torch.zeros(self.model.num_latents)
            feature_acts = torch.zeros(self.model.num_latents)
            
            # Count activations and sum magnitudes
            indices = encoded.top_indices.flatten()
            acts = encoded.top_acts.flatten()
            
            feature_counts.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
            feature_acts.index_add_(0, indices, acts)
            
            # Average activation when active
            avg_acts = torch.where(feature_counts > 0, feature_acts / feature_counts, 
                                 torch.zeros_like(feature_acts))
            
            # Most frequently activated features
            most_frequent = torch.topk(feature_counts, top_k)
            
            # Highest average activation features
            highest_avg = torch.topk(avg_acts, top_k)
            
            # Most total activation features
            highest_total = torch.topk(feature_acts, top_k)
            
            metrics = {
                "most_frequent_features": {
                    "indices": most_frequent.indices.tolist(),
                    "counts": most_frequent.values.tolist(),
                },
                "highest_avg_activation": {
                    "indices": highest_avg.indices.tolist(),
                    "values": highest_avg.values.tolist(),
                },
                "highest_total_activation": {
                    "indices": highest_total.indices.tolist(),
                    "values": highest_total.values.tolist(),
                },
                "feature_statistics": {
                    "mean_frequency": feature_counts.mean().item(),
                    "std_frequency": feature_counts.std().item(),
                    "mean_avg_activation": avg_acts.mean().item(),
                    "std_avg_activation": avg_acts.std().item(),
                },
            }
            
        logger.info(f"✅ Feature analysis complete")
        
        return metrics
        
    def full_analysis(self, activations: torch.Tensor) -> AnalysisResults:
        """Run complete analysis suite"""
        logger.info(f"🚀 Running full analysis on {activations.shape[0]} samples...")
        
        sparsity_metrics = self.analyze_sparsity(activations)
        reconstruction_metrics = self.analyze_reconstruction(activations)
        feature_metrics = self.analyze_features(activations)
        
        results = AnalysisResults(
            model_type=self.model_type,
            model_path=str(self.model_path),
            sparsity_metrics=sparsity_metrics,
            reconstruction_metrics=reconstruction_metrics,
            feature_metrics=feature_metrics,
        )
        
        logger.info("✅ Full analysis complete!")
        return results
        
    def plot_sparsity_distribution(self, activations: torch.Tensor, save_path: Optional[str] = None):
        """Plot sparsity distribution"""
        with torch.no_grad():
            encoded = self.model.encode(activations)
            
            # Feature activation frequencies
            feature_counts = torch.zeros(self.model.num_latents)
            feature_counts.index_add_(0, encoded.top_indices.flatten(), 
                                    torch.ones_like(encoded.top_indices.flatten(), dtype=torch.float))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram of feature activation frequencies
            ax1.hist(feature_counts.numpy(), bins=50, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Activation Frequency')
            ax1.set_ylabel('Number of Features')
            ax1.set_title(f'{self.model_type.upper()} Feature Activation Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Top active features
            top_features = torch.topk(feature_counts, min(50, self.model.num_latents))
            ax2.bar(range(len(top_features.values)), top_features.values.numpy())
            ax2.set_xlabel('Feature Rank')
            ax2.set_ylabel('Activation Count')
            ax2.set_title('Top 50 Most Active Features')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Saved sparsity plot to {save_path}")
            
            plt.show()
            
    def plot_reconstruction_quality(self, activations: torch.Tensor, save_path: Optional[str] = None):
        """Plot reconstruction quality"""
        with torch.no_grad():
            reconstruction = self.model(activations).sae_out
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot of original vs reconstructed
            orig_flat = activations.flatten().numpy()
            recon_flat = reconstruction.flatten().numpy()
            
            # Sample points for visualization if too many
            if len(orig_flat) > 10000:
                indices = np.random.choice(len(orig_flat), 10000, replace=False)
                orig_sample = orig_flat[indices]
                recon_sample = recon_flat[indices]
            else:
                orig_sample = orig_flat
                recon_sample = recon_flat
            
            ax1.scatter(orig_sample, recon_sample, alpha=0.5, s=1)
            ax1.plot([orig_sample.min(), orig_sample.max()], 
                    [orig_sample.min(), orig_sample.max()], 'r--', lw=2)
            ax1.set_xlabel('Original Activations')
            ax1.set_ylabel('Reconstructed Activations') 
            ax1.set_title(f'{self.model_type.upper()} Reconstruction Quality')
            ax1.grid(True, alpha=0.3)
            
            # Residual distribution
            residuals = (activations - reconstruction).flatten().numpy()
            ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Reconstruction Error')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Reconstruction Error Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Saved reconstruction plot to {save_path}")
            
            plt.show()