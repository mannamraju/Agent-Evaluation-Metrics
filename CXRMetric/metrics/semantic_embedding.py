"""
Semantic embedding evaluation using CheXbert encodings.

This module implements semantic similarity evaluation using CheXbert model
embeddings to measure semantic similarity between radiology reports.
"""

import pandas as pd
import numpy as np
import os
import torch
from typing import List, Dict, Any, Optional

from .base_evaluator import BaseEvaluator


class SemanticEmbeddingEvaluator(BaseEvaluator):
    """Evaluator for semantic embedding similarity using CheXbert.
    
    This evaluator uses CheXbert model to generate embeddings for radiology reports
    and computes cosine similarity between reference and generated report embeddings.
    """
    
    def __init__(self, 
                 chexbert_path: str,
                 cache_dir: str = "cache",
                 **kwargs):
        """Initialize semantic embedding evaluator.
        
        Args:
            chexbert_path: Path to CheXbert model checkpoint
            cache_dir: Directory for caching embeddings
            **kwargs: Additional arguments passed to BaseEvaluator
        """
        super().__init__(**kwargs)
        
        self.chexbert_path = chexbert_path
        self.cache_dir = cache_dir
        
        # Embedding file paths
        self.pred_embed_path = os.path.join(cache_dir, "pred_embeddings.pt")
        self.gt_embed_path = os.path.join(cache_dir, "gt_embeddings.pt")
    
    def _run_embedding_generation(self, csv_path: str, embed_path: str) -> None:
        """Run CheXbert encode.py to generate embeddings.
        
        Args:
            csv_path: Path to CSV file with reports
            embed_path: Output path for embeddings
        """
        import sys
        import subprocess
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Use the same Python executable that's running this script
        python_exe = sys.executable
        cmd = [python_exe, "CXRMetric/CheXbert/src/encode.py", "-c", self.chexbert_path, "-d", csv_path, "-o", embed_path]
        
        print(f"Running CheXbert encoding: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("CheXbert encoding completed successfully")
            if result.stdout:
                print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"CheXbert encoding failed with exit code {e.returncode}")
            print(f"Error output: {e.stderr}")
            raise
    
    def _compute_cosine_similarity(self, pred_embed_path: str, gt_embed_path: str) -> List[float]:
        """Compute cosine similarity between prediction and ground truth embeddings.
        
        Args:
            pred_embed_path: Path to prediction embeddings
            gt_embed_path: Path to ground truth embeddings
            
        Returns:
            List of cosine similarity scores
        """
        if not os.path.exists(pred_embed_path) or not os.path.exists(gt_embed_path):
            raise FileNotFoundError("Embedding files not found. Run embedding generation first.")
        
        # Load embeddings
        label_embeds = torch.load(gt_embed_path)
        pred_embeds = torch.load(pred_embed_path)
        
        # Convert to lists in sorted order
        list_label_embeds = []
        list_pred_embeds = []
        
        for data_idx in sorted(label_embeds.keys()):
            list_label_embeds.append(label_embeds[data_idx])
            list_pred_embeds.append(pred_embeds[data_idx])
        
        # Convert to numpy arrays
        np_label_embeds = torch.stack(list_label_embeds, dim=0).numpy()
        np_pred_embeds = torch.stack(list_pred_embeds, dim=0).numpy()
        
        # Compute cosine similarities
        scores = []
        for label_embed, pred_embed in zip(np_label_embeds, np_pred_embeds):
            # Cosine similarity: dot product / (norm1 * norm2)
            similarity = (label_embed * pred_embed).sum() / (
                np.linalg.norm(label_embed) * np.linalg.norm(pred_embed)
            )
            scores.append(float(similarity))
        
        return scores
    
    def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Compute semantic embedding similarity and add as column to pred_df.
        
        Args:
            gt_df: Ground truth dataframe with study_id and report columns
            pred_df: Predictions dataframe with study_id and report columns
            
        Returns:
            Updated pred_df with semb_score column added
        """
        # Align dataframes
        gt_aligned, pred_aligned = self.align_dataframes(gt_df, pred_df)
        
        # Create cache CSV files
        cache_gt_csv = os.path.join(
            os.path.dirname(self.cache_dir), f"cache_gt_{os.path.basename(self.cache_dir)}.csv"
        )
        cache_pred_csv = os.path.join(
            os.path.dirname(self.cache_dir), f"cache_pred_{os.path.basename(self.cache_dir)}.csv"
        )
        
        # Save aligned dataframes for embedding generation
        gt_aligned.to_csv(cache_gt_csv, index=False)
        pred_aligned.to_csv(cache_pred_csv, index=False)
        
        # Generate embeddings if they don't exist
        try:
            # If model path is not set or model files are missing, warn and
            # produce zero scores instead of raising an error.
            if self.chexbert_path is None or not os.path.exists(self.chexbert_path):
                raise FileNotFoundError(f"CheXbert model not found at {self.chexbert_path}")
            
            if not os.path.exists(self.pred_embed_path):
                print("Generating prediction embeddings...")
                self._run_embedding_generation(cache_pred_csv, self.pred_embed_path)
            
            if not os.path.exists(self.gt_embed_path):
                print("Generating ground truth embeddings...")
                self._run_embedding_generation(cache_gt_csv, self.gt_embed_path)
            
            # Compute similarities
            similarities = self._compute_cosine_similarity(self.pred_embed_path, self.gt_embed_path)
            pred_aligned["semb_score"] = similarities
            print(f"Computed semantic embedding scores (mean: {np.mean(similarities):.4f})")
            
        except FileNotFoundError as e:
            print(f"Warning: CheXbert model not available: {e}")
            print("To use semantic embedding evaluation:")
            print("1. Download CheXbert model checkpoint")
            print("2. Update config.py with the correct path")
            print("Setting semb_score to 0.0 for now")
            pred_aligned["semb_score"] = [0.0] * len(pred_aligned)
            
        except Exception as e:
            print(f"Warning: Semantic embedding computation failed: {e}")
            print("This may be due to missing dependencies or model issues")
            print("Setting semb_score to 0.0")
            pred_aligned["semb_score"] = [0.0] * len(pred_aligned)
        
        return pred_aligned
    
    def get_metric_columns(self) -> List[str]:
        """Get the list of column names this metric adds.
        
        Returns:
            List containing 'semb_score'
        """
        return ["semb_score"]
    
    def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for semantic embedding metric.
        
        Args:
            pred_df: Dataframe containing computed semantic similarity scores
            
        Returns:
            Dictionary with semantic embedding summary statistics
        """
        summary = super().get_summary_stats(pred_df)
        
        # Add semantic embedding-specific analysis
        semb_info = {
            'description': 'Semantic embedding similarity using CheXbert model',
            'model_path': self.chexbert_path,
            'range': '[-1, 1] where 1 is identical semantic content',
            'similarity_measure': 'cosine_similarity',
            'interpretation': {
                'advantages': 'Captures domain-specific semantic similarity for radiology',
                'use_case': 'Good for evaluating clinical content accuracy',
                'model_note': 'CheXbert trained specifically on chest X-ray reports'
            }
        }
        
        if 'semb_score' in summary and summary['semb_score']:
            semb_scores = pred_df['semb_score'].dropna()
            if len(semb_scores) > 0:
                # Analyze score distribution
                zero_scores = (semb_scores == 0.0).sum()
                negative_scores = (semb_scores < 0).sum()
                high_scores = (semb_scores > 0.8).sum()
                
                semb_info['score_distribution'] = {
                    'zero_scores_count': int(zero_scores),
                    'negative_scores_count': int(negative_scores),
                    'high_similarity_pct': f"{100 * high_scores / len(semb_scores):.1f}% (> 0.8)",
                    'note': 'Zero scores may indicate embedding generation failures'
                }
                
                # Check for potential issues
                if zero_scores > len(semb_scores) * 0.1:
                    semb_info['warnings'] = [
                        f"High number of zero scores ({zero_scores}/{len(semb_scores)})",
                        "This may indicate issues with embedding generation"
                    ]
        
        summary['semantic_embedding_analysis'] = semb_info
        return summary
    
    @property
    def name(self) -> str:
        """Get descriptive name for this evaluator."""
        return f"SemanticEmbeddingEvaluator(model=CheXbert)"
