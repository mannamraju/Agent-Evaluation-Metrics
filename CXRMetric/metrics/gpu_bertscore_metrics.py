"""
GPU-optimized BERTScore evaluation for CXR report generation on Azure.

This module implements BERTScore metric with GPU optimizations for Azure deployment.
"""

import pandas as pd
import numpy as np
import re
import torch
from typing import List, Dict, Any, Optional

from .gpu_optimized_base import GPUOptimizedBaseEvaluator

try:
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


class GPUOptimizedBERTScoreEvaluator(GPUOptimizedBaseEvaluator):
    """GPU-optimized evaluator for BERTScore metric on Azure.
    
    This evaluator leverages GPU acceleration and smart batching for efficient
    BERTScore computation in Azure cloud environments.
    """
    
    def __init__(self, 
                 model_type: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 use_idf: bool = False,
                 rescale_with_baseline: bool = True,
                 lang: str = "en",
                 batch_size: Optional[int] = None,
                 device: Optional[str] = None,
                 **kwargs):
        """Initialize GPU-optimized BERTScore evaluator.
        
        Args:
            model_type: Pre-trained model optimized for medical texts
            use_idf: Whether to use inverse document frequency weighting
            rescale_with_baseline: Whether to rescale scores with baseline
            lang: Language code for the texts
            batch_size: Override automatic batch size calculation
            device: Override automatic device selection
            **kwargs: Additional arguments
        """
        if not BERTSCORE_AVAILABLE:
            raise ImportError("bert_score package is required for BERTScoreEvaluator")
        
        self.model_type = model_type
        self.use_idf = use_idf
        self.rescale_with_baseline = rescale_with_baseline
        self.lang = lang
        
        super().__init__(batch_size=batch_size, device=device, **kwargs)
        
        self._scorer = None
    
    @property
    def model_type_for_batch(self) -> str:
        """Return model type for batch size optimization"""
        return "bert"
    
    def _initialize_models(self):
        """Initialize BERTScore model with GPU optimization"""
        # Model will be initialized lazily in _get_scorer
        pass
    
    def _get_scorer(self, reference_texts: List[str]) -> BERTScorer:
        """Get or create GPU-optimized BERTScorer instance.
        
        Args:
            reference_texts: List of reference texts for IDF computation
            
        Returns:
            Configured BERTScorer instance with GPU optimization
        """
        if self._scorer is None:
            idf_sents = reference_texts if self.use_idf else None
            
            # Extract device number for BERTScorer
            device_id = None
            if "cuda" in self.device:
                device_id = int(self.device.split(":")[1])
            
            self._scorer = BERTScorer(
                model_type=self.model_type,
                batch_size=self.batch_size,
                lang=self.lang,
                rescale_with_baseline=self.rescale_with_baseline,
                idf=self.use_idf,
                idf_sents=idf_sents,
                device=self.device
            )
            
            self.logger.info(f"Initialized BERTScore with {self.model_type} on {self.device}")
        
        return self._scorer
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for optimal BERTScore computation"""
        processed = []
        
        for text in texts:
            # Clean text
            text = re.sub(r' +', ' ', text)  # Remove extra spaces
            text = text.strip()
            
            # Truncate very long texts to prevent memory issues
            # BERTScore uses tokenization, but we can pre-truncate
            if len(text) > 2000:  # Conservative limit
                text = text[:2000]
                self.logger.warning("Truncated text exceeding 2000 characters")
            
            processed.append(text)
        
        return processed
    
    def compute_metric(self, gt_reports: List[str], pred_reports: List[str]) -> Dict[str, Any]:
        """Compute BERTScore with GPU optimization.
        
        Args:
            gt_reports: Ground truth reports
            pred_reports: Predicted reports
            
        Returns:
            Dictionary containing BERTScore metrics
        """
        if len(gt_reports) != len(pred_reports):
            raise ValueError("Ground truth and predicted reports must have the same length")
        
        # Preprocess texts
        reference_texts = self._preprocess_texts(gt_reports)
        candidate_texts = self._preprocess_texts(pred_reports)
        
        # Get scorer and compute scores
        scorer = self._get_scorer(reference_texts)
        
        try:
            # Move scorer to correct device if needed
            if hasattr(scorer, 'device') and scorer.device != self.device:
                scorer = scorer.to(self.device)
            
            # Compute scores with memory management
            with torch.cuda.amp.autocast() if "cuda" in self.device else torch.no_grad():
                precision, recall, f1 = scorer.score(candidate_texts, reference_texts)
            
            # Convert tensors to numpy/python types
            if torch.is_tensor(precision):
                precision = precision.cpu().numpy()
            if torch.is_tensor(recall):
                recall = recall.cpu().numpy()
            if torch.is_tensor(f1):
                f1 = f1.cpu().numpy()
            
            # Return results for each report
            results = []
            for i in range(len(gt_reports)):
                results.append({
                    "bertscore_precision": float(precision[i]) if hasattr(precision, '__len__') else float(precision),
                    "bertscore_recall": float(recall[i]) if hasattr(recall, '__len__') else float(recall),
                    "bertscore_f1": float(f1[i]) if hasattr(f1, '__len__') else float(f1)
                })
            
            return results
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Handle OOM by processing in smaller batches
                self.logger.warning(f"GPU OOM, falling back to smaller batches")
                # Recursive processing with smaller batches handled by base class
                raise e
            else:
                raise e
    
    def get_metric_columns(self) -> List[str]:
        """Get the list of column names this metric adds."""
        return ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
    
    def get_summary_stats(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for BERTScore metric.
        
        Args:
            results_df: DataFrame containing computed BERTScore values
            
        Returns:
            Dictionary with BERTScore summary statistics
        """
        summary = {}
        
        for col in self.get_metric_columns():
            if col in results_df.columns:
                values = results_df[col].dropna()
                if len(values) > 0:
                    summary[col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median())
                    }
        
        # Add BERTScore-specific analysis
        summary['bertscore_analysis'] = {
            'description': 'GPU-optimized BERTScore using contextual embeddings for semantic similarity',
            'model_type': self.model_type,
            'device': self.device,
            'batch_size': self.batch_size,
            'use_idf': self.use_idf,
            'rescale_with_baseline': self.rescale_with_baseline,
            'optimizations': [
                'Mixed precision training',
                'Adaptive batching for OOM handling',
                'Medical domain model (PubMedBERT)',
                'GPU memory management'
            ]
        }
        
        if 'bertscore_f1' in results_df.columns:
            f1_scores = results_df['bertscore_f1'].dropna()
            if len(f1_scores) > 0:
                # Analyze score characteristics
                high_scores = (f1_scores > 0.8).sum()
                low_scores = (f1_scores < 0.5).sum()
                
                summary['bertscore_analysis']['score_characteristics'] = {
                    'high_scores_count': int(high_scores),
                    'high_scores_pct': f"{100 * high_scores / len(f1_scores):.1f}%",
                    'low_scores_count': int(low_scores),
                    'low_scores_pct': f"{100 * low_scores / len(f1_scores):.1f}%",
                    'score_range': f"[{f1_scores.min():.3f}, {f1_scores.max():.3f}]"
                }
        
        return summary
    
    def warm_up(self, sample_texts: Optional[List[str]] = None):
        """Warm up BERTScore model with sample medical texts"""
        if not sample_texts:
            sample_texts = [
                "No acute cardiopulmonary abnormality.",
                "Heart size is normal. The lungs are clear.",
                "There is no evidence of pneumonia or other acute findings.",
                "Mild cardiomegaly with clear lungs.",
                "Bilateral lower lobe opacity consistent with pneumonia."
            ]
        
        self.logger.info("Warming up BERTScore model...")
        
        try:
            # Initialize scorer with warm-up texts
            scorer = self._get_scorer(sample_texts[:3])
            
            # Run a small evaluation to warm up CUDA kernels
            with torch.no_grad():
                _, _, _ = scorer.score(sample_texts[:2], sample_texts[1:3])
            
            self.logger.info("BERTScore model warm-up completed successfully")
        except Exception as e:
            self.logger.warning(f"BERTScore model warm-up failed: {e}")
    
    @property
    def name(self) -> str:
        """Get descriptive name for this evaluator."""
        return f"GPUOptimizedBERTScoreEvaluator(model={self.model_type}, device={self.device})"
