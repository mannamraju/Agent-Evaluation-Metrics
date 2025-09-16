"""
BERTScore evaluation for CXR report generation.

This module implements BERTScore metric which uses contextual embeddings
from BERT models to measure semantic similarity between texts.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional

from ..base_evaluator import BaseEvaluator

try:
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    # Create a dummy class for type hints when BERTScorer is not available
    class BERTScorer:
        pass


class BERTScoreEvaluator(BaseEvaluator):
    """Evaluator for BERTScore metric.
    
    BERTScore leverages pre-trained BERT embeddings to compute similarity scores
    between reference and generated texts at the token level, providing a more
    nuanced semantic similarity measure than traditional n-gram based metrics.
    """
    
    def __init__(self, 
                 model_type: str = "distilroberta-base",
                 batch_size: int = 256,
                 use_idf: bool = False,
                 rescale_with_baseline: bool = True,
                 lang: str = "en",
                 **kwargs):
        """Initialize BERTScore evaluator.
        
        Args:
            model_type: Pre-trained model to use for embeddings
            batch_size: Batch size for processing
            use_idf: Whether to use inverse document frequency weighting
            rescale_with_baseline: Whether to rescale scores with baseline
            lang: Language code for the texts
            **kwargs: Additional arguments passed to BaseEvaluator
        """
        super().__init__(**kwargs)
        
        if not BERTSCORE_AVAILABLE:
            raise ImportError("bert_score package is required for BERTScoreEvaluator")
        
        self.model_type = model_type
        self.batch_size = batch_size
        self.use_idf = use_idf
        self.rescale_with_baseline = rescale_with_baseline
        self.lang = lang
        
        self._scorer = None
    
    def _get_scorer(self, reference_texts: List[str]) -> BERTScorer:
        """Get or create BERTScorer instance.
        
        Args:
            reference_texts: List of reference texts for IDF computation
            
        Returns:
            Configured BERTScorer instance
        """
        if self._scorer is None:
            idf_sents = reference_texts if self.use_idf else None
            
            self._scorer = BERTScorer(
                model_type=self.model_type,
                batch_size=self.batch_size,
                lang=self.lang,
                rescale_with_baseline=self.rescale_with_baseline,
                idf=self.use_idf,
                idf_sents=idf_sents
            )
        
        return self._scorer
    
    def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Compute BERTScore and add as column to pred_df.
        
        Args:
            gt_df: Ground truth dataframe with study_id and report columns
            pred_df: Predictions dataframe with study_id and report columns
            
        Returns:
            Updated pred_df with bertscore column added
        """
        # Align dataframes
        gt_aligned, pred_aligned = self.align_dataframes(gt_df, pred_df)
        
        # Prepare texts
        reference_texts = gt_aligned[self.report_col].tolist()
        candidate_texts = pred_aligned[self.report_col].tolist()
        
        # Clean texts (remove extra spaces)
        reference_texts = [re.sub(r' +', ' ', text) for text in reference_texts]
        candidate_texts = [re.sub(r' +', ' ', text) for text in candidate_texts]
        
        # Get scorer and compute scores
        scorer = self._get_scorer(reference_texts)
        precision, recall, f1 = scorer.score(candidate_texts, reference_texts)
        
        # Add F1 scores to dataframe (most commonly used BERTScore variant)
        pred_aligned["bertscore"] = f1
        
        # Optionally add precision and recall
        pred_aligned["bertscore_precision"] = precision
        pred_aligned["bertscore_recall"] = recall
        
        return pred_aligned
    
    def get_metric_columns(self) -> List[str]:
        """Get the list of column names this metric adds.
        
        Returns:
            List of BERTScore column names
        """
        return ["bertscore", "bertscore_precision", "bertscore_recall"]
    
    def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for BERTScore metric.
        
        Args:
            pred_df: Dataframe containing computed BERTScore values
            
        Returns:
            Dictionary with BERTScore summary statistics
        """
        summary = super().get_summary_stats(pred_df)
        
        # Add BERTScore-specific analysis
        bertscore_info = {
            'description': 'BERTScore uses contextual embeddings for semantic similarity',
            'model_type': self.model_type,
            'use_idf': self.use_idf,
            'rescale_with_baseline': self.rescale_with_baseline,
            'range': 'Typically [-1, 1] but varies by model and baseline',
            'interpretation': {
                'advantages': 'Captures semantic similarity beyond surface-level matches',
                'f1_score': 'Harmonic mean of precision and recall (most commonly reported)',
                'use_case': 'Good for evaluating semantic content preservation'
            }
        }
        
        if 'bertscore' in summary:
            bert_scores = pred_df['bertscore'].dropna()
            if len(bert_scores) > 0:
                # Analyze score characteristics
                negative_scores = (bert_scores < 0).sum()
                high_scores = (bert_scores > 0.8).sum()
                
                bertscore_info['score_characteristics'] = {
                    'negative_scores_count': int(negative_scores),
                    'high_scores_pct': f"{100 * high_scores / len(bert_scores):.1f}% (> 0.8)",
                    'baseline_note': 'Scores rescaled with baseline if enabled'
                }
                
                # Compare precision vs recall if available
                if 'bertscore_precision' in pred_df.columns and 'bertscore_recall' in pred_df.columns:
                    precision_mean = pred_df['bertscore_precision'].mean()
                    recall_mean = pred_df['bertscore_recall'].mean()
                    
                    bertscore_info['precision_vs_recall'] = {
                        'precision_mean': float(precision_mean),
                        'recall_mean': float(recall_mean),
                        'difference': float(precision_mean - recall_mean),
                        'note': 'Precision > Recall suggests conservative generation'
                    }
        
        summary['bertscore_analysis'] = bertscore_info
        return summary
    
    @property
    def name(self) -> str:
        """Get descriptive name for this evaluator."""
        return f"BERTScoreEvaluator(model={self.model_type}, idf={self.use_idf})"
