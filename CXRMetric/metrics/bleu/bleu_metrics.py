"""
BLEU score evaluation for CXR report generation.

This module implements BLEU-2 and BLEU-4 evaluation metrics commonly used 
for assessing text generation quality in natural language processing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

try:
    # Use NLTK as the canonical BLEU implementation. fast-bleu support has
    # been removed to simplify deployment and maintenance.
    from nltk.translate.bleu_score import sentence_bleu
    import nltk
    NLTK_BLEU_AVAILABLE = True
except Exception:
    NLTK_BLEU_AVAILABLE = False

from ..base_evaluator import BaseEvaluator


class BLEUEvaluator(BaseEvaluator):
    """Evaluator for BLEU scores (BLEU-2 and BLEU-4).
    
    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between
    generated and reference texts, with geometric mean of precision scores.
    """
    
    def __init__(self, compute_bleu2: bool = True, compute_bleu4: bool = True, **kwargs):
        """Initialize BLEU evaluator.
        
        Args:
            compute_bleu2: Whether to compute BLEU-2 scores
            compute_bleu4: Whether to compute BLEU-4 scores
            **kwargs: Additional arguments passed to BaseEvaluator
        """
        super().__init__(**kwargs)
        self.compute_bleu2 = compute_bleu2
        self.compute_bleu4 = compute_bleu4
        
        # Define n-gram weights
        self.bleu2_weights = {"bigram": (1/2., 1/2.)}
        self.bleu4_weights = {"bleu4": (1/4., 1/4., 1/4., 1/4.)}
        # Backwards-compatible runtime flags
        self._fast_bleu_available = False
        self._nltk_available = NLTK_BLEU_AVAILABLE
    
    def _compute_bleu_score(self, reference_tokens: List[str], candidate_tokens: List[str], n_grams: int) -> float:
        """Compute BLEU score using available implementation.
        
        Args:
            reference_tokens: Ground truth tokens
            candidate_tokens: Predicted tokens
            n_grams: Maximum n-gram order (2 for BLEU-2, 4 for BLEU-4)
            
        Returns:
            BLEU score
        """
        if NLTK_BLEU_AVAILABLE:
            # Use NLTK implementation
            if n_grams == 2:
                weights = (0.5, 0.5, 0.0, 0.0)
            else:  # n_grams == 4
                weights = (0.25, 0.25, 0.25, 0.25)

            try:
                score = sentence_bleu([reference_tokens], candidate_tokens, weights=weights)
                return score
            except ZeroDivisionError:
                return 0.0
        
        else:
            # Fallback: simple n-gram precision
            return self._simple_bleu_fallback(reference_tokens, candidate_tokens, n_grams)
    
    def _simple_bleu_fallback(self, reference: List[str], candidate: List[str], max_n: int) -> float:
        """Simple BLEU implementation fallback when libraries are not available.
        
        Args:
            reference: Reference tokens
            candidate: Candidate tokens  
            max_n: Maximum n-gram order
            
        Returns:
            Approximate BLEU score
        """
        if not candidate or not reference:
            return 0.0
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = self._get_ngrams(reference, n)
            cand_ngrams = self._get_ngrams(candidate, n)
            
            if not cand_ngrams:
                precisions.append(0.0)
                continue
            
            # Count matches
            matches = 0
            for ngram in cand_ngrams:
                if ngram in ref_ngrams:
                    matches += min(cand_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(cand_ngrams.values())
            precisions.append(precision)
        
        # Geometric mean of precisions (simplified BLEU)
        if all(p > 0 for p in precisions):
            bleu = 1.0
            for p in precisions:
                bleu *= p
            return bleu ** (1.0 / len(precisions))
        else:
            return 0.0
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[tuple, int]:
        """Get n-gram counts from token list.
        
        Args:
            tokens: List of tokens
            n: N-gram order
            
        Returns:
            Dictionary mapping n-grams to counts
        """
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Compute BLEU scores and add as columns to pred_df.
        
        Args:
            gt_df: Ground truth dataframe with study_id and report columns
            pred_df: Predictions dataframe with study_id and report columns
            
        Returns:
            Updated pred_df with BLEU score columns added
        """
        # Implementation selection is driven at import time; if NLTK is not
        # available this evaluator will gracefully fall back to the simple
        # in-repo n-gram precision implementation without printing a warning.
        # (This keeps demos quiet and avoids noisy output in CI.)
        
        # Align dataframes
        gt_aligned, pred_aligned = self.align_dataframes(gt_df, pred_df)
        
        # Initialize score columns
        if self.compute_bleu2:
            pred_aligned["bleu_score"] = [0.0] * len(pred_aligned)
        if self.compute_bleu4:
            pred_aligned["bleu4_score"] = [0.0] * len(pred_aligned)
        
        # Compute scores for each report pair
        for i, row in gt_aligned.iterrows():
            gt_report = self.prep_reports([row[self.report_col]])[0]
            pred_row = pred_aligned[pred_aligned[self.study_id_col] == row[self.study_id_col]]
            
            if len(pred_row) == 1:
                predicted_report = self.prep_reports([pred_row[self.report_col].values[0]])[0]
                _index = pred_aligned.index[
                    pred_aligned[self.study_id_col] == row[self.study_id_col]
                ].tolist()[0]
                
                # Compute BLEU-2
                if self.compute_bleu2:
                    try:
                        score2 = self._compute_bleu_score(gt_report, predicted_report, 2)
                        pred_aligned.at[_index, "bleu_score"] = score2
                    except Exception as e:
                        pred_aligned.at[_index, "bleu_score"] = 0.0
                
                # Compute BLEU-4
                if self.compute_bleu4:
                    try:
                        score4 = self._compute_bleu_score(gt_report, predicted_report, 4)
                        pred_aligned.at[_index, "bleu4_score"] = score4
                    except Exception as e:
                        pred_aligned.at[_index, "bleu4_score"] = 0.0
        
        return pred_aligned
    
    def get_metric_columns(self) -> List[str]:
        """Get the list of column names this metric adds.
        
        Returns:
            List of column names for BLEU metrics
        """
        columns = []
        if self.compute_bleu2:
            columns.append("bleu_score")
        if self.compute_bleu4:
            columns.append("bleu4_score")
        return columns
    
    def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for BLEU metrics.
        
        Args:
            pred_df: Dataframe containing computed BLEU scores
            
        Returns:
            Dictionary with BLEU summary statistics
        """
        summary = super().get_summary_stats(pred_df)
        
        # Add BLEU-specific analysis
        bleu_info = {
            'description': 'BLEU scores measure n-gram overlap between generated and reference texts',
            'range': '[0, 1] where 1 is perfect match',
            'interpretation': {
                'bleu_score': 'BLEU-2: considers unigrams and bigrams',
                'bleu4_score': 'BLEU-4: considers 1-4 grams (standard evaluation)'
            }
        }
        
        if 'bleu_score' in summary and 'bleu4_score' in summary:
            # Compare BLEU-2 vs BLEU-4 performance
            bleu2_mean = summary['bleu_score']['mean']
            bleu4_mean = summary['bleu4_score']['mean']
            bleu_info['comparison'] = {
                'bleu2_vs_bleu4_diff': bleu2_mean - bleu4_mean,
                'note': 'BLEU-2 typically higher than BLEU-4 (fewer constraints)'
            }
        
        summary['bleu_analysis'] = bleu_info
        return summary
    
    @property 
    def name(self) -> str:
        """Get descriptive name for this evaluator."""
        metrics = []
        if self.compute_bleu2:
            metrics.append('BLEU-2')
        if self.compute_bleu4:
            metrics.append('BLEU-4')
        return f"BLEUEvaluator({', '.join(metrics)})"
