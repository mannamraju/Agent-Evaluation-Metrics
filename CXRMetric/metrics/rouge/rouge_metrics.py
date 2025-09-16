"""
ROUGE score evaluation for CXR report generation.

This module implements ROUGE-L evaluation metric which measures the longest
common subsequence overlap between generated and reference texts.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any

from ..base_evaluator import BaseEvaluator


class ROUGEEvaluator(BaseEvaluator):
    """Evaluator for ROUGE-L scores.
    
    ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)
    measures the longest common subsequence between generated and reference texts,
    providing a measure of content preservation that doesn't require consecutive matches.
    """
    
    def __init__(self, beta: float = 1.2, **kwargs):
        """Initialize ROUGE-L evaluator.
        
        Args:
            beta: Beta parameter for F1 score computation (weights recall vs precision)
            **kwargs: Additional arguments passed to BaseEvaluator
        """
        super().__init__(**kwargs)
        self.beta = beta
    
    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        """Compute length of longest common subsequence between token lists.
        
        Uses dynamic programming to efficiently find LCS length.
        
        Args:
            a: First token sequence
            b: Second token sequence
            
        Returns:
            Length of longest common subsequence
        """
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
            
        # DP table for LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table bottom-up
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if a[i] == b[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        
        return dp[0][0]
    
    def rouge_l_score(self, reference: str, candidate: str) -> float:
        """Compute ROUGE-L F1 score between reference and candidate strings.
        
        Args:
            reference: Ground truth text
            candidate: Generated text
            
        Returns:
            ROUGE-L F1 score in range [0, 1]
        """
        ref_tokens = str(reference).split()
        cand_tokens = str(candidate).split()
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return 0.0
        
        lcs_len = self._lcs_length(ref_tokens, cand_tokens)
        
        if lcs_len == 0:
            return 0.0
        
        # Compute precision and recall
        precision = lcs_len / len(cand_tokens)
        recall = lcs_len / len(ref_tokens)
        
        # Compute F1 with beta weighting
        beta_sq = self.beta * self.beta
        denominator = recall + beta_sq * precision
        
        if denominator == 0:
            return 0.0
        
        f1_score = ((1 + beta_sq) * precision * recall) / denominator
        return float(f1_score)
    
    def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Compute ROUGE-L scores and add as column to pred_df.
        
        Args:
            gt_df: Ground truth dataframe with study_id and report columns
            pred_df: Predictions dataframe with study_id and report columns
            
        Returns:
            Updated pred_df with rouge_l column added
        """
        # Align dataframes
        gt_aligned, pred_aligned = self.align_dataframes(gt_df, pred_df)
        
        # Initialize score column
        pred_aligned["rouge_l"] = [0.0] * len(pred_aligned)
        
        # Compute ROUGE-L for each report pair
        for i, row in gt_aligned.iterrows():
            gt_report = re.sub(r' +', ' ', str(row[self.report_col]).strip())
            pred_row = pred_aligned[pred_aligned[self.study_id_col] == row[self.study_id_col]]
            
            if len(pred_row) == 1:
                predicted_report = re.sub(r' +', ' ', str(pred_row[self.report_col].values[0]).strip())
                score = self.rouge_l_score(gt_report, predicted_report)
                
                _index = pred_aligned.index[
                    pred_aligned[self.study_id_col] == row[self.study_id_col]
                ].tolist()[0]
                
                pred_aligned.at[_index, "rouge_l"] = score
        
        return pred_aligned
    
    def get_metric_columns(self) -> List[str]:
        """Get the list of column names this metric adds.
        
        Returns:
            List containing 'rouge_l'
        """
        return ["rouge_l"]
    
    def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for ROUGE-L metric.
        
        Args:
            pred_df: Dataframe containing computed ROUGE-L scores
            
        Returns:
            Dictionary with ROUGE-L summary statistics
        """
        summary = super().get_summary_stats(pred_df)
        
        # Add ROUGE-specific analysis
        rouge_info = {
            'description': 'ROUGE-L measures longest common subsequence overlap',
            'range': '[0, 1] where 1 is perfect sequence match',
            'beta_parameter': self.beta,
            'interpretation': {
                'advantages': 'Captures content similarity without requiring exact phrase matches',
                'use_case': 'Good for evaluating content preservation and fluency'
            }
        }
        
        if 'rouge_l' in summary:
            rouge_scores = pred_df['rouge_l'].dropna()
            if len(rouge_scores) > 0:
                # Analyze score distribution
                low_scores = (rouge_scores < 0.2).sum()
                high_scores = (rouge_scores > 0.6).sum()
                
                rouge_info['score_distribution'] = {
                    'low_scores_pct': f"{100 * low_scores / len(rouge_scores):.1f}% (< 0.2)",
                    'high_scores_pct': f"{100 * high_scores / len(rouge_scores):.1f}% (> 0.6)",
                    'note': 'ROUGE-L > 0.4 generally indicates good content overlap'
                }
        
        summary['rouge_analysis'] = rouge_info
        return summary
    
    @property
    def name(self) -> str:
        """Get descriptive name for this evaluator."""
        return f"ROUGEEvaluator(beta={self.beta})"
