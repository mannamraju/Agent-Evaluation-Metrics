"""
Base evaluator class for CXR report evaluation metrics.

This module provides the abstract base class that all metric evaluators should inherit from,
ensuring consistent interface and behavior across different evaluation metrics.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation metrics.
    
    All metric evaluators should inherit from this class and implement the required methods.
    This ensures consistent interface and behavior across different metrics.
    """
    
    def __init__(self, study_id_col: str = "study_id", report_col: str = "report"):
        """Initialize the base evaluator.
        
        Args:
            study_id_col: Name of the study ID column in dataframes
            report_col: Name of the report text column in dataframes
        """
        self.study_id_col = study_id_col
        self.report_col = report_col
    
    @abstractmethod
    def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the metric and add results as columns to pred_df.
        
        Args:
            gt_df: Ground truth dataframe with study_id and report columns
            pred_df: Predictions dataframe with study_id and report columns
            
        Returns:
            Updated pred_df with new metric columns added
        """
        pass
    
    @abstractmethod
    def get_metric_columns(self) -> List[str]:
        """Get the list of column names this metric adds to the dataframe.
        
        Returns:
            List of column names that will be added by compute_metric()
        """
        pass
    
    def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for this metric.
        
        Args:
            pred_df: Dataframe containing computed metrics
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        for col in self.get_metric_columns():
            if col in pred_df.columns:
                values = pred_df[col].dropna()
                if len(values) > 0:
                    summary[col] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values))
                    }
        return summary
    
    def prep_reports(self, reports: List[str]) -> List[List[str]]:
        """Preprocess reports for evaluation (tokenization, cleaning, etc.).
        
        Args:
            reports: List of report strings
            
        Returns:
            List of tokenized reports (each report as list of tokens)
        """
        return [list(filter(
            lambda val: val != "", str(elem)
            .lower().replace(".", " .").split(" "))) for elem in reports]
    
    def align_dataframes(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align ground truth and prediction dataframes by study ID.
        
        Args:
            gt_df: Ground truth dataframe
            pred_df: Predictions dataframe
            
        Returns:
            Tuple of aligned (gt_df, pred_df) with same study IDs and order
        """
        # Sort by study ID
        gt_df = gt_df.sort_values(by=[self.study_id_col]).reset_index(drop=True)
        pred_df = pred_df.sort_values(by=[self.study_id_col]).reset_index(drop=True)
        
        # Find intersection of study IDs
        gt_study_ids = set(gt_df[self.study_id_col])
        pred_study_ids = set(pred_df[self.study_id_col])
        shared_study_ids = gt_study_ids.intersection(pred_study_ids)
        
        # Filter to shared IDs
        gt_aligned = gt_df.loc[gt_df[self.study_id_col].isin(shared_study_ids)].reset_index(drop=True)
        pred_aligned = pred_df.loc[pred_df[self.study_id_col].isin(shared_study_ids)].reset_index(drop=True)
        
        # Verify alignment
        assert len(gt_aligned) == len(pred_aligned), "Aligned dataframes must have same length"
        assert gt_aligned[self.study_id_col].equals(pred_aligned[self.study_id_col]), "Study IDs must match after alignment"
        
        return gt_aligned, pred_aligned
    
    @property
    def name(self) -> str:
        """Get the name of this evaluator."""
        return self.__class__.__name__
    
    def __str__(self) -> str:
        return f"{self.name}(columns={self.get_metric_columns()})"
    
    def __repr__(self) -> str:
        return self.__str__()
