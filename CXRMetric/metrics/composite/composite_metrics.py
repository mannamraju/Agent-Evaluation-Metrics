"""
Composite metric evaluation for CXR report generation.

This module implements RadCliQ composite metrics (v0 and v1) that combine
multiple evaluation metrics into single quality scores using trained models.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional

from ..base_evaluator import BaseEvaluator

try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CompositeMetricEvaluator(BaseEvaluator):
    """Evaluator for RadCliQ composite metrics (v0 and v1).
    
    Composite metrics combine multiple individual evaluation metrics
    (BLEU, BERTScore, semantic embedding, RadGraph) into single quality
    scores using trained regression models.
    """
    
    def __init__(self, 
                 composite_v0_path: str = "CXRMetric/composite_metric_model.pkl",
                 composite_v1_path: str = "CXRMetric/radcliq-v1.pkl",
                 normalizer_path: str = "CXRMetric/normalizer.pkl",
                 input_columns: List[str] = None,
                 compute_v0: bool = True,
                 compute_v1: bool = True,
                 **kwargs):
        """Initialize composite metric evaluator.
        
        Args:
            composite_v0_path: Path to RadCliQ-v0 model
            composite_v1_path: Path to RadCliQ-v1 model  
            normalizer_path: Path to normalizer for v0 model
            input_columns: List of column names to use as input features
            compute_v0: Whether to compute RadCliQ-v0
            compute_v1: Whether to compute RadCliQ-v1
            **kwargs: Additional arguments passed to BaseEvaluator
        """
        super().__init__(**kwargs)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for CompositeMetricEvaluator")
        
        self.composite_v0_path = composite_v0_path
        self.composite_v1_path = composite_v1_path
        self.normalizer_path = normalizer_path
        self.compute_v0 = compute_v0
        self.compute_v1 = compute_v1
        
        # Default input columns (standard metrics used by RadCliQ)
        if input_columns is None:
            self.input_columns = ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]
        else:
            self.input_columns = input_columns
        
        # Model storage
        self._v0_model = None
        self._v1_model = None
        self._normalizer = None
    
    def _load_models(self) -> None:
        """Load composite metric models and normalizer."""
        try:
            # Load RadCliQ-v0 model and normalizer
            if self.compute_v0:
                if not os.path.exists(self.composite_v0_path):
                    raise FileNotFoundError(f"RadCliQ-v0 model not found: {self.composite_v0_path}")
                if not os.path.exists(self.normalizer_path):
                    raise FileNotFoundError(f"Normalizer not found: {self.normalizer_path}")
                
                with open(self.composite_v0_path, "rb") as f:
                    self._v0_model = pickle.load(f)
                with open(self.normalizer_path, "rb") as f:
                    self._normalizer = pickle.load(f)
                print("Loaded RadCliQ-v0 model and normalizer")
            
            # Load RadCliQ-v1 model
            if self.compute_v1:
                if not os.path.exists(self.composite_v1_path):
                    raise FileNotFoundError(f"RadCliQ-v1 model not found: {self.composite_v1_path}")
                
                with open(self.composite_v1_path, "rb") as f:
                    self._v1_model = pickle.load(f)
                print("Loaded RadCliQ-v1 model")
        
        except Exception as e:
            print(f"Warning: Failed to load composite models: {e}")
            self.compute_v0 = False
            self.compute_v1 = False
    
    def _validate_input_data(self, pred_df: pd.DataFrame) -> np.ndarray:
        """Validate and extract input data for composite models.
        
        Args:
            pred_df: Dataframe with individual metric columns
            
        Returns:
            Input array for composite models
        """
        missing_columns = [col for col in self.input_columns if col not in pred_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for composite metrics: {missing_columns}")
        
        # Extract input data
        input_data = pred_df[self.input_columns].values
        
        # Check for missing values
        if np.any(np.isnan(input_data)):
            print("Warning: Found NaN values in input data. Filling with 0.")
            input_data = np.nan_to_num(input_data, nan=0.0)
        
        return input_data
    
    def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite metrics and add as columns to pred_df.
        
        Note: This assumes individual metrics have already been computed.
        
        Args:
            gt_df: Ground truth dataframe (not used directly)
            pred_df: Predictions dataframe with individual metric columns
            
        Returns:
            Updated pred_df with composite metric columns added
        """
        # Load models if not already loaded
        if self._v0_model is None and self._v1_model is None:
            self._load_models()
        
        # Initialize columns
        if self.compute_v0:
            pred_df["RadCliQ-v0"] = [0.0] * len(pred_df)
        if self.compute_v1:
            pred_df["RadCliQ-v1"] = [0.0] * len(pred_df)
        
        try:
            # Validate and extract input data
            input_data = self._validate_input_data(pred_df)
            
            # Compute RadCliQ-v0
            if self.compute_v0 and self._v0_model is not None and self._normalizer is not None:
                try:
                    # Normalize input data
                    normalized_input = self._normalizer.transform(input_data)
                    
                    # Generate predictions
                    v0_scores = self._v0_model.predict(normalized_input)
                    pred_df["RadCliQ-v0"] = v0_scores
                    print(f"Computed RadCliQ-v0 scores (mean: {np.mean(v0_scores):.4f})")
                
                except Exception as e:
                    print(f"Warning: RadCliQ-v0 computation failed: {e}")
                    pred_df["RadCliQ-v0"] = [0.0] * len(pred_df)
            
            # Compute RadCliQ-v1
            if self.compute_v1 and self._v1_model is not None:
                try:
                    # Generate predictions (v1 model handles normalization internally)
                    v1_scores = self._v1_model.predict(input_data)
                    pred_df["RadCliQ-v1"] = v1_scores
                    print(f"Computed RadCliQ-v1 scores (mean: {np.mean(v1_scores):.4f})")
                
                except Exception as e:
                    print(f"Warning: RadCliQ-v1 computation failed: {e}")
                    pred_df["RadCliQ-v1"] = [0.0] * len(pred_df)
        
        except Exception as e:
            print(f"Warning: Composite metric computation failed: {e}")
            if self.compute_v0:
                pred_df["RadCliQ-v0"] = [0.0] * len(pred_df)
            if self.compute_v1:
                pred_df["RadCliQ-v1"] = [0.0] * len(pred_df)
        
        return pred_df
    
    def get_metric_columns(self) -> List[str]:
        """Get the list of column names this metric adds.
        
        Returns:
            List of composite metric column names
        """
        columns = []
        if self.compute_v0:
            columns.append("RadCliQ-v0")
        if self.compute_v1:
            columns.append("RadCliQ-v1")
        return columns
    
    def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for composite metrics.
        
        Args:
            pred_df: Dataframe containing computed composite scores
            
        Returns:
            Dictionary with composite metric summary statistics
        """
        summary = super().get_summary_stats(pred_df)
        
        # Add composite metric-specific analysis
        composite_info = {
            'description': 'RadCliQ composite metrics combine multiple evaluation scores',
            'input_features': self.input_columns,
            'models_computed': [],
            'interpretation': {
                'purpose': 'Single quality scores trained to correlate with radiologist preferences',
                'range': 'Model-dependent, typically normalized to meaningful scale',
                'use_case': 'Overall report quality assessment combining multiple aspects'
            }
        }
        
        if self.compute_v0 and "RadCliQ-v0" in pred_df.columns:
            composite_info['models_computed'].append('RadCliQ-v0')
        
        if self.compute_v1 and "RadCliQ-v1" in pred_df.columns:
            composite_info['models_computed'].append('RadCliQ-v1')
        
        # Compare versions if both available
        if ("RadCliQ-v0" in summary and "RadCliQ-v1" in summary and 
            summary["RadCliQ-v0"] and summary["RadCliQ-v1"]):
            
            v0_mean = summary["RadCliQ-v0"]['mean']
            v1_mean = summary["RadCliQ-v1"]['mean']
            
            composite_info['version_comparison'] = {
                'v0_mean': v0_mean,
                'v1_mean': v1_mean,
                'difference': v1_mean - v0_mean,
                'correlation_note': 'v1 typically shows higher correlation with radiologist preferences'
            }
        
        # Analyze score distributions
        for version in ['RadCliQ-v0', 'RadCliQ-v1']:
            if version in pred_df.columns:
                scores = pred_df[version].dropna()
                if len(scores) > 0:
                    # Check for potential model issues
                    zero_scores = (scores == 0.0).sum()
                    if zero_scores > len(scores) * 0.1:
                        if 'warnings' not in composite_info:
                            composite_info['warnings'] = []
                        composite_info['warnings'].append(
                            f"High number of zero scores in {version} ({zero_scores}/{len(scores)})"
                        )
        
        summary['composite_analysis'] = composite_info
        return summary
    
    @property
    def name(self) -> str:
        """Get descriptive name for this evaluator."""
        versions = []
        if self.compute_v0:
            versions.append('v0')
        if self.compute_v1:
            versions.append('v1')
        return f"CompositeMetricEvaluator(RadCliQ-{'+'.join(versions)})"
