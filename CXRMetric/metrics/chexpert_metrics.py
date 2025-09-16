"""
CheXpert evaluation for CXR report generation.

This module implements CheXpert micro-F1 evaluation using CheXbert labeling
to assess clinical finding accuracy across multiple pathology conditions.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple

from .base_evaluator import BaseEvaluator

try:
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CheXpertEvaluator(BaseEvaluator):
    """Evaluator for CheXpert micro-F1 scores using CheXbert labeling.
    
    This evaluator uses the CheXbert model to label radiology reports for
    14 pathology conditions and computes micro-averaged F1 scores across
    all conditions, treating uncertain labels as positive.
    """
    
    def __init__(self, 
                 chexbert_path: str,
                 cache_dir: str = "cache",
                 treat_uncertain_as_positive: bool = True,
                 **kwargs):
        """Initialize CheXpert evaluator.
        
        Args:
            chexbert_path: Path to CheXbert model checkpoint
            cache_dir: Directory for caching labeled results
            treat_uncertain_as_positive: Whether to treat uncertain (-1) labels as positive
            **kwargs: Additional arguments passed to BaseEvaluator
        """
        super().__init__(**kwargs)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for CheXpertEvaluator")
        
        self.chexbert_path = chexbert_path
        self.cache_dir = cache_dir
        self.treat_uncertain_as_positive = treat_uncertain_as_positive
        
        # Paths for labeled CSV files
        self.pred_labels_dir = os.path.join(cache_dir, "pred_labels")
        self.gt_labels_dir = os.path.join(cache_dir, "gt_labels")
    
    def _run_chexbert_labeler(self, csv_path: str, output_dir: str) -> Optional[str]:
        """Run CheXbert labeling script on a CSV file.
        
        Args:
            csv_path: Path to input CSV with reports
            output_dir: Output directory for labeled results
            
        Returns:
            Path to labeled_reports.csv if successful, None otherwise
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = f"python CXRMetric/CheXbert/src/label.py -c {self.chexbert_path} -d {csv_path} -o {output_dir}"
        print(f"Running CheXbert labeler: {cmd}")
        
        # Run labeling command
        exit_code = os.system(cmd)
        
        # Check if output file was created
        labeled_csv = os.path.join(output_dir, "labeled_reports.csv")
        if exit_code == 0 and os.path.exists(labeled_csv):
            return labeled_csv
        else:
            print(f"Warning: CheXbert labeling may have failed (exit code: {exit_code})")
            return labeled_csv if os.path.exists(labeled_csv) else None
    
    def _parse_chex_labels(self, labeled_csv: str) -> Tuple[np.ndarray, List[str]]:
        """Parse CheXbert labeled CSV into binary label array.
        
        CheXbert labeling convention:
        - Positive: 1
        - Negative: 0
        - Uncertain: -1
        - Blank/NaN: treated as 0
        
        Args:
            labeled_csv: Path to labeled_reports.csv
            
        Returns:
            Tuple of (label_array, condition_names)
        """
        df = pd.read_csv(labeled_csv)
        
        # Get condition columns (exclude report column)
        condition_cols = [c for c in df.columns if c != self.report_col]
        
        # Initialize binary array
        label_array = np.zeros((len(df), len(condition_cols)), dtype=int)
        
        for i in range(len(df)):
            for j, condition in enumerate(condition_cols):
                value = df[condition].iloc[i]
                
                try:
                    if pd.isna(value):
                        label_array[i, j] = 0  # Blank treated as negative
                    else:
                        num_value = float(value)
                        if num_value == 1:  # Positive
                            label_array[i, j] = 1
                        elif num_value == -1 and self.treat_uncertain_as_positive:  # Uncertain
                            label_array[i, j] = 1
                        else:  # Negative (0) or uncertain treated as negative
                            label_array[i, j] = 0
                except (ValueError, TypeError):
                    # Handle string labels
                    str_value = str(value).lower()
                    if 'pos' in str_value:
                        label_array[i, j] = 1
                    elif 'uncer' in str_value and self.treat_uncertain_as_positive:
                        label_array[i, j] = 1
                    else:
                        label_array[i, j] = 0
        
        return label_array, condition_cols
    
    def _compute_chexpert_metrics(self, 
                                 gt_labeled_csv: str, 
                                 pred_labeled_csv: str) -> Dict[str, Any]:
        """Compute CheXpert micro-F1 and per-condition metrics.
        
        Args:
            gt_labeled_csv: Path to ground truth labeled CSV
            pred_labeled_csv: Path to predictions labeled CSV
            
        Returns:
            Dictionary with micro-F1 and per-condition metrics
        """
        # Parse label arrays
        y_true, conditions = self._parse_chex_labels(gt_labeled_csv)
        y_pred, _ = self._parse_chex_labels(pred_labeled_csv)
        
        # Ensure same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Label arrays have different shapes: {y_true.shape} vs {y_pred.shape}")
        
        # Compute micro-averaged F1 (flatten all labels)
        micro_f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='micro')
        
        # Compute per-condition metrics
        per_condition = {}
        for j, condition in enumerate(conditions):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, j], y_pred[:, j], average='binary', zero_division=0
            )
            per_condition[condition] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
        
        return {
            'micro_f1': float(micro_f1),
            'per_condition': per_condition,
            'conditions': conditions,
            'total_samples': y_true.shape[0],
            'total_labels': y_true.shape[0] * y_true.shape[1]
        }
    
    def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Compute CheXpert micro-F1 and store results in summary.
        
        Note: This method doesn't add columns to pred_df since CheXpert metrics
        are computed at the dataset level, not per-report level.
        
        Args:
            gt_df: Ground truth dataframe with study_id and report columns
            pred_df: Predictions dataframe with study_id and report columns
            
        Returns:
            pred_df unchanged (CheXpert results stored in summary)
        """
        # Align dataframes
        gt_aligned, pred_aligned = self.align_dataframes(gt_df, pred_df)
        
        # Create cache CSV files for labeling
        cache_gt_csv = os.path.join(self.cache_dir, "chexpert_gt.csv")
        cache_pred_csv = os.path.join(self.cache_dir, "chexpert_pred.csv")
        
        # Save aligned dataframes
        gt_aligned.to_csv(cache_gt_csv, index=False)
        pred_aligned.to_csv(cache_pred_csv, index=False)
        
        # Run CheXbert labeling
        print("Running CheXbert labeling for predictions...")
        pred_labeled_csv = self._run_chexbert_labeler(cache_pred_csv, self.pred_labels_dir)
        
        print("Running CheXbert labeling for ground truth...")
        gt_labeled_csv = self._run_chexbert_labeler(cache_gt_csv, self.gt_labels_dir)
        
        # Compute metrics if labeling succeeded
        if pred_labeled_csv and gt_labeled_csv:
            try:
                chexpert_results = self._compute_chexpert_metrics(gt_labeled_csv, pred_labeled_csv)
                # Store results for retrieval by summary methods
                self._last_results = chexpert_results
                print(f"CheXpert micro-F1: {chexpert_results['micro_f1']:.4f}")
            except Exception as e:
                print(f"Warning: CheXpert metric computation failed: {e}")
                self._last_results = None
        else:
            print("Warning: CheXbert labeling failed")
            self._last_results = None
        
        return pred_aligned
    
    def get_metric_columns(self) -> List[str]:
        """Get the list of column names this metric adds.
        
        CheXpert evaluation is dataset-level, so no columns are added.
        
        Returns:
            Empty list (no per-report columns)
        """
        return []
    
    def get_chexpert_results(self) -> Optional[Dict[str, Any]]:
        """Get the most recent CheXpert evaluation results.
        
        Returns:
            Dictionary with CheXpert results or None if not available
        """
        return getattr(self, '_last_results', None)
    
    def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for CheXpert metric.
        
        Args:
            pred_df: Dataframe (not used for CheXpert dataset-level metrics)
            
        Returns:
            Dictionary with CheXpert summary statistics
        """
        summary = {}
        
        # Get CheXpert results
        chexpert_results = self.get_chexpert_results()
        
        if chexpert_results:
            summary['chexpert'] = chexpert_results
            
            # Add detailed analysis
            chexpert_info = {
                'description': 'CheXpert micro-F1 using CheXbert labeling across 14 pathology conditions',
                'labeling_approach': 'Uncertain labels treated as positive' if self.treat_uncertain_as_positive else 'Uncertain labels treated as negative',
                'evaluation_level': 'Dataset-level (not per-report)',
                'conditions_evaluated': len(chexpert_results.get('conditions', [])),
                'interpretation': {
                    'micro_f1': 'Micro-averaged F1 across all conditions and samples',
                    'range': '[0, 1] where 1 is perfect clinical finding accuracy',
                    'use_case': 'Evaluates clinical finding detection accuracy'
                }
            }
            
            # Analyze per-condition performance
            if 'per_condition' in chexpert_results:
                per_cond = chexpert_results['per_condition']
                condition_f1s = [metrics['f1'] for metrics in per_cond.values()]
                
                if condition_f1s:
                    best_condition = max(per_cond.items(), key=lambda x: x[1]['f1'])
                    worst_condition = min(per_cond.items(), key=lambda x: x[1]['f1'])
                    
                    chexpert_info['condition_analysis'] = {
                        'mean_condition_f1': float(np.mean(condition_f1s)),
                        'std_condition_f1': float(np.std(condition_f1s)),
                        'best_condition': {
                            'name': best_condition[0],
                            'f1': best_condition[1]['f1']
                        },
                        'worst_condition': {
                            'name': worst_condition[0],
                            'f1': worst_condition[1]['f1']
                        },
                        'conditions_above_05': sum(1 for f1 in condition_f1s if f1 > 0.5)
                    }
            
            # Performance interpretation
            micro_f1 = chexpert_results['micro_f1']
            if micro_f1 > 0.8:
                performance_level = "Excellent clinical finding accuracy"
            elif micro_f1 > 0.6:
                performance_level = "Good clinical finding accuracy"
            elif micro_f1 > 0.4:
                performance_level = "Moderate clinical finding accuracy"
            else:
                performance_level = "Poor clinical finding accuracy"
            
            chexpert_info['performance_assessment'] = {
                'level': performance_level,
                'micro_f1': micro_f1,
                'benchmark_note': 'Compare with radiologist agreement (~0.7-0.8 for many conditions)'
            }
            
            summary['chexpert_analysis'] = chexpert_info
        
        else:
            summary['chexpert_analysis'] = {
                'status': 'CheXpert evaluation not available',
                'note': 'Run compute_metric() first to generate CheXpert results'
            }
        
        return summary
    
    @property
    def name(self) -> str:
        """Get descriptive name for this evaluator."""
        return f"CheXpertEvaluator(uncertain={'positive' if self.treat_uncertain_as_positive else 'negative'})"
