"""
RadGraph evaluation for CXR report generation.

This module implements RadGraph F1 evaluation which measures clinical
information extraction accuracy using entity and relation extraction.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional

from ..base_evaluator import BaseEvaluator


class RadGraphEvaluator(BaseEvaluator):
    """Evaluator for RadGraph F1 scores.
    
    RadGraph evaluates the clinical content of radiology reports by measuring
    the accuracy of extracted entities and relations using a structured clinical
    information extraction approach.
    """
    
    def __init__(self, 
                 radgraph_path: str,
                 cache_dir: str = "cache",
                 **kwargs):
        """Initialize RadGraph evaluator.
        
        Args:
            radgraph_path: Path to RadGraph model/evaluation script
            cache_dir: Directory for caching results
            **kwargs: Additional arguments passed to BaseEvaluator
        """
        super().__init__(**kwargs)
        self.radgraph_path = radgraph_path
        self.cache_dir = cache_dir
        
        # Cache file paths
        self.entities_path = os.path.join(cache_dir, "entities_cache.json")
        self.relations_path = os.path.join(cache_dir, "relations_cache.json")
    
    def _run_radgraph_evaluation(self, 
                                gt_csv_path: str, 
                                pred_csv_path: str) -> None:
        """Run RadGraph evaluation to generate entity and relation scores.
        
        Args:
            gt_csv_path: Path to ground truth CSV
            pred_csv_path: Path to predictions CSV
        """
        # Import here to avoid circular imports
        from CXRMetric.radgraph_evaluate_model import run_radgraph
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print("Running RadGraph evaluation...")
        run_radgraph(
            gt_csv_path, 
            pred_csv_path, 
            self.cache_dir, 
            self.radgraph_path,
            self.entities_path, 
            self.relations_path
        )
    
    def _load_radgraph_scores(self) -> Dict[int, float]:
        """Load and combine RadGraph entity and relation scores.
        
        Returns:
            Dictionary mapping study_id to combined RadGraph F1 score
        """
        if not os.path.exists(self.entities_path) or not os.path.exists(self.relations_path):
            raise FileNotFoundError("RadGraph cache files not found. Run evaluation first.")
        
        study_id_to_radgraph = {}
        
        # Load entity scores
        with open(self.entities_path, "r") as f:
            entity_scores = json.load(f)
            for study_id, (f1, _, _) in entity_scores.items():
                try:
                    study_id_to_radgraph[int(study_id)] = float(f1)
                except (ValueError, TypeError):
                    continue
        
        # Load and combine relation scores
        with open(self.relations_path, "r") as f:
            relation_scores = json.load(f)
            for study_id, (f1, _, _) in relation_scores.items():
                try:
                    study_id_int = int(study_id)
                    if study_id_int in study_id_to_radgraph:
                        # Average entity and relation F1 scores
                        study_id_to_radgraph[study_id_int] += float(f1)
                        study_id_to_radgraph[study_id_int] /= 2.0
                except (ValueError, TypeError):
                    continue
        
        return study_id_to_radgraph
    
    def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Compute RadGraph F1 scores and add as column to pred_df.
        
        Args:
            gt_df: Ground truth dataframe with study_id and report columns
            pred_df: Predictions dataframe with study_id and report columns
            
        Returns:
            Updated pred_df with radgraph_combined column added
        """
        # Align dataframes
        gt_aligned, pred_aligned = self.align_dataframes(gt_df, pred_df)
        
        # Create cache CSV files for RadGraph evaluation
        cache_gt_csv = os.path.join(self.cache_dir, "radgraph_gt.csv")
        cache_pred_csv = os.path.join(self.cache_dir, "radgraph_pred.csv")
        
        # Save aligned dataframes
        gt_aligned.to_csv(cache_gt_csv, index=False)
        pred_aligned.to_csv(cache_pred_csv, index=False)
        
        # Run RadGraph evaluation if cache doesn't exist
        if not os.path.exists(self.entities_path) or not os.path.exists(self.relations_path):
            self._run_radgraph_evaluation(cache_gt_csv, cache_pred_csv)
        
        # Load scores
        try:
            study_id_to_scores = self._load_radgraph_scores()
            
            # Map scores to dataframe
            radgraph_scores = []
            missing_scores = 0
            
            for _, row in pred_aligned.iterrows():
                study_id = int(row[self.study_id_col])
                if study_id in study_id_to_scores:
                    radgraph_scores.append(study_id_to_scores[study_id])
                else:
                    radgraph_scores.append(0.0)
                    missing_scores += 1
            
            pred_aligned["radgraph_combined"] = radgraph_scores
            
            if missing_scores > 0:
                print(f"Warning: {missing_scores} study IDs missing RadGraph scores, set to 0.0")
        
        except Exception as e:
            print(f"Warning: RadGraph evaluation failed: {e}")
            print("Setting radgraph_combined to 0.0")
            pred_aligned["radgraph_combined"] = [0.0] * len(pred_aligned)
        
        return pred_aligned
    
    def get_metric_columns(self) -> List[str]:
        """Get the list of column names this metric adds.
        
        Returns:
            List containing 'radgraph_combined'
        """
        return ["radgraph_combined"]
    
    def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for RadGraph metric.
        
        Args:
            pred_df: Dataframe containing computed RadGraph scores
            
        Returns:
            Dictionary with RadGraph summary statistics
        """
        summary = super().get_summary_stats(pred_df)
        
        # Add RadGraph-specific analysis
        radgraph_info = {
            'description': 'RadGraph F1 measures clinical information extraction accuracy',
            'components': 'Average of entity F1 and relation F1 scores',
            'range': '[0, 1] where 1 is perfect clinical content match',
            'interpretation': {
                'entities': 'Medical concepts and anatomical structures',
                'relations': 'Clinical relationships between entities',
                'use_case': 'Evaluates clinical accuracy and completeness'
            }
        }
        
        if 'radgraph_combined' in summary and summary['radgraph_combined']:
            radgraph_scores = pred_df['radgraph_combined'].dropna()
            if len(radgraph_scores) > 0:
                # Analyze score distribution
                zero_scores = (radgraph_scores == 0.0).sum()
                low_scores = (radgraph_scores < 0.3).sum()
                high_scores = (radgraph_scores > 0.7).sum()
                
                radgraph_info['score_distribution'] = {
                    'zero_scores_count': int(zero_scores),
                    'low_scores_pct': f"{100 * low_scores / len(radgraph_scores):.1f}% (< 0.3)",
                    'high_scores_pct': f"{100 * high_scores / len(radgraph_scores):.1f}% (> 0.7)",
                    'note': 'RadGraph > 0.5 indicates good clinical content preservation'
                }
                
                # Clinical performance interpretation
                mean_score = summary['radgraph_combined']['mean']
                if mean_score > 0.6:
                    performance_level = "Good clinical accuracy"
                elif mean_score > 0.4:
                    performance_level = "Moderate clinical accuracy"
                elif mean_score > 0.2:
                    performance_level = "Low clinical accuracy"
                else:
                    performance_level = "Poor clinical accuracy"
                
                radgraph_info['clinical_performance'] = {
                    'level': performance_level,
                    'mean_score': mean_score,
                    'interpretation_note': 'Higher scores indicate better clinical content preservation'
                }
                
                # Check for evaluation issues
                if zero_scores > len(radgraph_scores) * 0.1:
                    radgraph_info['warnings'] = [
                        f"High number of zero scores ({zero_scores}/{len(radgraph_scores)})",
                        "This may indicate issues with RadGraph evaluation"
                    ]
        
        summary['radgraph_analysis'] = radgraph_info
        return summary
    
    @property
    def name(self) -> str:
        """Get descriptive name for this evaluator."""
        return "RadGraphEvaluator(entity+relation F1)"
