"""
Modular evaluation runner for CXR report generation metrics.

This module provides a flexible evaluation pipeline that allows running
individual metrics or combinations of metrics as needed, with optional
caching and comprehensive reporting.
"""

import pandas as pd
import numpy as np
import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# Import metric evaluators
from .metrics.bleu import BLEUEvaluator
from .metrics.rouge.rouge_metrics import ROUGEEvaluator
from .metrics.bertscore import BERTScoreEvaluator
from .metrics.semantic_embedding import SemanticEmbeddingEvaluator
from .metrics.radgraph_metrics import RadGraphEvaluator
from .metrics.bounding_box_metrics import BoundingBoxEvaluator
from .metrics.composite_metrics import CompositeMetricEvaluator


class ModularEvaluationRunner:
    """Flexible evaluation runner that supports individual or combined metric evaluation.
    
    This runner allows users to select which specific metrics to compute,
    enabling faster evaluation when only certain metrics are needed.
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 cache_dir: str = "cache",
                 study_id_col: str = "study_id",
                 report_col: str = "report"):
        """Initialize the modular evaluation runner.
        
        Args:
            config_file: Optional path to configuration file with model paths
            cache_dir: Directory for caching results and intermediate files
            study_id_col: Name of study ID column
            report_col: Name of report text column
        """
        self.cache_dir = cache_dir
        self.study_id_col = study_id_col
        self.report_col = report_col
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize evaluators (lazy loading)
        self.evaluators = {}
        # Remove evaluators that depend on deleted model folders (CheXbert, DyGIE++)
        # These evaluators remain in the repo but are not enabled by default to
        # avoid import/runtime errors when model files are not present.
        self._available_metrics = {
            'bleu': BLEUEvaluator,
            'rouge': ROUGEEvaluator,
            'bertscore': BERTScoreEvaluator,
            'composite': CompositeMetricEvaluator,
            'bounding_box': BoundingBoxEvaluator
        }
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'chexbert_path': None,
            'radgraph_path': None,
            'composite_v0_path': 'CXRMetric/composite_metric_model.pkl',
            'composite_v1_path': 'CXRMetric/radcliq-v1.pkl',
            'normalizer_path': 'CXRMetric/normalizer.pkl'
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def list_available_metrics(self) -> List[str]:
        """Get list of available metric names.
        
        Returns:
            List of available metric identifiers
        """
        return list(self._available_metrics.keys())
    
    def get_evaluator(self, metric_name: str, **kwargs) -> Any:
        """Get or create an evaluator instance.
        
        Args:
            metric_name: Name of the metric evaluator
            **kwargs: Additional arguments for evaluator initialization
            
        Returns:
            Evaluator instance
        """
        if metric_name not in self._available_metrics:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {self.list_available_metrics()}")
        
        # Use cached evaluator if available
        cache_key = f"{metric_name}_{hash(str(sorted(kwargs.items())))}"
        if cache_key in self.evaluators:
            return self.evaluators[cache_key]
        
        # Create new evaluator with appropriate configuration
        evaluator_class = self._available_metrics[metric_name]
        
        # Add common parameters
        eval_kwargs = {
            'study_id_col': self.study_id_col,
            'report_col': self.report_col
        }
        
        # Add kwargs that don't conflict with base class
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key not in ['study_id_col', 'report_col']:
                filtered_kwargs[key] = value
        
        # Add metric-specific configuration
        if metric_name == 'bertscore':
            # BERTScore specifically supports use_idf
            eval_kwargs.update(filtered_kwargs)
        
        elif metric_name == 'semantic_embedding':
            eval_kwargs['chexbert_path'] = self.config['chexbert_path']
            eval_kwargs['cache_dir'] = self.cache_dir
        
        elif metric_name == 'radgraph':
            eval_kwargs['radgraph_path'] = self.config['radgraph_path']
            eval_kwargs['cache_dir'] = self.cache_dir
        
        # For other metrics (bleu, rouge, bounding_box), only add basic parameters
        # No additional kwargs to avoid parameter conflicts
        
        # Create and cache evaluator
        evaluator = evaluator_class(**eval_kwargs)
        self.evaluators[cache_key] = evaluator
        
        return evaluator
    
    def _generate_cache_key(self, 
                          gt_csv: str, 
                          pred_csv: str, 
                          metrics: List[str],
                          options: Dict[str, Any]) -> str:
        """Generate cache key for evaluation results.
        
        Args:
            gt_csv: Ground truth CSV path
            pred_csv: Predictions CSV path
            metrics: List of metrics to compute
            options: Evaluation options
            
        Returns:
            Cache key string
        """
        # Include file modification times for cache invalidation
        file_info = []
        for path in [gt_csv, pred_csv]:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                file_info.append(f"{path}:{mtime}")
        
        # Create hash from file info, metrics, and options
        content = "|".join(file_info) + "|" + "|".join(sorted(metrics)) + "|" + str(sorted(options.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cached_results(self, cache_key: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Load cached evaluation results.
        
        Args:
            cache_key: Unique cache identifier
            
        Returns:
            Tuple of (dataframe, summary) if cache exists, None otherwise
        """
        cache_path = os.path.join(self.cache_dir, f"eval_cache_{cache_key}.pkl")
        if os.path.exists(cache_path):
            try:
                import pickle
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"✓ Loaded cached results from {cache_path}")
                return cached_data['dataframe'], cached_data['summary']
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return None
    
    def _save_cached_results(self, 
                           cache_key: str, 
                           dataframe: pd.DataFrame, 
                           summary: Dict[str, Any]) -> None:
        """Save evaluation results to cache.
        
        Args:
            cache_key: Unique cache identifier
            dataframe: Results dataframe
            summary: Summary statistics
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"eval_cache_{cache_key}.pkl")
        
        try:
            import pickle
            cached_data = {
                'dataframe': dataframe,
                'summary': summary,
                'timestamp': time.time()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"✓ Saved results to cache: {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def run_evaluation(self,
                      gt_csv: str,
                      pred_csv: str,
                      metrics: Union[str, List[str]] = 'all',
                      output_csv: Optional[str] = None,
                      use_cache: bool = True,
                      **metric_options) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run evaluation with specified metrics.
        
        Args:
            gt_csv: Path to ground truth CSV file
            pred_csv: Path to predictions CSV file  
            metrics: Single metric name, list of metric names, or 'all'
            output_csv: Optional path to save results CSV
            use_cache: Whether to use cached results
            **metric_options: Additional options passed to metric evaluators
            
        Returns:
            Tuple of (results_dataframe, summary_dict)
        """
        print("🚀 Starting modular evaluation pipeline...")
        start_time = time.time()
        
        # Resolve metrics to evaluate
        if metrics == 'all':
            metrics_to_run = self.list_available_metrics()
        elif isinstance(metrics, str):
            metrics_to_run = [metrics]
        else:
            metrics_to_run = list(metrics)
        
        # Validate metrics
        invalid_metrics = [m for m in metrics_to_run if m not in self._available_metrics]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Available: {self.list_available_metrics()}")
        
        print(f"📊 Computing metrics: {', '.join(metrics_to_run)}")
        
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(gt_csv, pred_csv, metrics_to_run, metric_options)
            cached_results = self._load_cached_results(cache_key)
            if cached_results is not None:
                elapsed = time.time() - start_time
                print(f"⚡ Evaluation completed in {elapsed:.1f}s (from cache)")
                return cached_results
        
        # Load and align data
        print("📁 Loading datasets...")
        gt_df = pd.read_csv(gt_csv)
        pred_df = pd.read_csv(pred_csv)
        
        print(f"   Ground truth: {len(gt_df)} samples")
        print(f"   Predictions: {len(pred_df)} samples")
        
        # Find shared study IDs
        gt_study_ids = set(gt_df[self.study_id_col])
        pred_study_ids = set(pred_df[self.study_id_col])
        shared_study_ids = gt_study_ids.intersection(pred_study_ids)
        print(f"   Shared samples: {len(shared_study_ids)}")
        
        if len(shared_study_ids) == 0:
            raise ValueError("No shared study IDs found between ground truth and predictions")
        
        # Run individual metrics
        results_df = pred_df.copy()
        summary = {'evaluation_info': {'metrics_computed': [], 'timing': {}}}
        
        # Define evaluation order (dependencies)
        metric_dependencies = {
            'bleu': [],                # BLEU-2 and BLEU-4 (no dependencies)
            'rouge': [],               # ROUGE-L (no dependencies)
            'bertscore': [],           # BERTScore (no dependencies)
            'semantic_embedding': [],  # Semantic embedding (no dependencies)
            'radgraph': [],            # RadGraph (no dependencies)
            'bounding_box': [],        # Bounding box IoU (no dependencies, dataset-level)
            'composite': ['bleu', 'bertscore', 'semantic_embedding', 'radgraph']  # Needs other metrics
        }
        
        # Sort metrics by dependencies
        ordered_metrics = []
        remaining_metrics = set(metrics_to_run)
        
        while remaining_metrics:
            # Find metrics with satisfied dependencies
            ready_metrics = []
            for metric in remaining_metrics:
                deps = metric_dependencies.get(metric, [])
                if all(dep in ordered_metrics or dep not in metrics_to_run for dep in deps):
                    ready_metrics.append(metric)
            
            if not ready_metrics:
                # Add remaining metrics (circular dependencies or missing deps)
                ready_metrics = list(remaining_metrics)
            
            ordered_metrics.extend(ready_metrics)
            remaining_metrics -= set(ready_metrics)
        
        # Run metrics in dependency order
        for metric_name in ordered_metrics:
            print(f"🔄 Computing {metric_name}...")
            metric_start = time.time()
            
            try:
                # Get evaluator
                evaluator = self.get_evaluator(metric_name, **metric_options)
                
                # Run evaluation
                results_df = evaluator.compute_metric(gt_df, results_df)
                
                # Get summary statistics
                metric_summary = evaluator.get_summary_stats(results_df)
                summary[metric_name] = metric_summary
                
                # Track timing
                metric_time = time.time() - metric_start
                summary['evaluation_info']['timing'][metric_name] = metric_time
                summary['evaluation_info']['metrics_computed'].append(metric_name)
                
                print(f"   ✓ {metric_name} completed in {metric_time:.1f}s")
                
            except Exception as e:
                print(f"   ❌ {metric_name} failed: {e}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # Collect special results (dataset-level metrics)
        for metric_name in metrics_to_run:
            evaluator = self.evaluators.get(f"{metric_name}_{hash(str(sorted(metric_options.items())))}")
            if evaluator is None:
                continue
                
            # Get bounding box results
            if hasattr(evaluator, 'get_bbox_results'):
                bbox_results = evaluator.get_bbox_results()
                if bbox_results:
                    summary['bounding_box_results'] = bbox_results
        
        # Save results
        if output_csv:
            print(f"💾 Saving results to {output_csv}")
            results_df.to_csv(output_csv, index=False)
            
            # Save summary
            summary_path = f"{output_csv}.summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
        
        # Cache results
        if use_cache and cache_key:
            self._save_cached_results(cache_key, results_df, summary)
        
        elapsed_time = time.time() - start_time
        summary['evaluation_info']['total_time'] = elapsed_time
        
        print(f"✅ Evaluation completed in {elapsed_time:.1f}s")
        
        return results_df, summary
    
    def run_single_metric(self,
                         gt_csv: str,
                         pred_csv: str,
                         metric_name: str,
                         output_csv: Optional[str] = None,
                         **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Convenience method to run a single metric.
        
        Args:
            gt_csv: Path to ground truth CSV
            pred_csv: Path to predictions CSV
            metric_name: Name of metric to compute
            output_csv: Optional output path
            **kwargs: Additional arguments for the metric
            
        Returns:
            Tuple of (results_dataframe, summary_dict)
        """
        return self.run_evaluation(gt_csv, pred_csv, [metric_name], output_csv, **kwargs)
    
    def compare_metrics(self,
                       gt_csv: str,
                       pred_csv: str,
                       metrics: List[str],
                       correlation_threshold: float = 0.7) -> Dict[str, Any]:
        """Compare multiple metrics and analyze correlations.
        
        Args:
            gt_csv: Path to ground truth CSV
            pred_csv: Path to predictions CSV
            metrics: List of metrics to compare
            correlation_threshold: Threshold for reporting high correlations
            
        Returns:
            Dictionary with comparison analysis
        """
        results_df, summary = self.run_evaluation(gt_csv, pred_csv, metrics, use_cache=True)
        
        # Get metric columns
        metric_columns = []
        for metric in metrics:
            evaluator = self.get_evaluator(metric)
            metric_columns.extend(evaluator.get_metric_columns())
        
        # Filter to existing columns
        metric_columns = [col for col in metric_columns if col in results_df.columns]
        
        if len(metric_columns) < 2:
            return {'error': 'Need at least 2 metrics for comparison'}
        
        # Compute correlations
        corr_matrix = results_df[metric_columns].corr()
        
        # Find high correlations
        high_correlations = []
        for i in range(len(metric_columns)):
            for j in range(i + 1, len(metric_columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= correlation_threshold:
                    high_correlations.append({
                        'metric1': metric_columns[i],
                        'metric2': metric_columns[j],
                        'correlation': float(corr_val)
                    })
        
        return {
            'metrics_compared': metric_columns,
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations,
            'summary_stats': {col: results_df[col].describe().to_dict() for col in metric_columns}
        }


# Compatibility function matching run_eval.py signature
def calc_metric(gt_csv: str, 
               pred_csv: str, 
               out_csv: str, 
               use_idf: bool = True) -> Dict[str, Any]:
    """Compatibility function matching the signature of run_eval.py calc_metric.
    
    This provides backward compatibility for code that depends on the original
    calc_metric function while using the new modular evaluation system.
    
    Args:
        gt_csv: Path to ground truth CSV file
        pred_csv: Path to predictions CSV file
        out_csv: Path to output CSV file
        use_idf: Whether to use IDF weighting for BERTScore (passed to bertscore evaluator)
        
    Returns:
        Dictionary with evaluation summary (matching run_eval.py format)
    """
    # Set up evaluation with all metrics (matching run_eval.py behavior)
    runner = ModularEvaluationRunner()
    
    # Run comprehensive evaluation with all metrics
    results_df, summary = runner.run_evaluation(
        gt_csv=gt_csv,
        pred_csv=pred_csv,
        metrics='all',
        output_csv=out_csv,
        use_cache=True,
        use_idf=use_idf  # Pass IDF setting to BERTScore
    )
    
    # Reformat summary to match run_eval.py output format
    compat_summary = {}
    
    # Add mean metrics (matching run_eval.py format)
    mean_metrics = {}
    metric_columns = [
        'bleu_score', 'bleu4_score', 'rouge_l', 'bertscore', 
        'semb_score', 'radgraph_combined', 'RadCliQ-v0', 'RadCliQ-v1'
    ]
    
    for col in metric_columns:
        if col in results_df.columns:
            mean_metrics[col] = float(np.nanmean(results_df[col].values))
    
    compat_summary['mean_metrics'] = mean_metrics
    
    # Add bounding box results if available  
    if 'bounding_box_results' in summary:
        compat_summary['box_precision'] = summary['bounding_box_results']['precision']
        compat_summary['box_recall'] = summary['bounding_box_results']['recall']
    
    # Add evaluation metadata
    compat_summary['evaluation_info'] = summary.get('evaluation_info', {})
    
    return compat_summary


# Convenience function for quick evaluation
def evaluate_reports(gt_csv: str,
                    pred_csv: str,
                    metrics: Union[str, List[str]] = 'all',
                    output_csv: Optional[str] = None,
                    config_file: Optional[str] = None,
                    **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function for quick evaluation.
    
    Args:
        gt_csv: Path to ground truth CSV
        pred_csv: Path to predictions CSV
        metrics: Metrics to compute ('all' or list of metric names)
        output_csv: Optional output CSV path
        config_file: Optional configuration file
        **kwargs: Additional options
        
    Returns:
        Tuple of (results_dataframe, summary_dict)
    """
    runner = ModularEvaluationRunner(config_file=config_file)
    return runner.run_evaluation(gt_csv, pred_csv, metrics, output_csv, **kwargs)
