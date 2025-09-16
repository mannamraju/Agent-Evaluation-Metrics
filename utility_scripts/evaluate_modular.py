#!/usr/bin/env python3
"""
Command-line interface for modular CXR report evaluation.

This script provides a simple CLI for running individual or combined
evaluation metrics on chest X-ray report datasets.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from CXRMetric.modular_evaluation import ModularEvaluationRunner, evaluate_reports


def main():
    parser = argparse.ArgumentParser(
        description="Modular evaluation for CXR report generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all metrics
  python evaluate_modular.py ground_truth.csv predictions.csv --output results.csv
  
  # Run only BLEU and ROUGE
  python evaluate_modular.py ground_truth.csv predictions.csv --metrics bleu rouge
  
  # Run single metric with custom options
  python evaluate_modular.py ground_truth.csv predictions.csv --metrics bertscore --use-idf
  
  # Compare multiple metrics
  python evaluate_modular.py ground_truth.csv predictions.csv --compare --metrics bleu rouge bertscore
  
Available metrics:
  - bleu: BLEU-2 and BLEU-4 scores
  - rouge: ROUGE-L score
  - bertscore: BERTScore semantic similarity
  - semantic_embedding: CheXbert embedding similarity (requires model)
  - radgraph: RadGraph clinical information extraction (requires model)  
  - chexpert: (removed) CheXpert micro-F1 support has been removed from this repository
  - composite: RadCliQ composite metrics (requires other metrics)
  - bounding_box: Bounding box IoU evaluation (requires box data)
        """)
    
    # Required arguments
    parser.add_argument('gt_csv', help='Path to ground truth CSV file')
    parser.add_argument('pred_csv', help='Path to predictions CSV file')
    
    # Optional arguments
    parser.add_argument('--output', '-o', help='Path for output CSV file')
    parser.add_argument('--config', '-c', default='evaluation_config.json',
                       help='Path to configuration file (default: evaluation_config.json)')
    parser.add_argument('--metrics', '-m', nargs='*', default=['all'],
                       help='Metrics to compute (default: all)')
    parser.add_argument('--cache-dir', default='cache',
                       help='Directory for caching results (default: cache)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    parser.add_argument('--compare', action='store_true',
                       help='Compare metrics and show correlations')
    parser.add_argument('--list-metrics', action='store_true',
                       help='List available metrics and exit')
    
    # Metric-specific options
    parser.add_argument('--use-idf', action='store_true',
                       help='Use IDF weighting for BERTScore')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for bounding box evaluation')
    parser.add_argument('--rouge-beta', type=float, default=1.2,
                       help='Beta parameter for ROUGE-L')
    
    args = parser.parse_args()
    
    # Initialize runner
    try:
        runner = ModularEvaluationRunner(
            config_file=args.config if os.path.exists(args.config) else None,
            cache_dir=args.cache_dir
        )
    except Exception as e:
        print(f"Error initializing evaluation runner: {e}")
        return 1
    
    # List available metrics
    if args.list_metrics:
        metrics = runner.list_available_metrics()
        print("Available metrics:")
        for metric in sorted(metrics):
            print(f"  - {metric}")
        return 0
    
    # Validate input files
    if not os.path.exists(args.gt_csv):
        print(f"Error: Ground truth file not found: {args.gt_csv}")
        return 1
    
    if not os.path.exists(args.pred_csv):
        print(f"Error: Predictions file not found: {args.pred_csv}")
        return 1
    
    # Resolve metrics
    if args.metrics == ['all']:
        metrics_to_run = 'all'
    else:
        metrics_to_run = args.metrics
        
        # Validate metrics
        available = runner.list_available_metrics()
        invalid = [m for m in metrics_to_run if m not in available]
        if invalid:
            print(f"Error: Invalid metrics: {invalid}")
            print(f"Available metrics: {available}")
            return 1
    
    # Prepare metric options
    metric_options = {}
    
    if args.use_idf:
        metric_options['use_idf'] = True
    
    if args.iou_threshold != 0.5:
        metric_options['iou_threshold'] = args.iou_threshold
    
    if args.rouge_beta != 1.2:
        metric_options['beta'] = args.rouge_beta
    
    print(f"📊 CXR Report Evaluation")
    print(f"Ground truth: {args.gt_csv}")
    print(f"Predictions: {args.pred_csv}")
    print(f"Metrics: {metrics_to_run}")
    print()
    
    try:
        # Run evaluation
        results_df, summary = runner.run_evaluation(
            args.gt_csv,
            args.pred_csv,
            metrics_to_run,
            args.output,
            use_cache=not args.no_cache,
            **metric_options
        )
        
        # Print summary
        print("\n📈 Results Summary:")
        print("=" * 50)
        
        if 'evaluation_info' in summary:
            eval_info = summary['evaluation_info']
            print(f"Metrics computed: {', '.join(eval_info['metrics_computed'])}")
            print(f"Total time: {eval_info.get('total_time', 0):.1f}s")
            
            if 'timing' in eval_info:
                print("\nTiming breakdown:")
                for metric, time_taken in eval_info['timing'].items():
                    print(f"  {metric}: {time_taken:.1f}s")
        
        # Show metric results
        for metric_name, metric_data in summary.items():
            if metric_name == 'evaluation_info':
                continue
                
            print(f"\n{metric_name.upper()}:")
            
            # Show main metric values
            if isinstance(metric_data, dict) and 'mean_metrics' in metric_data:
                for col, value in metric_data['mean_metrics'].items():
                    print(f"  {col}: {value:.4f}")
            
            # Show dataset-level metrics
            # CheXpert support removed — chexpert results will not be present
            
        # Dataset-level results
        if 'bounding_box_results' in summary:
            bbox_results = summary['bounding_box_results']
            print(f"\nBOUNDING BOX RESULTS:")
            print(f"  Precision: {bbox_results['precision']:.4f}")
            print(f"  Recall: {bbox_results['recall']:.4f}")
            print(f"  F1: {bbox_results['f1_score']:.4f}")
        
        # Comparison analysis
        if args.compare and len(args.metrics) > 1:
            print(f"\n🔗 METRIC CORRELATIONS:")
            try:
                comparison = runner.compare_metrics(args.gt_csv, args.pred_csv, args.metrics)
                
                if 'high_correlations' in comparison:
                    high_corr = comparison['high_correlations']
                    if high_corr:
                        print("High correlations (|r| ≥ 0.7):")
                        for corr_info in high_corr:
                            print(f"  {corr_info['metric1']} ↔ {corr_info['metric2']}: r = {corr_info['correlation']:.3f}")
                    else:
                        print("No high correlations found")
            except Exception as e:
                print(f"Comparison analysis failed: {e}")
        
        print(f"\n✅ Evaluation completed successfully!")
        if args.output:
            print(f"Results saved to: {args.output}")
            print(f"Summary saved to: {args.output}.summary.json")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
