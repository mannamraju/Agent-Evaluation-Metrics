#!/usr/bin/env python3
"""
Example usage of the modular CXR evaluation system.

This script demonstrates how to use individual metrics and the
modular evaluation runner with various configurations.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from CXRMetric.modular_evaluation import ModularEvaluationRunner, evaluate_reports


def create_sample_data():
    """Create sample data for demonstration."""
    
    # Sample reports
    gt_reports = [
        "The chest X-ray shows clear lungs with no acute cardiopulmonary process.",
        "There is a small pleural effusion on the right side. The heart size is normal.",
        "Bilateral lower lobe pneumonia is present. Recommend follow-up imaging.",
        "No acute findings. The cardiomediastinal silhouette is within normal limits.",
        "Left upper lobe consolidation consistent with pneumonia. Clinical correlation advised."
    ]
    
    pred_reports = [
        "Clear lungs, no acute process identified.",
        "Small right pleural effusion noted. Heart appears normal size.", 
        "Bilateral pneumonia in lower lobes. Follow-up recommended.",
        "Normal chest X-ray without acute abnormalities.",
        "Left upper lobe pneumonia present. Suggest clinical correlation."
    ]
    
    study_ids = [f"study_{i:03d}" for i in range(len(gt_reports))]
    
    # Create dataframes
    gt_df = pd.DataFrame({
        'study_id': study_ids,
        'report': gt_reports
    })
    
    pred_df = pd.DataFrame({
        'study_id': study_ids, 
        'report': pred_reports
    })
    
    return gt_df, pred_df


def example_basic_usage():
    """Example 1: Basic usage with fast metrics."""
    print("🚀 Example 1: Basic Usage")
    print("=" * 50)
    
    gt_df, pred_df = create_sample_data()
    
    # Save to temporary files
    gt_df.to_csv('temp_gt.csv', index=False)
    pred_df.to_csv('temp_pred.csv', index=False)
    
    try:
        # Run fast metrics only
        results_df, summary = evaluate_reports(
            gt_csv='temp_gt.csv',
            pred_csv='temp_pred.csv',
            metrics=['bleu', 'rouge'],
            use_cache=False
        )
        
        print("\nResults:")
        if 'bleu' in summary:
            bleu_stats = summary['bleu']
            if 'bleu_score' in bleu_stats:
                print(f"  BLEU-2 mean: {bleu_stats['bleu_score']['mean']:.4f}")
            if 'bleu4_score' in bleu_stats:
                print(f"  BLEU-4 mean: {bleu_stats['bleu4_score']['mean']:.4f}")
        
        if 'rouge' in summary:
            rouge_stats = summary['rouge']
            if 'rouge_l' in rouge_stats:
                print(f"  ROUGE-L mean: {rouge_stats['rouge_l']['mean']:.4f}")
        
        print(f"  Samples evaluated: {len(results_df)}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        import os
        for file in ['temp_gt.csv', 'temp_pred.csv']:
            if os.path.exists(file):
                os.remove(file)


def example_individual_metrics():
    """Example 2: Using individual metric evaluators."""
    print("\n🔧 Example 2: Individual Metric Evaluators")
    print("=" * 50)
    
    gt_df, pred_df = create_sample_data()
    
    # Initialize runner
    runner = ModularEvaluationRunner(cache_dir="example_cache")
    
    try:
        # Use individual BLEU evaluator
        print("Testing BLEU evaluator...")
        bleu_evaluator = runner.get_evaluator('bleu', compute_bleu2=True, compute_bleu4=True)
        results_df = bleu_evaluator.compute_metric(gt_df, pred_df)
        bleu_summary = bleu_evaluator.get_summary_stats(results_df)
        
        print(f"  BLEU columns added: {bleu_evaluator.get_metric_columns()}")
        if 'bleu_score' in bleu_summary:
            print(f"  BLEU-2 mean: {bleu_summary['bleu_score']['mean']:.4f}")
        
        # Use individual ROUGE evaluator  
        print("\nTesting ROUGE evaluator...")
        rouge_evaluator = runner.get_evaluator('rouge', beta=1.2)
        results_df = rouge_evaluator.compute_metric(gt_df, results_df)
        rouge_summary = rouge_evaluator.get_summary_stats(results_df)
        
        print(f"  ROUGE columns added: {rouge_evaluator.get_metric_columns()}")
        if 'rouge_l' in rouge_summary:
            print(f"  ROUGE-L mean: {rouge_summary['rouge_l']['mean']:.4f}")
        
        print(f"\nFinal columns in results: {list(results_df.columns)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def example_custom_configuration():
    """Example 3: Custom configuration and options."""
    print("\n⚙️ Example 3: Custom Configuration") 
    print("=" * 50)
    
    gt_df, pred_df = create_sample_data()
    
    # Save to files
    gt_df.to_csv('temp_gt.csv', index=False)
    pred_df.to_csv('temp_pred.csv', index=False)
    
    try:
        # Create custom runner with options
        runner = ModularEvaluationRunner(
            cache_dir="custom_cache",
            study_id_col="study_id",
            report_col="report"
        )
        
        # Run with custom metric options
        results_df, summary = runner.run_evaluation(
            gt_csv='temp_gt.csv',
            pred_csv='temp_pred.csv', 
            metrics=['rouge', 'bertscore'],
            use_cache=False,
            beta=1.5,  # Custom ROUGE beta
            use_idf=False  # BERTScore without IDF
        )
        
        print("Evaluation completed with custom options:")
        
        if 'rouge' in summary:
            rouge_analysis = summary['rouge'].get('rouge_analysis', {})
            beta = rouge_analysis.get('beta_parameter', 'unknown')
            print(f"  ROUGE-L with β={beta}")
        
        if 'bertscore' in summary:
            bert_analysis = summary['bertscore'].get('bertscore_analysis', {})
            use_idf = bert_analysis.get('use_idf', 'unknown')
            print(f"  BERTScore with IDF={use_idf}")
        
        if 'evaluation_info' in summary:
            timing = summary['evaluation_info'].get('timing', {})
            for metric, time_taken in timing.items():
                print(f"  {metric}: {time_taken:.2f}s")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        import os
        for file in ['temp_gt.csv', 'temp_pred.csv']:
            if os.path.exists(file):
                os.remove(file)


def example_metric_comparison():
    """Example 4: Metric comparison and correlation analysis."""
    print("\n📊 Example 4: Metric Comparison")
    print("=" * 50)
    
    # Create larger sample for better correlation analysis
    np.random.seed(42)
    n_samples = 20
    
    gt_reports = [f"Ground truth report {i} with various medical findings." for i in range(n_samples)]
    
    # Create predictions with different quality levels
    pred_reports = []
    for i in range(n_samples):
        if i < 5:  # High quality predictions
            pred_reports.append(f"Ground truth report {i} with various medical findings.")
        elif i < 10:  # Medium quality
            pred_reports.append(f"Report {i} showing medical findings and observations.")
        else:  # Lower quality
            pred_reports.append(f"Medical report number {i}.")
    
    gt_df = pd.DataFrame({
        'study_id': [f"study_{i:03d}" for i in range(n_samples)],
        'report': gt_reports
    })
    
    pred_df = pd.DataFrame({
        'study_id': [f"study_{i:03d}" for i in range(n_samples)],
        'report': pred_reports
    })
    
    # Save to files
    gt_df.to_csv('temp_gt.csv', index=False)
    pred_df.to_csv('temp_pred.csv', index=False)
    
    try:
        runner = ModularEvaluationRunner()
        
        # Compare multiple metrics
        comparison = runner.compare_metrics(
            gt_csv='temp_gt.csv',
            pred_csv='temp_pred.csv',
            metrics=['bleu', 'rouge'],
            correlation_threshold=0.5
        )
        
        print("Metric comparison results:")
        print(f"  Metrics compared: {comparison['metrics_compared']}")
        
        if 'high_correlations' in comparison:
            high_corr = comparison['high_correlations']
            if high_corr:
                print("  High correlations found:")
                for corr in high_corr:
                    print(f"    {corr['metric1']} ↔ {corr['metric2']}: r = {corr['correlation']:.3f}")
            else:
                print("  No high correlations found")
        
        if 'summary_stats' in comparison:
            print("  Summary statistics:")
            for metric, stats in comparison['summary_stats'].items():
                print(f"    {metric}: mean={stats.get('mean', 0):.3f}, std={stats.get('std', 0):.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback 
        traceback.print_exc()
    finally:
        # Clean up
        import os
        for file in ['temp_gt.csv', 'temp_pred.csv']:
            if os.path.exists(file):
                os.remove(file)


def main():
    """Run all examples."""
    print("🎯 CXR Modular Evaluation Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_individual_metrics()
    example_custom_configuration()
    example_metric_comparison()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("\nNext steps:")
    print("- Try the CLI: python evaluate_modular.py --help")
    print("- Check the configuration: evaluation_config.json")
    print("- Read the guide: MODULAR_README.md")


if __name__ == '__main__':
    main()
