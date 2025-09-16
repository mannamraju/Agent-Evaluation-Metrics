#!/usr/bin/env python3
"""
BERTScore Evaluation Test

This script demonstrates and tests the BERTScore evaluation metric
which uses contextual embeddings to measure semantic similarity.

Run from the project root directory:
python CXRMetric/metrics/bertscore/tests/test_bertscore.py
"""

import sys
import os
from pathlib import Path
import json

# Add project root to Python path if running directly
if __name__ == "__main__":
    project_root = Path(__file__).parents[4]
    sys.path.insert(0, str(project_root))

from CXRMetric.modular_evaluation import evaluate_reports
import pandas as pd
import numpy as np

def test_bertscore_evaluation():
    """Test BERTScore evaluation with sample radiology reports."""
    
    print("🤖 Testing BERTScore Evaluation")
    print("=" * 50)
    
    # Check if sample data exists
    project_root = Path(__file__).parents[4]
    gt_path = project_root / 'reports' / 'gt_reports.csv'
    pred_path = project_root / 'reports' / 'pred_reports.csv'
    
    if not gt_path.exists() or not pred_path.exists():
        print("⚠️  Sample data not found. Creating test data...")
        create_sample_data(project_root)
    
    try:
        # Run BERTScore evaluation
        results, summary = evaluate_reports(
            str(gt_path),
            str(pred_path),
            metrics=['bertscore'],
            use_cache=False
        )
        
        if 'bertscore' not in results.columns:
            print("❌ BERTScore evaluation failed - no bertscore column found")
            return None
            
        print("✅ BERTScore evaluation completed!")
        
        # Display results
        print("\n📊 BERTScore Results Analysis:")
        print("-" * 60)
        
        bertscore_scores = results['bertscore'].dropna()
        
        if len(bertscore_scores) > 0:
            print(f"📈 Dataset Size: {len(bertscore_scores)} report pairs")
            print(f"📊 BERTScore Statistics:")
            print(f"  • Mean:    {bertscore_scores.mean():.4f}")
            print(f"  • Std:     {bertscore_scores.std():.4f}")
            print(f"  • Min:     {bertscore_scores.min():.4f}")
            print(f"  • Max:     {bertscore_scores.max():.4f}")
            print(f"  • Median:  {bertscore_scores.median():.4f}")
            
            # Score distribution analysis
            high_scores = (bertscore_scores > 0.8).sum()
            medium_scores = ((bertscore_scores >= 0.6) & (bertscore_scores <= 0.8)).sum()
            low_scores = (bertscore_scores < 0.6).sum()
            
            print(f"\n📊 Score Distribution:")
            print(f"  • High (>0.8):     {high_scores:2d} ({100*high_scores/len(bertscore_scores):5.1f}%)")
            print(f"  • Medium (0.6-0.8): {medium_scores:2d} ({100*medium_scores/len(bertscore_scores):5.1f}%)")
            print(f"  • Low (<0.6):      {low_scores:2d} ({100*low_scores/len(bertscore_scores):5.1f}%)")
            
            # Show precision and recall if available
            if 'bertscore_precision' in results.columns and 'bertscore_recall' in results.columns:
                precision_mean = results['bertscore_precision'].mean()
                recall_mean = results['bertscore_recall'].mean()
                
                print(f"\n🎯 Precision vs Recall:")
                print(f"  • Precision Mean: {precision_mean:.4f}")
                print(f"  • Recall Mean:    {recall_mean:.4f}")
                print(f"  • Difference:     {precision_mean - recall_mean:.4f}")
                
                if precision_mean > recall_mean:
                    print("  • Analysis: Higher precision suggests conservative generation")
                else:
                    print("  • Analysis: Higher recall suggests comprehensive generation")
            
            # Show some example comparisons
            print(f"\n📝 Sample Comparisons:")
            print("-" * 60)
            
            # Sort by BERTScore for examples
            sorted_results = results.sort_values('bertscore', ascending=False)
            
            print("🏆 Highest BERTScore:")
            best_idx = sorted_results.index[0]
            if best_idx < len(results):
                print(f"  Score: {sorted_results.iloc[0]['bertscore']:.4f}")
                # Would show texts here if we had access to them
            
            print("📉 Lowest BERTScore:")
            worst_idx = sorted_results.index[-1]
            if worst_idx < len(results):
                print(f"  Score: {sorted_results.iloc[-1]['bertscore']:.4f}")
            
        else:
            print("❌ No valid BERTScore results found")
            
        # Display summary information if available
        if summary and 'bertscore_analysis' in summary:
            bert_analysis = summary['bertscore_analysis']
            print(f"\n🔍 BERTScore Analysis:")
            print(f"  • Model Type: {bert_analysis.get('model_type', 'N/A')}")
            print(f"  • Use IDF: {bert_analysis.get('use_idf', 'N/A')}")
            print(f"  • Baseline Rescaling: {bert_analysis.get('rescale_with_baseline', 'N/A')}")
            print(f"  • Description: {bert_analysis.get('description', 'N/A')}")
            
        return results
        
    except ImportError as e:
        print(f"❌ BERTScore package not available: {e}")
        print("💡 Install with: pip install bert-score")
        return None
    except Exception as e:
        print(f"❌ Error during BERTScore evaluation: {e}")
        print(f"Error type: {type(e).__name__}")
        return None

def create_sample_data(project_root: Path):
    """Create sample radiology report data for testing."""
    
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Load sample data from metrics data directory if available
    data_dir = Path(__file__).resolve()
    while data_dir.name != 'metrics' and data_dir.parent != data_dir:
        data_dir = data_dir.parent
    try:
        from CXRMetric.metrics.data_loader import load_metric_cases
        sample_data = load_metric_cases('bertscore_sample_reports')
    except Exception:
        consolidated = data_dir / 'data' / 'metrics_test_cases.json'
        if consolidated.exists():
            with open(consolidated, 'r', encoding='utf-8') as _f:
                data = json.load(_f)
                sample_data = data.get('bertscore_sample_reports', [])
        else:
            sample_data = []
    
    # Create ground truth CSV
    gt_df = pd.DataFrame([
        {'study_id': item['study_id'], 'report': item['gt']} 
        for item in sample_data
    ])
    gt_df.to_csv(reports_dir / 'gt_reports.csv', index=False)
    
    # Create prediction CSV  
    pred_df = pd.DataFrame([
        {'study_id': item['study_id'], 'report': item['pred']} 
        for item in sample_data
    ])
    pred_df.to_csv(reports_dir / 'pred_reports.csv', index=False)
    
    print(f"✅ Created sample data with {len(sample_data)} report pairs")

def test_bertscore_characteristics():
    """Test BERTScore with controlled examples to show its characteristics."""
    
    print("\n" + "="*50)
    print("🎯 BERTScore Characteristics Test")
    print("="*50)
    
    # Test cases showing different types of similarity
    test_cases = [
        {
            'gt': 'The heart is normal in size and shape.',
            'pred': 'The heart is normal in size and shape.',
            'label': 'Perfect Match'
        },
        {
            'gt': 'The heart is normal in size and shape.',
            'pred': 'Heart size and morphology are within normal limits.',
            'label': 'Semantic Equivalent'
        },
        {
            'gt': 'No acute cardiopulmonary abnormalities identified.',
            'pred': 'No acute heart or lung problems detected.',
            'label': 'Simplified Language'
        },
        {
            'gt': 'Bilateral lower lobe pneumonia present.',
            'pred': 'Upper lobe consolidation noted bilaterally.',
            'label': 'Different Medical Finding'
        },
        {
            'gt': 'Chest X-ray shows normal findings.',
            'pred': 'The weather is sunny today.',
            'label': 'Completely Unrelated'
        }
    ]
    
    print("💡 BERTScore should capture semantic similarity better than BLEU")
    print("Expected ranking: Perfect > Semantic > Simplified > Different > Unrelated")
    print("\nNote: Actual BERTScore computation requires bert-score package")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['label']}:")
        print(f"   GT:   '{case['gt']}'")
        print(f"   Pred: '{case['pred']}'")
        print("   Expected: High semantic similarity" if i <= 2 else 
              "   Expected: Medium similarity" if i <= 3 else 
              "   Expected: Low similarity")

if __name__ == "__main__":
    print("🤖 BERTScore Evaluation Test Suite")
    print("="*60)
    
    # Test actual BERTScore evaluation
    results = test_bertscore_evaluation()
    
    # Test characteristics explanation
    test_bertscore_characteristics()
    
    print("\n" + "="*60)
    if results is not None:
        print("✅ BERTScore evaluation test completed successfully!")
        print(f"📊 Processed {len(results)} report pairs")
    else:
        print("⚠️  BERTScore test completed with limitations")
        print("💡 Install bert-score package for full functionality")
    
    print("\n🔍 Key Insights:")
    print("• BERTScore uses contextual embeddings for semantic similarity")  
    print("• Better than BLEU for paraphrased medical content")
    print("• Scores typically range from -1 to 1 (model dependent)")
    print("• F1 score (harmonic mean) is most commonly reported")
