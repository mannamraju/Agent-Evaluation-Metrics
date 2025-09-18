#!/usr/bin/env python3
"""
Direct BLEU-4 Test

Tests BLEU-4 functionality directly without importing problematic modules.
"""

import pandas as pd
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import only the BLEU evaluator directly
from CXRMetric.metrics.bleu.bleu_metrics import BLEUEvaluator

def test_bleu_direct():
    """Test BLEU-4 with direct implementation."""
    
    print("🧪 Direct BLEU-4 Test")
    print("=" * 40)
    
    # Test cases with different similarity levels
    test_cases = [
        {
            'gt': "The heart is normal in size.",
            'pred': "The heart is normal in size.",  # Identical
            'label': 'Perfect match'
        },
        {
            'gt': "The heart is normal in size and shape.",
            'pred': "Heart is normal in size and shape.",  # Missing "The"
            'label': 'Near perfect'
        },
        {
            'gt': "No acute cardiopulmonary abnormalities are identified.",
            'pred': "No acute abnormalities identified.",  # Missing words
            'label': 'Good match'
        },
        {
            'gt': "The lungs are clear bilaterally without infiltrate.",
            'pred': "Chest X-ray shows normal findings.",  # Different phrasing
            'label': 'Different phrasing'
        }
    ]
    
    # Create test dataframes
    gt_data = []
    pred_data = []
    
    for i, case in enumerate(test_cases):
        gt_data.append({'study_id': i+1, 'report': case['gt']})
        pred_data.append({'study_id': i+1, 'report': case['pred']})
    
    gt_df = pd.DataFrame(gt_data)
    pred_df = pd.DataFrame(pred_data)
    
    # Run BLEU evaluation
    evaluator = BLEUEvaluator(compute_bleu2=True, compute_bleu4=True)
    results = evaluator.compute_metric(gt_df, pred_df)
    
    print("\n📊 BLEU Score Analysis:")
    print("-" * 50)
    
    for i, case in enumerate(test_cases):
        row = results[results['study_id'] == i+1].iloc[0]
        bleu2 = row['bleu_score']
        bleu4 = row['bleu4_score']
        ratio = bleu2 / bleu4 if bleu4 > 0 else float('inf')
        
        print(f"\n{case['label']}:")
        print(f"  GT:     '{case['gt']}'")
        print(f"  Pred:   '{case['pred']}'")
        print(f"  BLEU-2: {bleu2:.4f}")
        print(f"  BLEU-4: {bleu4:.4f}")
        print(f"  Ratio:  {ratio:.2f}x" if ratio != float('inf') else "  Ratio:  ∞")
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"  Mean BLEU-2: {results['bleu_score'].mean():.4f}")
    print(f"  Mean BLEU-4: {results['bleu4_score'].mean():.4f}")
    print(f"  Mean Ratio:  {(results['bleu_score'] / results['bleu4_score']).replace([float('inf')], None).mean():.2f}x")
    
    # Implementation check
    print(f"\n🔧 Implementation Info:")
    try:
        import nltk
        print("  • Using NLTK (standard)")
    except ImportError:
        print("  • Using custom implementation (fallback)")
    
    return results

if __name__ == "__main__":
    try:
        results = test_bleu_direct()
        print("\n✅ BLEU-4 test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
