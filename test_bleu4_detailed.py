#!/usr/bin/env python3
"""
BLEU-4 Implementation Test

Tests the BLEU-4 functionality with controlled examples to demonstrate
the differences between BLEU-2 and BLEU-4 scoring.
"""

import pandas as pd
from CXRMetric.metrics.bleu.bleu_metrics import BLEUEvaluator

def test_bleu_implementation():
    """Test BLEU implementation with controlled examples."""
    
    print("🧪 Testing BLEU-4 Implementation")
    print("=" * 50)
    
    # Create test data with varying similarity levels
    test_data = [
        {
            'study_id': 1,
            'gt': "The heart is normal in size and shape.",
            'pred': "Heart is normal in size and shape.",  # Very similar
            'expected_level': 'High similarity'
        },
        {
            'study_id': 2, 
            'gt': "No acute cardiopulmonary abnormalities identified.",
            'pred': "No acute abnormalities seen.",  # Moderate similarity
            'expected_level': 'Moderate similarity'
        },
        {
            'study_id': 3,
            'gt': "The lungs are clear bilaterally without infiltrate.",
            'pred': "Chest X-ray shows normal findings.",  # Low similarity
            'expected_level': 'Low similarity'
        }
    ]
    
    # Create dataframes
    gt_df = pd.DataFrame([
        {'study_id': item['study_id'], 'report': item['gt']} 
        for item in test_data
    ])
    
    pred_df = pd.DataFrame([
        {'study_id': item['study_id'], 'report': item['pred']} 
        for item in test_data
    ])
    
    # Test BLEU evaluator
    evaluator = BLEUEvaluator(compute_bleu2=True, compute_bleu4=True)
    results = evaluator.compute_metric(gt_df, pred_df)
    
    print("\n📊 Controlled BLEU Test Results:")
    print("-" * 60)
    
    for i, item in enumerate(test_data):
        study_id = item['study_id']
        gt_text = item['gt']
        pred_text = item['pred']
        expected = item['expected_level']
        
        bleu2 = results.loc[results['study_id'] == study_id, 'bleu_score'].iloc[0]
        bleu4 = results.loc[results['study_id'] == study_id, 'bleu4_score'].iloc[0]
        
        print(f"\n🔍 Test Case {study_id} ({expected}):")
        print(f"  GT:     {gt_text}")
        print(f"  Pred:   {pred_text}")
        print(f"  BLEU-2: {bleu2:.4f}")
        print(f"  BLEU-4: {bleu4:.4f}")
        print(f"  Ratio:  {bleu2/bleu4 if bleu4 > 0 else 'inf'}")
    
    # Test fallback implementation info
    print("\n🛠️  Implementation Details:")
    print(f"• Using fallback BLEU implementation: {not hasattr(evaluator, '_fast_bleu_available')}")
    print("• BLEU-2 weights: (0.5, 0.5, 0.0, 0.0)")
    print("• BLEU-4 weights: (0.25, 0.25, 0.25, 0.25)")
    
    # Show why BLEU-4 is stricter
    print("\n💡 Why BLEU-4 is Stricter:")
    print("• Requires 4-gram matches for high scores")
    print("• Medical reports often lack long exact phrase matches")
    print("• Better at detecting fluency vs just content overlap")
    print("• More sensitive to word order and phrasing")
    
    return results

if __name__ == "__main__":
    print("This test was consolidated. Use pytest to run CXRMetric/metrics/bleu/tests/test_bleu_consolidated.py")
