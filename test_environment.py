#!/usr/bin/env python3
"""
Quick test script to verify the Agent-Evaluation-Metrics environment setup.

This script tests the main evaluation metrics to ensure they work correctly.
"""

import pandas as pd
import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        import transformers
        import bert_score
        import nltk
        from CXRMetric.modular_evaluation import ModularEvaluationRunner
        from CXRMetric.metrics.bleu import BLEUEvaluator
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_loading():
    """Test that sample data can be loaded."""
    print("Testing data loading...")
    
    try:
        gt_df = pd.read_csv('reports/gt_reports.csv')
        pred_df = pd.read_csv('reports/predicted_reports.csv')
        print(f"✓ Loaded {len(gt_df)} ground truth reports and {len(pred_df)} predictions")
        return gt_df, pred_df
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return None, None

def test_bleu_evaluation(gt_df, pred_df):
    """Test BLEU evaluation."""
    print("Testing BLEU evaluation...")
    
    try:
        evaluator = BLEUEvaluator()
        result_df = evaluator.compute_metric(gt_df, pred_df)
        
        if 'bleu4_score' in result_df.columns:
            mean_bleu4 = result_df['bleu4_score'].mean()
            print(f"✓ BLEU-4 evaluation successful (mean score: {mean_bleu4:.4f})")
        else:
            print("✓ BLEU evaluation completed (scores in other columns)")
        return True
    except Exception as e:
        print(f"✗ BLEU evaluation error: {e}")
        return False

def test_modular_runner():
    """Test the modular evaluation runner."""
    print("Testing modular evaluation runner...")
    
    try:
        runner = ModularEvaluationRunner()
        print("✓ Modular evaluation runner initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Modular runner error: {e}")
        return False

def main():
    """Run all tests."""
    print("Agent-Evaluation-Metrics Environment Test")
    print("=" * 45)
    print()
    
    # Test imports
    if not test_imports():
        print("\n❌ Environment setup failed - import errors")
        sys.exit(1)
    
    print()
    
    # Test data loading
    gt_df, pred_df = test_data_loading()
    if gt_df is None:
        print("\n❌ Environment setup failed - data loading errors")
        sys.exit(1)
    
    print()
    
    # Test BLEU evaluation
    if not test_bleu_evaluation(gt_df, pred_df):
        print("\n❌ Environment setup failed - BLEU evaluation errors")
        sys.exit(1)
    
    print()
    
    # Test modular runner
    if not test_modular_runner():
        print("\n❌ Environment setup failed - modular runner errors")
        sys.exit(1)
    
    print()
    print("🎉 All tests passed! Environment is ready.")
    print()
    print("Next steps:")
    print("1. Use './activate_env.sh' to activate the environment")
    print("2. Run evaluations with the ModularEvaluationRunner")
    print("3. Check the README.md for detailed usage instructions")

if __name__ == "__main__":
    main()