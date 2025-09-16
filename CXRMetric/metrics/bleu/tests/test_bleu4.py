#!/usr/bin/env python3
"""
B    # Run BLEU evaluation (adjust path for test location)
    project_root = Path(__file__).parents[4]
    gt_path = project_root / 'reports' / 'gt_reports.csv'
    pred_path = project_root / 'reports' / 'pred_reports.csv'
    
    results, summary = evaluate_reports(
        str(gt_path),
        str(pred_path),4 Test Script

This script demonstrates and tests the BLEU-4 evaluation metric
which considers 1-4 gram overlaps for comprehensive text similarity.

Run from the project root directory:
python CXRMetric/metrics/bleu/tests/test_bleu4.py
"""

import sys
import os
from pathlib import Path

# Add project root to Python path if running directly
if __name__ == "__main__":
    project_root = Path(__file__).parents[4]
    sys.path.insert(0, str(project_root))

from CXRMetric.modular_evaluation import evaluate_reports
import pandas as pd
import numpy as np

def test_bleu_4():
    """Integration-style smoke test that invokes the modular runner for BLEU.

    This test asserts basic invariants (columns exist, BLEU-2 >= BLEU-4)
    and is intentionally lightweight to serve as a quick CI-level check.
    """
    gt = 'reports/gt_reports.csv'
    pred = 'reports/predicted_reports.csv'

    results, summary = evaluate_reports(gt, pred, metrics=['bleu'], use_cache=False)

    # Basic assertions
    assert 'bleu_score' in results.columns
    assert 'bleu4_score' in results.columns
    assert len(results) > 0

    # BLEU-2 should not be smaller than BLEU-4
    assert results['bleu_score'].mean() >= results['bleu4_score'].mean() - 1e-12

    # A minimal sanity check on summary
    assert 'bleu' in summary

    # No return value — pytest will treat assertion failures as test failures
    return None

if __name__ == "__main__":
    results, summary = test_bleu_4()
    print("\n✅ BLEU-4 test completed successfully!")
