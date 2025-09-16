"""
BLEU metrics package for CXR Report evaluation.

This package contains BLEU-2 and BLEU-4 implementations for evaluating
the quality of chest X-ray report generation models.
"""

from .bleu_metrics import BLEUEvaluator

__all__ = ['BLEUEvaluator']
