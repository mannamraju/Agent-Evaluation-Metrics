"""
ROUGE metrics package for CXR Report evaluation.

This package contains ROUGE-L (Longest Common Subsequence) metric implementation
for evaluating chest X-ray report generation quality.
"""

from .rouge_metrics import ROUGEEvaluator

__all__ = ['ROUGEEvaluator']
