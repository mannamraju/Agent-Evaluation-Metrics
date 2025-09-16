"""
BERTScore metrics package for CXR Report evaluation.

This package contains BERTScore implementation for evaluating semantic similarity
between chest X-ray reports using contextual embeddings from BERT models.
"""

from .bertscore_metrics import BERTScoreEvaluator

__all__ = ['BERTScoreEvaluator']
