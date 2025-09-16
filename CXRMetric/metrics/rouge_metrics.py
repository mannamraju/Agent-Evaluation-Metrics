"""
ROUGE metrics import compatibility layer.

This module provides backward compatibility for importing ROUGEEvaluator
from the original location.
"""

# Import from the correct location
from .rouge.rouge_metrics import ROUGEEvaluator

__all__ = ['ROUGEEvaluator']