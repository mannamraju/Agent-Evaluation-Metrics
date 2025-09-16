"""Compatibility wrapper for BLEU evaluator.

Historically some modules imported BLEUEvaluator from
CXRMetric.metrics.bleu_metrics. The canonical implementation now lives
in CXRMetric.metrics.bleu. Re-export the class here for backward
compatibility.
"""

from .bleu.bleu_metrics import BLEUEvaluator

__all__ = ["BLEUEvaluator"]
