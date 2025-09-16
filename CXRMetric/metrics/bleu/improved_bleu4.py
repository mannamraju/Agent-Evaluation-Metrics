"""
Compatibility shim for legacy `improved_bleu4` API.

Core implementations were consolidated into `bleu_impl.py`. This shim
re-exports the main functions to preserve backward compatibility.
"""
from CXRMetric.metrics.bleu.bleu_impl import (
    compute_smoothed_bleu4,
    evaluate_medical_reports_bleu4,
)

__all__ = [
    'compute_smoothed_bleu4',
    'evaluate_medical_reports_bleu4',
]
