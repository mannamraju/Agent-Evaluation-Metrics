"""
Compatibility shim for legacy `bleu_tools` API.

The implementation was consolidated into `bleu_impl.py`. This shim
re-exports the primary helper functions so existing imports continue
working while keeping the heavy implementations centralized.
"""
from CXRMetric.metrics.bleu.bleu_impl import (
    run_bleu_evaluator_consolidated,
    run_smoothed_examples,
)

# Keep the original function names available
__all__ = [
    'run_bleu_evaluator_consolidated',
    'run_smoothed_examples',
]
