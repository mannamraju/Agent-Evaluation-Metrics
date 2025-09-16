#!/usr/bin/env python3
"""
Compatibility shim for legacy `bleu_evaluation_summary` API.

The detailed functionality has been consolidated into `bleu_impl.py` and
exposed via the new `run_bleu_demo.py` entrypoint. This shim keeps the
original function names available for simple imports.
"""
from CXRMetric.metrics.bleu.bleu_impl import (
    run_bleu_evaluation_with_logging,
    analyze_bleu_strictness,
    display_bleu_evaluation_history,
)

__all__ = [
    'run_bleu_evaluation_with_logging',
    'analyze_bleu_strictness',
    'display_bleu_evaluation_history',
]

if __name__ == '__main__':
    print('This module has been consolidated into CXRMetric.metrics.bleu.run_bleu_demo and bleu_impl.')
    print('Please run: python -m CXRMetric.metrics.bleu.run_bleu_demo')
