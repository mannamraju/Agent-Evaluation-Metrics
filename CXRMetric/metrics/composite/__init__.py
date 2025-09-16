"""
Composite metrics package for CXR Report evaluation.

This package contains RadCliQ composite metrics implementation that combines
multiple evaluation metrics into single quality scores using trained models.
"""

from .composite_metrics import CompositeMetricEvaluator

__all__ = ['CompositeMetricEvaluator']
