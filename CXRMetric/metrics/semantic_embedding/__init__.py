"""
Semantic Embedding Evaluation Module

This module provides semantic similarity evaluation using CheXbert embeddings
for chest X-ray report quality assessment.
"""

from .semantic_embedding_metrics import SemanticEmbeddingEvaluator

__all__ = ['SemanticEmbeddingEvaluator']
