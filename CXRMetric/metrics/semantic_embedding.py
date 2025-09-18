"""
Semantic embedding evaluation using CheXbert encodings.

This module implements semantic similarity evaluation using CheXbert model
embeddings to measure semantic similarity between radiology reports.
"""

import pandas as pd
from typing import Any, Dict, List

try:
    # Prefer the implementation under the "optional" folder when present.
    from .optional.semantic_embedding import SemanticEmbeddingEvaluator  # type: ignore
except Exception:
    class SemanticEmbeddingEvaluator:
        """Disabled fallback evaluator when CheXbert/model artifacts are not present.

        This evaluator is intentionally permissive: compute_metric will add a
        zeroed 'semb_score' column and get_summary_stats returns a small
        summary dict indicating the metric was skipped. This prevents callers
        from crashing when attempting to instantiate the evaluator in minimal
        environments.
        """
        def __init__(self, *args, **kwargs):
            self._disabled = True

        def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
            pred_df = pred_df.copy()
            # Ensure a column exists so downstream code can continue.
            pred_df['semb_score'] = [0.0] * len(pred_df)
            print("⚠️ SemanticEmbeddingEvaluator: model not available — returning zeroed 'semb_score' values.")
            return pred_df

        def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
            return {'skipped': True, 'reason': 'Semantic embedding model not available'}

        def get_metric_columns(self) -> List[str]:
            return ['semb_score']

        @property
        def name(self) -> str:
            return 'DisabledSemanticEmbeddingEvaluator'
