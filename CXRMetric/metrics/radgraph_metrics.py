"""
RadGraph evaluation for CXR report generation.

This module implements RadGraph F1 evaluation which measures clinical
information extraction accuracy using entity and relation extraction.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional

from .base_evaluator import BaseEvaluator


# Shim wrapper: delegate to optional implementation when available; provide a
# disabled fallback that returns zeroed metric values and a skipped summary
# when the RadGraph model artifacts or optional module are not present.

try:
    from .optional.radgraph_metrics import RadGraphEvaluator  # type: ignore
except Exception:
    class RadGraphEvaluator:
        """Disabled RadGraph evaluator shim.

        If the real RadGraph implementation cannot be imported (because the
        model artifacts are missing or the optional module was not installed),
        this shim will add a 'radgraph_combined' column filled with zeros and
        return a minimal skipped summary to keep evaluation pipelines running.
        """
        def __init__(self, *args, **kwargs):
            self._disabled = True

        def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
            pred_df = pred_df.copy()
            pred_df['radgraph_combined'] = [0.0] * len(pred_df)
            print("⚠️ RadGraphEvaluator: model not available — returning zeroed 'radgraph_combined' values.")
            return pred_df

        def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
            return {'skipped': True, 'reason': 'RadGraph model not available'}

        def get_metric_columns(self) -> List[str]:
            return ['radgraph_combined']

        @property
        def name(self) -> str:
            return 'DisabledRadGraphEvaluator'
