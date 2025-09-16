class CompositeMetricEvaluator:
    """Minimal stub of CompositeMetricEvaluator for test and integration runs.

    This evaluator provides a simple, deterministic composite score so the
    modular evaluation pipeline and tests can import and exercise the
    'composite' metric without depending on heavyweight model files.
    """

    def __init__(self, study_id_col: str = 'study_id', report_col: str = 'report',
                 composite_v0_path: str = None, composite_v1_path: str = None,
                 normalizer_path: str = None):
        self.study_id_col = study_id_col
        self.report_col = report_col
        self.composite_v0_path = composite_v0_path
        self.composite_v1_path = composite_v1_path
        self.normalizer_path = normalizer_path

    def compute_metric(self, gt_df, pred_df):
        """Add a deterministic composite score column to pred_df and return it.

        The score is a simple heuristic (fraction of overlapping words) so
        it is lightweight and deterministic for tests.
        """
        import pandas as pd
        from collections import Counter

        df = pred_df.copy()

        # Ensure index alignment by study id if present
        if self.study_id_col in df.columns and self.study_id_col in gt_df.columns:
            gt_map = gt_df.set_index(self.study_id_col)[self.report_col].to_dict()
            scores = []
            for _, row in df.iterrows():
                sid = row.get(self.study_id_col)
                cand = str(row.get(self.report_col, '')).lower().split()
                ref = str(gt_map.get(sid, '')).lower().split()
                if not cand or not ref:
                    scores.append(0.0)
                    continue
                # simple overlap metric
                ref_counts = Counter(ref)
                match = sum(min(ref_counts[w], cand.count(w)) for w in set(cand))
                score = match / max(len(cand), 1)
                scores.append(score)
            df['composite_score'] = scores
        else:
            # Fallback: compute simple similarity based on token overlap position-wise
            def overlap(a, b):
                a_tokens = str(a).lower().split()
                b_tokens = str(b).lower().split()
                if not a_tokens or not b_tokens:
                    return 0.0
                common = set(a_tokens) & set(b_tokens)
                return len(common) / max(len(a_tokens), 1)

            scores = []
            for ref, cand in zip(gt_df.get(self.report_col, []), df.get(self.report_col, [])):
                scores.append(overlap(ref, cand))
            df['composite_score'] = scores

        return df

    def get_summary_stats(self, results_df):
        """Return simple summary statistics for composite_score."""
        scores = results_df.get('composite_score')
        if scores is None or len(scores) == 0:
            return {'mean_composite': None, 'num_samples': 0}
        import math
        vals = [float(x) for x in scores]
        mean = sum(vals) / len(vals)
        std = (sum((x-mean)**2 for x in vals)/len(vals))**0.5 if len(vals)>0 else 0.0
        return {'mean_composite': mean, 'std_composite': std, 'num_samples': len(vals)}

__all__ = ['CompositeMetricEvaluator']