"""
Run BERTScore evaluation using the in-repo evaluator and the sample data
located in `CXRMetric/metrics/data/metrics_test_cases.json`.

This script prints per-sample precision/recall/f1 and a summary.
"""
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure project root is importable so `CXRMetric` package can be imported
# When this file lives under CXRMetric/metrics/bertscore, the repo root is three
# levels above this file (../../..)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

DATA_PATH = os.path.join("CXRMetric", "metrics", "data", "metrics_test_cases.json")

if not os.path.exists(DATA_PATH):
    print(f"ERROR: data file not found at {DATA_PATH}")
    sys.exit(2)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data.get("bertscore_sample_reports", [])
if len(samples) == 0:
    print("No sample reports found under 'bertscore_sample_reports' in the test data.")
    sys.exit(0)

# Create dataframes expected by the evaluator
gt_rows = []
pred_rows = []
for s in samples:
    study_id = s.get("study_id")
    gt_rows.append({"study_id": int(study_id), "report": s.get("gt")})
    pred_rows.append({"study_id": int(study_id), "report": s.get("pred")})

gt_df = pd.DataFrame(gt_rows)
pred_df = pd.DataFrame(pred_rows)

# Try to import evaluator
try:
    from CXRMetric.metrics.bertscore.bertscore_metrics import BERTScoreEvaluator
except Exception as e:
    print("BERTScoreEvaluator import failed:", e)
    # Try direct bert_score import to show a minimal fallback
    try:
        from bert_score import BERTScorer
    except Exception as e2:
        print("bert_score package is not available in the environment. To run BERTScore install: pip install bert-score[tf,torch] or bert-score==0.3.13")
        sys.exit(3)
    else:
        print("bert_score is available; running a direct scorer over samples")
        refs = gt_df['report'].tolist()
        cands = pred_df['report'].tolist()
        scorer = BERTScorer(model_type='distilroberta-base', batch_size=32, lang='en', rescale_with_baseline=True)
        P, R, F = scorer.score(cands, refs)
        # Convert outputs to numpy arrays (handles torch tensors as returned by bert-score)
        def _to_numpy(x):
            try:
                import torch
                if hasattr(x, 'cpu'):
                    return x.cpu().numpy()
            except Exception:
                pass
            if isinstance(x, list):
                return np.array(x)
            return np.asarray(x)

        P_np = _to_numpy(P)
        R_np = _to_numpy(R)
        F_np = _to_numpy(F)

        for sid, p, r, f in zip(gt_df['study_id'].tolist(), P_np.tolist(), R_np.tolist(), F_np.tolist()):
            print(f"study_id={sid}: P={p:.4f} R={r:.4f} F1={f:.4f}")
        print('\nSummary:')
        print(f" mean_P={np.mean(P_np):.4f} mean_R={np.mean(R_np):.4f} mean_F1={np.mean(F_np):.4f}")
        sys.exit(0)

# Instantiate evaluator and run
try:
    evaluator = BERTScoreEvaluator(model_type='distilroberta-base', batch_size=32, use_idf=False)
except Exception as e:
    print("Failed to initialize BERTScoreEvaluator:", e)
    sys.exit(4)

try:
    result_df = evaluator.compute_metric(gt_df, pred_df)
except Exception as e:
    print("Error while computing BERTScore via evaluator:", e)
    sys.exit(5)

# Print per-sample results
print("Per-sample BERTScore results:")
for _, row in result_df.iterrows():
    sid = row['study_id']
    f1 = row.get('bertscore', None)
    p = row.get('bertscore_precision', None)
    r = row.get('bertscore_recall', None)
    print(f"study_id={sid}: P={p:.4f} R={r:.4f} F1={f1:.4f}")

# Print summary
summary = evaluator.get_summary_stats(result_df)
import json as _json
print('\nSummary statistics:')
print(_json.dumps(summary, indent=2))

print('\nDone.')

# Also write a timestamped Markdown copy of the per-sample results and summary
try:
    out_dir = os.path.join('outputs', 'metrics')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    md_path = os.path.join(out_dir, f'bertscore_summary_{ts}.md')
    with open(md_path, 'w', encoding='utf-8') as md:
        md.write(f"# BERTScore run summary ({ts})\n\n")
        md.write('## Per-sample BERTScore Results\n\n')
        md.write('| study_id | precision | recall | f1 | good_range | why |\n')
        md.write('|---|---:|---:|---:|---|---|\n')
        for _, row in result_df.iterrows():
            sid = int(row['study_id'])
            p = row.get('bertscore_precision', 0.0)
            r = row.get('bertscore_recall', 0.0)
            f = row.get('bertscore', 0.0)
            # Categorize F1 into a simple guidance range
            if f >= 0.65:
                good_range = 'Good (>=0.65)'
                explanation = 'High semantic overlap: strong matching of key content and phrasing.'
            elif f >= 0.55:
                good_range = 'Moderate (0.55-0.65)'
                explanation = 'Moderate overlap: some paraphrasing or missing details reduce exact match on phrasing.'
            else:
                good_range = 'Poor (<0.55)'
                explanation = 'Low overlap: predictions miss key content or use substantially different wording (paraphrasing).'
            # Add nuance based on precision vs recall
            diff = p - r
            if diff > 0.05:
                explanation += ' Precision > Recall → outputs are precise but may miss content.'
            elif diff < -0.05:
                explanation += ' Recall > Precision → outputs cover content but may be imprecise.'
            md.write(f"| {sid} | {p:.4f} | {r:.4f} | {f:.4f} | {good_range} | {explanation} |\n")
        md.write('\n## Summary statistics\n\n')
        # Write JSON summary block for clarity
        md.write('```json\n')
        md.write(json.dumps(summary, indent=2))
        md.write('\n```\n')
    print(f"Saved BERTScore markdown summary to: {md_path}")
except Exception as _e:
    print('Failed to save BERTScore markdown summary:', _e)
