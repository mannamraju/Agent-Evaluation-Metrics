"""
Run ROUGE-L evaluation using the in-repo ROUGEEvaluator and the sample data
located in `CXRMetric/metrics/data/metrics_test_cases.json`.

This script prints per-sample ROUGE-L scores and a summary.
"""
import json
import os
import sys

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

# Build dataframes
import pandas as pd

gt_rows = []
pred_rows = []
for s in samples:
    study_id = s.get("study_id")
    gt_rows.append({"study_id": int(study_id), "report": s.get("gt")})
    pred_rows.append({"study_id": int(study_id), "report": s.get("pred")})

gt_df = pd.DataFrame(gt_rows)
pred_df = pd.DataFrame(pred_rows)

# Make project importable
# When this file is under CXRMetric/metrics/bleu, the repo root is three levels up (../../..)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import evaluator
try:
    from CXRMetric.metrics.rouge.rouge_metrics import ROUGEEvaluator
except Exception as e:
    print("Failed to import ROUGEEvaluator:", e)
    sys.exit(3)

# Run the evaluator
evaluator = ROUGEEvaluator(beta=1.2)
result_df = evaluator.compute_metric(gt_df, pred_df)

print("Per-sample ROUGE-L results:")
for _, row in result_df.iterrows():
    sid = row['study_id']
    r = row.get('rouge_l', None)
    print(f"study_id={sid}: ROUGE-L={r:.4f}")

# Summary
summary = evaluator.get_summary_stats(result_df)
import json as _json
print('\nSummary statistics:')
print(_json.dumps(summary, indent=2))

print('\nDone.')
