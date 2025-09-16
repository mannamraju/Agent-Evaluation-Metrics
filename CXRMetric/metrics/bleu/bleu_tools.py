"""
Consolidated BLEU tools for evaluation and experiments.

This module centralizes the small utility scripts previously spread across
/tools into a single module under the canonical package so you can run:

  python -m CXRMetric.metrics.bleu.bleu_tools evaluator
  python -m CXRMetric.metrics.bleu.bleu_tools modular
  python -m CXRMetric.metrics.bleu.bleu_tools smooth --method add_one

It uses the canonical consolidated dataset located at
CXRMetric/metrics/data/metrics_test_cases.json.
"""

from pathlib import Path
import argparse
import os
import pandas as pd
from typing import List

from CXRMetric.metrics.data_loader import load_consolidated
from CXRMetric.metrics.bleu.bleu_metrics import BLEUEvaluator
try:
    from CXRMetric.metrics.bleu.improved_bleu4 import compute_smoothed_bleu4
except Exception:
    # Best-effort import; smoothing utility lives in the bleu package
    def compute_smoothed_bleu4(a, b, smoothing_method='epsilon'):
        return {'bleu4': 0.0, 'bleu2': 0.0, 'bleu1': 0.0, 'smoothing_applied': False}

from CXRMetric.modular_evaluation import evaluate_reports


def _build_dfs_from_bleu_cases(bleu_cases: List[dict]):
    rows_gt = []
    rows_pred = []
    for i, c in enumerate(bleu_cases, start=1):
        sid = c.get('study_id', str(i))
        rows_gt.append({'study_id': sid, 'report': c['reference']})
        rows_pred.append({'study_id': sid, 'report': c['candidate']})
    gt_df = pd.DataFrame(rows_gt)
    pred_df = pd.DataFrame(rows_pred)
    return gt_df, pred_df


def run_bleu_evaluator_consolidated(output_csv: str = 'outputs/metrics/bleu_metrics_results.csv'):
    data = load_consolidated()
    bleu_cases = data.get('bleu', [])

    gt_df, pred_df = _build_dfs_from_bleu_cases(bleu_cases)

    evaluator = BLEUEvaluator(compute_bleu2=True, compute_bleu4=True, study_id_col='study_id', report_col='report')
    results_df = evaluator.compute_metric(gt_df, pred_df)

    out_dir = Path(output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    summary = evaluator.get_summary_stats(results_df)
    print(f"Saved per-sample BLEU results to {output_csv}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return results_df, summary


def run_modular_on_reports(gt_csv: str = 'reports/gt_reports.csv', pred_csv: str = 'reports/predicted_reports.csv', output_csv: str = 'outputs/metrics/bleu_results.csv'):
    os.makedirs(Path(output_csv).parent, exist_ok=True)
    results, summary = evaluate_reports(gt_csv, pred_csv, metrics=['bleu'], output_csv=output_csv, use_cache=False)
    print(f"Saved modular BLEU results to {output_csv}")
    return results, summary


def run_smoothed_examples(methods: List[str] = None):
    if methods is None:
        methods = ['epsilon', 'add_one', 'chen_cherry']

    # Small curated examples
    examples = [
        ('No acute cardiopulmonary abnormalities are identified.', 'No acute abnormalities are identified.'),
        ('The heart is normal in size and shape.', 'Heart is normal in size and shape.'),
        ('Bilateral lower lobe consolidation present.', 'Consolidation is present in bilateral lower lobes.')
    ]

    out = {}
    for method in methods:
        scores = []
        for ref, cand in examples:
            res = compute_smoothed_bleu4(ref, cand, smoothing_method=method)
            scores.append(res)
            print(f"Method={method} | Ref={ref[:40]}... | Cand={cand[:40]}... | BLEU-4={res['bleu4']:.4f}")
        out[method] = scores
    return out


def main():
    parser = argparse.ArgumentParser(prog='bleu_tools')
    sub = parser.add_subparsers(dest='cmd')

    parser_eval = sub.add_parser('evaluator', help='Run BLEU evaluator on consolidated test cases')
    parser_eval.add_argument('--output', '-o', default='outputs/metrics/bleu_metrics_results.csv')

    parser_mod = sub.add_parser('modular', help='Run the modular runner using CSV reports')
    parser_mod.add_argument('--gt', default='reports/gt_reports.csv')
    parser_mod.add_argument('--pred', default='reports/predicted_reports.csv')
    parser_mod.add_argument('--output', '-o', default='outputs/metrics/bleu_results.csv')

    parser_smooth = sub.add_parser('smooth', help='Run smoothed BLEU examples')
    parser_smooth.add_argument('--methods', '-m', nargs='*', default=['epsilon', 'add_one', 'chen_cherry'])

    args = parser.parse_args()
    if args.cmd == 'evaluator':
        run_bleu_evaluator_consolidated(args.output)
    elif args.cmd == 'modular':
        run_modular_on_reports(args.gt, args.pred, args.output)
    elif args.cmd == 'smooth':
        run_smoothed_examples(args.methods)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
