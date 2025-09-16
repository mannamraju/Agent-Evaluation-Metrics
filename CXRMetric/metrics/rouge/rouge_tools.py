"""
Consolidated ROUGE tools for evaluation and experiments.

Provides a small CLI to run ROUGE-L evaluation on the canonical
consolidated dataset under CXRMetric/metrics/data or to run the
modular evaluator on CSV files.
"""
from pathlib import Path
import os
import pandas as pd
import argparse
from typing import List

from CXRMetric.metrics.data_loader import load_consolidated
from CXRMetric.metrics.rouge.rouge_metrics import ROUGEEvaluator
from CXRMetric.modular_evaluation import evaluate_reports


def _build_dfs_from_rouge_cases(rouge_cases: List[dict]):
    rows_gt = []
    rows_pred = []
    for i, c in enumerate(rouge_cases, start=1):
        sid = c.get('study_id', str(i))
        rows_gt.append({'study_id': sid, 'report': c['reference']})
        rows_pred.append({'study_id': sid, 'report': c['generated']})
    gt_df = pd.DataFrame(rows_gt)
    pred_df = pd.DataFrame(rows_pred)
    return gt_df, pred_df


def run_rouge_evaluator_consolidated(output_csv: str = 'outputs/metrics/rouge_metrics_results.csv', beta: float = 1.2):
    data = load_consolidated()
    rouge_cases = data.get('rouge', [])

    gt_df, pred_df = _build_dfs_from_rouge_cases(rouge_cases)

    evaluator = ROUGEEvaluator(beta=beta, study_id_col='study_id', report_col='report')
    results_df = evaluator.compute_metric(gt_df, pred_df)

    out_dir = Path(output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    summary = evaluator.get_summary_stats(results_df)
    print(f"Saved per-sample ROUGE results to {output_csv}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return results_df, summary


def run_modular_on_reports(gt_csv: str = 'reports/gt_reports.csv', pred_csv: str = 'reports/predicted_reports.csv', output_csv: str = 'outputs/metrics/rouge_results.csv', beta: float = 1.2):
    os.makedirs(Path(output_csv).parent, exist_ok=True)
    results, summary = evaluate_reports(gt_csv, pred_csv, metrics=['rouge'], output_csv=output_csv, use_cache=False)
    print(f"Saved modular ROUGE results to {output_csv}")
    return results, summary


def main():
    parser = argparse.ArgumentParser(prog='rouge_tools')
    sub = parser.add_subparsers(dest='cmd')

    parser_eval = sub.add_parser('evaluator', help='Run ROUGE evaluator on consolidated test cases')
    parser_eval.add_argument('--output', '-o', default='outputs/metrics/rouge_metrics_results.csv')
    parser_eval.add_argument('--beta', type=float, default=1.2)

    parser_mod = sub.add_parser('modular', help='Run the modular runner using CSV reports')
    parser_mod.add_argument('--gt', default='reports/gt_reports.csv')
    parser_mod.add_argument('--pred', default='reports/predicted_reports.csv')
    parser_mod.add_argument('--output', '-o', default='outputs/metrics/rouge_results.csv')

    args = parser.parse_args()
    if args.cmd == 'evaluator':
        run_rouge_evaluator_consolidated(args.output, beta=args.beta)
    elif args.cmd == 'modular':
        run_modular_on_reports(args.gt, args.pred, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
