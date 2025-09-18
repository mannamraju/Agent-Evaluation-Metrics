"""
Comprehensive BLEU demo runner.

This single entrypoint consolidates the functionality from:
 - bleu_evaluation_summary.py  (evaluation + logging + strictness analysis)
 - bleu_tools.py               (evaluator runner, modular runner, smoothing examples)
 - improved_bleu4.py          (improved BLEU-4 implementation and evaluation)

Run as a script to execute a full BLEU demonstration across the canonical
consolidated test cases in CXRMetric/metrics/data/metrics_test_cases.json.

Examples:
  python run_bleu_demo.py            # runs a full consolidated demo
  python run_bleu_demo.py --part evaluator
  python run_bleu_demo.py --part improved --smoothing chen_cherry

"""
import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Ensure repo root is importable when executed directly
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from CXRMetric.metrics.bleu.bleu_impl import (
    run_bleu_evaluator_consolidated,
    run_smoothed_examples,
    evaluate_medical_reports_bleu4,
    compute_smoothed_bleu4,
    run_bleu_evaluation_with_logging,
    analyze_bleu_strictness,
    display_bleu_evaluation_history,
)
_BLEU_IMPL_AVAILABLE = True
_BLEU_IMPL_IMPORT_ERROR = None


def run_all(smoothing_methods=None, improved_smoothing=None, output_csv=None):
    """Run a full BLEU demonstration: evaluator, smoothing examples, improved BLEU, strictness analysis."""
    results_cache = {}

    # 1) Run evaluator on consolidated cases (if available)
    print("\n=== Running BLEU evaluator (consolidated test cases) ===")
    try:
        out = run_bleu_evaluator_consolidated(output_csv or 'outputs/metrics/bleu_metrics_results.csv')
        # results_df can be large/non-serializable; store a summary and path
        results_cache['evaluator'] = {
            'output_csv': output_csv or 'outputs/metrics/bleu_metrics_results.csv',
            'summary': out[1] if isinstance(out, tuple) and len(out) > 1 else None
        }
    except Exception as e:
        print('  • Evaluator failed:', e)
        results_cache['evaluator'] = {'error': str(e)}

    # 2) Run smoothing example methods from bleu_tools (best-effort)
    print("\n=== BLEU smoothing examples ===")
    try:
        smooth_out = run_smoothed_examples(smoothing_methods)
        results_cache['smoothing_examples'] = smooth_out
    except Exception as e:
        print('  • Smoothing examples failed:', e)
        results_cache['smoothing_examples'] = {'error': str(e)}

    # 3) Run improved BLEU-4 evaluations if the improved module is present
    print("\n=== Improved BLEU-4 evaluations ===")
    improved_results = {}
    try:
        from CXRMetric.metrics.data_loader import load_consolidated
        data = load_consolidated()
        bleu_improved_cases = data.get('bleu_improved', [])
        refs = [c['reference'] for c in bleu_improved_cases]
        cands = [c['candidate'] for c in bleu_improved_cases]
    except Exception as ex:
        print('  • Failed to load improved BLEU test cases:', ex)
        refs, cands = [], []

    methods = [improved_smoothing] if improved_smoothing else ['epsilon', 'add_one', 'chen_cherry']
    for m in methods:
        if not refs:
            improved_results[m] = {'error': 'no test cases available'}
            continue
        try:
            r = evaluate_medical_reports_bleu4(refs, cands, smoothing_method=m)
            improved_results[m] = r
        except Exception as ex:
            improved_results[m] = {'error': str(ex)}

    results_cache['improved_bleu4'] = improved_results

    # 4) Strictness analysis and logging
    print("\n=== BLEU evaluation summary & strictness analysis ===")
    try:
        summary_results, log_entry = run_bleu_evaluation_with_logging()
        results_cache['summary'] = {
            'results_summary': summary_results,
            'log_entry': log_entry
        }
    except Exception as ex:
        print('  • Warning: run_bleu_evaluation_with_logging failed:', ex)
        results_cache['summary'] = {'error': str(ex)}

    try:
        strict_r = analyze_bleu_strictness()
        results_cache['strictness'] = strict_r
    except Exception as ex:
        print('  • Warning: analyze_bleu_strictness failed:', ex)
        results_cache['strictness'] = {'error': str(ex)}

    print("\n✅ Run complete. Collected results for parts:", ', '.join(results_cache.keys()))
    return results_cache


def main():
    parser = argparse.ArgumentParser(prog='run_bleu_demo', description='Comprehensive BLEU demo runner')
    # --part: choose which part of the demo to run. Options:
    #   'all'        : run the entire demo sequence (evaluator, smoothing examples,
    #                  improved BLEU evaluations, strictness analysis, and history).
    #   'evaluator'  : run only the consolidated evaluator over canonical test cases
    #                  and write per-sample scores to a CSV (see --output).
    #   'smoothing'  : run only the smoothing example harness that demonstrates
    #                  different smoothing strategies on short medical examples.
    #   'improved'   : run only the improved BLEU-4 evaluation (applies smoothing
    #                  strategies across the 'bleu_improved' test cases).
    #   'strictness' : run only the strictness analysis which compares BLEU-2 vs
    #                  BLEU-4 across the strictness test cases and logs results.
    #   'history'    : display recent evaluation history saved to the package
    #                  summary file (`bleu_evaluation_summary.json`).
    parser.add_argument('--part', choices=['all', 'evaluator', 'smoothing', 'improved', 'strictness', 'history'], default='all', help='Which demo part to run')

    # --output / -o: path to write evaluator CSV results. When omitted the
    # default is `outputs/metrics/bleu_metrics_results.csv` relative to the
    # repository root. The evaluator writes a per-sample CSV with columns such
    # as `study_id`, `bleu_score`, and `bleu4_score`.
    parser.add_argument('--output', '-o', help='Output CSV path for evaluator')

    # --smoothing-method / -s: select a single smoothing method when running the
    # 'improved' part. Supported methods are: 'epsilon', 'add_one',
    # 'chen_cherry'. If not specified, the demo will run all supported methods.
    parser.add_argument('--smoothing-method', '-s', help='Single smoothing method for improved BLEU (epsilon|add_one|chen_cherry)')

    # --smoothing-methods / -m: provide a list of smoothing methods to exercise
    # when running the 'smoothing' examples. Example: `-m epsilon add_one`.
    # If omitted, the demo will exercise the default set of smoothing methods.
    parser.add_argument('--smoothing-methods', '-m', nargs='*', help='Smoothing methods to exercise for examples (bleu_tools)')

    # --summary-json: when specified, the demo will re-run the minimal set of
    # selected parts to collect results and write a machine-readable JSON file
    # that contains a timestamp and a 'summary' object with per-part outputs
    # (e.g. evaluator summary, smoothing examples, improved BLEU stats). Use
    # this for automated experiment logging or CI artifact collection.
    parser.add_argument('--summary-json', help='Write a machine-readable JSON summary of the demo run to this path')

    args = parser.parse_args()

    if args.part in ('all', 'evaluator'):
        if run_bleu_evaluator_consolidated is None:
            print("ERROR: evaluator functionality unavailable (bleu_impl missing)")
        else:
            run_bleu_evaluator_consolidated(args.output or 'outputs/metrics/bleu_metrics_results.csv')

    if args.part in ('all', 'smoothing'):
        if run_smoothed_examples is None:
            print("ERROR: smoothing examples unavailable (bleu_impl missing)")
        else:
            run_smoothed_examples(args.smoothing_methods)

    if args.part in ('all', 'improved'):
        if evaluate_medical_reports_bleu4 is None:
            print("ERROR: improved_bleu4 unavailable (bleu_impl missing)")
        else:
            # Load test cases
            try:
                from CXRMetric.metrics.data_loader import load_consolidated
                data = load_consolidated()
                bleu_improved_cases = data.get('bleu_improved', [])
                refs = [c['reference'] for c in bleu_improved_cases]
                cands = [c['candidate'] for c in bleu_improved_cases]
            except Exception as ex:
                print("Failed to load improved BLEU test cases:", ex)
                refs, cands = [], []

            method = args.smoothing_method or 'epsilon'
            if refs:
                r = evaluate_medical_reports_bleu4(refs, cands, smoothing_method=method)
                print(f"Improved BLEU result summary for method={method}:")
                print(f"  Mean BLEU-4: {r['mean_bleu4']:.4f} ± {r['std_bleu4']:.4f}")

    if args.part in ('all', 'strictness'):
        if analyze_bleu_strictness is None:
            print("ERROR: strictness analysis unavailable (bleu_impl missing)")
        else:
            analyze_bleu_strictness()

    if args.part in ('all', 'history'):
        logs = display_bleu_evaluation_history() if display_bleu_evaluation_history else []
        if logs:
            print('\nRecent BLEU evaluation history:')
            for i, l in enumerate(logs, 1):
                print(f" {i}. {l.get('timestamp')}: mean_bleu2={l.get('summary_stats', {}).get('mean_bleu2')}")

    # If requested, write a JSON summary of the work we just executed
    if args.summary_json:
        print(f"\nWriting demo summary to {args.summary_json} ...")
        try:
            # Determine which parts were run and re-run minimal data collection for JSON
            summary = run_all(smoothing_methods=args.smoothing_methods, improved_smoothing=args.smoothing_method, output_csv=args.output)
            with open(args.summary_json, 'w', encoding='utf-8') as fh:
                json.dump({'timestamp': datetime.now().isoformat(), 'summary': summary}, fh, indent=2)
            print(f"Wrote summary to {args.summary_json}")
        except Exception as ex:
            print('Failed to write summary JSON:', ex)


if __name__ == '__main__':
    main()
