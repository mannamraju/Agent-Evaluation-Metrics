# BLEU evaluation (CXR report metrics)

This folder contains the BLEU evaluator implementation and a consolidated
CLI-style utility for running BLEU evaluations against the canonical
sample dataset.

## Canonical input data

- `CXRMetric/metrics/data/metrics_test_cases.json` (key: `"bleu"`)

## Primary entrypoint

Run the consolidated evaluator (loads canonical BLEU cases):

```
python -m CXRMetric.metrics.bleu.bleu_tools evaluator \
  --output outputs/metrics/bleu_metrics_results.csv
```

Run the modular runner on CSV files (ground truth and predictions):

```
python -m CXRMetric.metrics.bleu.bleu_tools modular \
  --gt reports/gt_reports.csv \
  --pred reports/predicted_reports.csv \
  --output outputs/metrics/bleu_results.csv
```

## Smoothing / experiments

Run smoothed BLEU examples (show different smoothing methods):

```
python -m CXRMetric.metrics.bleu.bleu_tools smooth --methods add_one chen_cherry
```

## Where results are written

Default outputs are under: `outputs/metrics/`

- `bleu_metrics_results.csv` — per-sample scores from the consolidated evaluator
- `bleu_results.csv` — output written by the modular runner when given CSVs

## Recommended workflow

1. Keep your canonical sample/test data updated in `CXRMetric/metrics/data/metrics_test_cases.json`.
2. Run the consolidated evaluator for quick checks (`evaluator`).
3. Use the modular runner when you have GT/pred CSVs from model outputs.
4. If BLEU-4 scores are all zero (common for short clinical text), use the `smooth` command
   to inspect smoothing methods and consider enabling smoothing in the evaluator.

## Implementation details

- NLTK vs fallback
  - The module includes a compact Python fallback BLEU implementation used when
    `nltk` is not installed. The fallback is easy to audit and suitable for
    demos and small tests, but it short-circuits to zero when any higher-order
    n-gram precision is zero (so BLEU-4 often becomes 0 for short or paraphrased
    clinical sentences).
  - `nltk`'s `sentence_bleu` is the canonical implementation used by this
    repository. NLTK provides a well-tested BLEU implementation with support for
    multiple references and provided smoothing functions (via
    `nltk.translate.bleu_score.SmoothingFunction`). Use NLTK + a smoothing
    function to obtain more stable, non-zero BLEU-4 aggregates for clinical text.

- Smoothing and its purpose
  - Medical reports are often short and paraphrased; exact 3- or 4-gram
    overlaps are therefore rare. Smoothing prevents BLEU-4 from becoming
    identically zero by substituting small positive values for missing
    higher-order n-gram precisions.
  - This repository implements and exposes several smoothing strategies:
    - `epsilon`     — substitute a tiny epsilon (e.g., 1e-7) so log precision is defined
    - `add_one`     — Laplace-style add-one smoothing on n-gram counts
    - `chen_cherry`  — heuristic scaling used in some BLEU smoothing literature
  - The improved BLEU utilities (`bleu_impl.py`) return a `smoothing_applied`
    flag per-sample and a run-level `smoothing_applied_count` so you can
    quantify how often smoothing was necessary. A high smoothing rate suggests
    BLEU-4 is sensitive to surface-level variations for your dataset.

- Practical recommendations
  - For quick demos and unit tests, the fallback is convenient and deterministic.
  - For aggregate evaluation on clinical reports use NLTK with a smoothing
    function or the consolidated smoothed BLEU utilities provided in
    `bleu_impl.py` to avoid overly pessimistic BLEU-4 statistics.
  - For production-scale evaluation prefer `fast-bleu` (if available) or run
    the smoothed implementations in batch to generate stable summary statistics.

## Files and purposes

- `bleu_evaluation_summary.py` — legacy compatibility shim that previously packaged evaluation-summary helpers; preserved historically but main summary/logging logic has been consolidated into `bleu_impl.py`.
- `bleu_impl.py` — the consolidated, authoritative BLEU implementation: tokenization, n-gram utilities, smoothed BLEU-4 computation, evaluator runner, logging, and strictness analysis. This is the primary implementation for improved/smoothed BLEU workflows.
- `bleu_metrics.py` — the package evaluator class (`BLEUEvaluator`) used by the modular evaluation framework. It adapts available implementations (fast-bleu, NLTK, or the repo fallback) into the `BaseEvaluator` interface used by the rest of the repo.
- `bleu_strictness_demo.py` — small interactive/demo script that demonstrates the strictness of BLEU-4 versus BLEU-2 on curated clinical examples and on the canonical `metrics_test_cases.json` data.
- `bleu_tools.py` — lightweight backward-compatible entrypoints and convenience wrappers that re-export or delegate to consolidated runner functions; these allow older CLI calls to continue working while the codebase migrates to `run_bleu_demo.py` and `bleu_impl.py`.
- `bleu4_comparison.py` — utility script that compares the original (legacy) BLEU implementation with the improved/smoothed BLEU-4 implementations to quantify differences and recommend analysis targets for medical text.
- `improved_bleu4.py` — compatibility shim that re-exports the improved BLEU functions (e.g., `compute_smoothed_bleu4`, `evaluate_medical_reports_bleu4`) from `bleu_impl.py` so older import sites remain valid during the consolidation.

## Notes

- The `BLEUEvaluator` lives at `CXRMetric/metrics/bleu/bleu_metrics.py`.
- The consolidated tools live at `CXRMetric/metrics/bleu/bleu_tools.py`.
- For reproducible tests, generate or use consistent CSVs under `reports/` and evaluate
  with the modular runner.

## Contact / maintenance

If you change the canonical dataset format, update `data_loader.py` and the reader code
in `bleu_tools.py` accordingly.
