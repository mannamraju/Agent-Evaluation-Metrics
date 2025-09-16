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

## Notes

- The `BLEUEvaluator` lives at `CXRMetric/metrics/bleu/bleu_metrics.py`.
- The consolidated tools live at `CXRMetric/metrics/bleu/bleu_tools.py`.
- For reproducible tests, generate or use consistent CSVs under `reports/` and evaluate
  with the modular runner.

## Contact / maintenance

If you change the canonical dataset format, update `data_loader.py` and the reader code
in `bleu_tools.py` accordingly.
