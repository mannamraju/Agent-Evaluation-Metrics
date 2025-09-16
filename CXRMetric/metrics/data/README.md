Metrics sample data directory

This folder contains canonical sample data used by the metric evaluation
and test scripts in this repository.

Canonical file:
- metrics_test_cases.json — authoritative consolidated dataset. Keys include:
  - bleu
  - bleu_strictness
  - bleu_improved
  - bleu_comparison
  - bertscore
  - bertscore_comparison
  - bertscore_sample_reports
  - perplexity
  - rouge
  - rouge_unit

How to use:
- Import the loader:
    from CXRMetric.metrics.data_loader import load_metric_cases
- Load cases:
    bleu_cases = load_metric_cases('bleu')

Compatibility notes:
- Per-metric JSON files remain in this folder for backwards compatibility, but
  are deprecated and now contain a small pointer to the consolidated file.
- If you add or update sample cases, update metrics_test_cases.json and
  avoid duplicating data across per-metric files.
