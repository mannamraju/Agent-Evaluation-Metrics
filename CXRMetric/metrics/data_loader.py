"""
Centralized data loader for sample metric test cases.

Provides a single API to load the consolidated sample data JSON file
located at CXRMetric/metrics/data/metrics_test_cases.json.

This module is robust to being imported from different working directories
and provides a fallback for direct script execution.
"""

from pathlib import Path
import json


def _find_metrics_dir() -> Path:
    """Return the absolute path to the nearest 'metrics' directory.

    Starting from this file's location, walk up parents until a directory
    named 'metrics' is found. If not found, return this file's parent.
    """
    p = Path(__file__).resolve()
    # This module lives inside the metrics package; parent should be metrics
    metrics_dir = p.parent
    if metrics_dir.name == 'metrics':
        return metrics_dir

    # Fallback: walk up until 'metrics' is found
    while metrics_dir.parent != metrics_dir and metrics_dir.name != 'metrics':
        metrics_dir = metrics_dir.parent
    return metrics_dir


def _consolidated_path() -> Path:
    metrics_dir = _find_metrics_dir()
    return metrics_dir / 'data' / 'metrics_test_cases.json'


def load_consolidated() -> dict:
    """Load and return the consolidated metrics_test_cases.json content.

    Raises FileNotFoundError if the canonical data file is missing. The
    project assumes a single canonical input data file; failing fast makes
    missing-data problems visible early in CI or local runs.
    """
    path = _consolidated_path()
    if not path.exists():
        raise FileNotFoundError(f"Canonical metrics data file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_metric_cases(key: str):
    """Return the list of test cases for a given metric key from the consolidated file.

    Example keys: 'bleu', 'rouge', 'bertscore', 'perplexity', 'bleu_strictness', 'rouge_unit'
    """
    data = load_consolidated()
    return data.get(key, [])
