"""
Consolidated BLEU implementation utilities extracted from legacy modules.

This module centralizes the core BLEU scoring, smoothing utilities,
consolidated evaluator runner, summary logging, and strictness analysis.
It is intended as the single authoritative implementation that new
runners (e.g. `run_bleu_demo.py`) can import.
"""
from pathlib import Path
from collections import Counter
import math
import re
import json
from datetime import datetime
from typing import List, Dict, Any

import os
import pandas as pd

# Utilities for tokenization and n-gram extraction (both simple and medical-aware)

def tokenize_simple(text: str) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def get_ngrams(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)


# Medical-aware tokenization and n-gram with padding

def tokenize_medical(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"\b[\w-]+\b", text)
    return tokens


def get_ngrams_with_padding(tokens: List[str], n: int, padding: bool = True) -> Counter:
    if len(tokens) == 0:
        return Counter()
    if padding and n > 1:
        padded_tokens = ['<s>'] * (n-1) + tokens + ['</s>'] * (n-1)
    else:
        padded_tokens = tokens
    if len(padded_tokens) < n:
        return Counter()
    ngrams = []
    for i in range(len(padded_tokens) - n + 1):
        ngram = tuple(padded_tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)


# Basic BLEU computation (used by summary/logging)

def compute_bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
    ref_tokens = tokenize_simple(reference)
    cand_tokens = tokenize_simple(candidate)

    if not cand_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(ref_tokens, n)
        cand_ngrams = get_ngrams(cand_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        matches = 0
        for ngram in cand_ngrams:
            matches += min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0))
        precision = matches / sum(cand_ngrams.values())
        precisions.append(precision)

    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0

    if any(p == 0 for p in precisions):
        return 0.0
    log_precision_sum = sum(math.log(p) for p in precisions)
    geometric_mean = math.exp(log_precision_sum / len(precisions))
    return bp * geometric_mean


# Improved BLEU-4 with smoothing

def compute_smoothed_bleu4(reference: str, candidate: str, smoothing_method: str = 'epsilon') -> Dict[str, Any]:
    ref_tokens = tokenize_medical(reference)
    cand_tokens = tokenize_medical(candidate)

    if not cand_tokens:
        return {
            'bleu4': 0.0,
            'bleu2': 0.0,
            'bleu1': 0.0,
            'brevity_penalty': 0.0,
            'precision_scores': [0.0, 0.0, 0.0, 0.0],
            'smoothing_applied': True
        }

    precisions = []
    smoothing_applied = False
    for n in range(1, 5):
        ref_ngrams = get_ngrams_with_padding(ref_tokens, n, padding=(n > 1))
        cand_ngrams = get_ngrams_with_padding(cand_tokens, n, padding=(n > 1))
        if not cand_ngrams:
            if smoothing_method == 'epsilon':
                precisions.append(1e-7)
                smoothing_applied = True
            elif smoothing_method == 'add_one':
                precisions.append(1.0 / (len(cand_tokens) + 1))
                smoothing_applied = True
            else:
                precisions.append(0.0)
            continue
        matches = 0
        total_cand_ngrams = sum(cand_ngrams.values())
        for ngram in cand_ngrams:
            matches += min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0))
        precision = matches / total_cand_ngrams
        if precision == 0.0 and n >= 3:
            if smoothing_method == 'epsilon':
                precision = 1e-7
                smoothing_applied = True
            elif smoothing_method == 'add_one':
                precision = 1.0 / (total_cand_ngrams + 1)
                smoothing_applied = True
            elif smoothing_method == 'chen_cherry':
                precision = 1.0 / (2 ** n)
                smoothing_applied = True
        precisions.append(precision)

    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    if cand_len > ref_len:
        bp = 1.0
    elif cand_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len / cand_len)

    def geometric_mean(scores):
        if any(p <= 0 for p in scores):
            return 0.0
        log_sum = sum(math.log(p) for p in scores)
        return math.exp(log_sum / len(scores))

    bleu1 = bp * precisions[0] if precisions[0] > 0 else 0.0
    bleu2 = bp * geometric_mean(precisions[:2]) if all(p > 0 for p in precisions[:2]) else 0.0
    bleu4 = bp * geometric_mean(precisions) if all(p > 0 for p in precisions) else 0.0

    return {
        'bleu4': bleu4,
        'bleu2': bleu2,
        'bleu1': bleu1,
        'brevity_penalty': bp,
        'precision_scores': precisions,
        'smoothing_applied': smoothing_applied,
        'ref_length': ref_len,
        'cand_length': cand_len
    }


# High-level evaluation utilities that use existing in-repo evaluators

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
    try:
        from CXRMetric.metrics.data_loader import load_consolidated
        from CXRMetric.metrics.bleu.bleu_metrics import BLEUEvaluator
    except Exception as e:
        raise ImportError('Required modules for evaluator are unavailable') from e

    data = load_consolidated()
    bleu_cases = data.get('bleu', [])
    gt_df, pred_df = _build_dfs_from_bleu_cases(bleu_cases)

    evaluator = BLEUEvaluator(compute_bleu2=True, compute_bleu4=True, study_id_col='study_id', report_col='report')
    # Report which implementation is active (NLTK vs fallback). This confirms
    # for users that the NLTK-based BLEU implementation is being used when
    # available and otherwise the repo fallback will be applied silently.
    try:
        impl_msg = "NLTK" if evaluator._nltk_available else "fallback"
    except Exception:
        impl_msg = "fallback"
    print(f"Using BLEU implementation: {impl_msg}")

    results_df = evaluator.compute_metric(gt_df, pred_df)

    out_dir = Path(output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    summary = evaluator.get_summary_stats(results_df)
    return results_df, summary


def run_smoothed_examples(methods: List[str] = None):
    if methods is None:
        methods = ['epsilon', 'add_one', 'chen_cherry']

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
        out[method] = scores
    return out


# Summary logging and strictness analysis (port of previous summary module)

def log_bleu_evaluation(results: Dict[str, Any], config: Dict[str, Any] = None):
    bleu_folder = Path(__file__).parent
    summary_file = bleu_folder / "bleu_evaluation_summary.json"
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "evaluation_type": "bleu",
        "configuration": config or {},
        "results": results,
        "summary_stats": {
            "mean_bleu2": results.get("mean_bleu2", None),
            "mean_bleu4": results.get("mean_bleu4", None),
            "std_bleu2": results.get("std_bleu2", None),
            "std_bleu4": results.get("std_bleu4", None),
            "num_samples": results.get("num_samples", None),
            "bleu4_bleu2_ratio": results.get("bleu4_bleu2_ratio", None),
            "high_quality_bleu2_count": results.get("high_quality_bleu2_count", None),
            "high_quality_bleu4_count": results.get("high_quality_bleu4_count", None)
        }
    }
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_entry)
    with open(summary_file, 'w') as f:
        json.dump(logs, f, indent=2)
    return log_entry


def run_bleu_evaluation_with_logging():
    try:
        from CXRMetric.metrics.data_loader import load_consolidated
    except Exception as e:
        raise ImportError('data_loader not available') from e

    config = {
        "bleu_variants": ["bleu2", "bleu4"],
        "implementation": "consolidated_bleu_impl",
        "tokenization": "simple_regex"
    }

    data = load_consolidated()
    test_cases = data['bleu']

    bleu2_scores = []
    bleu4_scores = []
    for case in test_cases:
        bleu2 = compute_bleu_score(case['reference'], case['candidate'], max_n=2)
        bleu4 = compute_bleu_score(case['reference'], case['candidate'], max_n=4)
        bleu2_scores.append(bleu2)
        bleu4_scores.append(bleu4)

    mean_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
    mean_bleu4 = sum(bleu4_scores) / len(bleu4_scores)
    std_bleu2 = math.sqrt(sum((x - mean_bleu2)**2 for x in bleu2_scores) / len(bleu2_scores))
    std_bleu4 = math.sqrt(sum((x - mean_bleu4)**2 for x in bleu4_scores) / len(bleu4_scores))

    high_quality_bleu2 = sum(1 for score in bleu2_scores if score > 0.3)
    high_quality_bleu4 = sum(1 for score in bleu4_scores if score > 0.1)

    valid_ratios = [b2/b4 for b2, b4 in zip(bleu2_scores, bleu4_scores) if b4 > 0]
    mean_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else float('inf')

    results = {
        "mean_bleu2": mean_bleu2,
        "mean_bleu4": mean_bleu4,
        "std_bleu2": std_bleu2,
        "std_bleu4": std_bleu4,
        "num_samples": len(test_cases),
        "individual_bleu2_scores": bleu2_scores,
        "individual_bleu4_scores": bleu4_scores,
        "high_quality_bleu2_count": high_quality_bleu2,
        "high_quality_bleu4_count": high_quality_bleu4,
        "bleu4_bleu2_ratio": mean_ratio,
        "zero_bleu4_count": sum(1 for score in bleu4_scores if score == 0)
    }
    # Add human-readable interpretation to the results so CLI runs can
    # explain what the central metrics mean and why a None may appear.
    results['interpretation'] = {
        'mean_bleu_explanation': (
            'Mean BLEU-X is the arithmetic mean of per-sample BLEU-X scores. '
            'It summarizes average n-gram overlap between candidate and reference '
            'reports. A value near 0 indicates few exact n-gram matches; values '
            'closer to 1 indicate high overlap.'
        ),
        'none_reason': (
            'If a mean is reported as None it indicates that no per-sample '
            'scores were available (for example, the evaluator was skipped, '
            'dataset had no matching samples, or an error occurred during '
            'computation). Ensure the BLEU evaluator ran successfully and '
            'that the input cases are present.'
        )
    }

    log_entry = log_bleu_evaluation(results, config)
    return results, log_entry


def analyze_bleu_strictness():
    try:
        from CXRMetric.metrics.data_loader import load_consolidated
    except Exception as e:
        raise ImportError('data_loader not available') from e

    data = load_consolidated()
    strictness_cases = data['bleu_strictness']
    strictness_results = []
    for case in strictness_cases:
        bleu2 = compute_bleu_score(case['reference'], case['candidate'], max_n=2)
        bleu4 = compute_bleu_score(case['reference'], case['candidate'], max_n=4)
        result = {
            'description': case.get('description', ''),
            'reference': case['reference'],
            'candidate': case['candidate'],
            'bleu2': bleu2,
            'bleu4': bleu4,
            'impact': 'High' if bleu4 == 0 else 'Medium' if bleu4 < bleu2/2 else 'Low'
        }
        strictness_results.append(result)
    log_bleu_evaluation({'analysis_type': 'strictness_analysis', 'test_cases': strictness_results}, {'evaluation_type': 'strictness_analysis'})
    return strictness_results


def display_bleu_evaluation_history(limit: int = 5):
    bleu_folder = Path(__file__).parent
    summary_file = bleu_folder / "bleu_evaluation_summary.json"
    if not summary_file.exists():
        return []
    with open(summary_file, 'r') as f:
        logs = json.load(f)
    return logs[-limit:]


def evaluate_medical_reports_bleu4(reference_reports: List[str], 
                                  candidate_reports: List[str],
                                  smoothing_method: str = 'epsilon') -> Dict[str, Any]:
    """
    Evaluate multiple medical reports with improved BLEU-4.
    Returns comprehensive metrics and analysis.
    """
    if len(reference_reports) != len(candidate_reports):
        raise ValueError("Reference and candidate lists must have same length")

    all_results = []
    bleu4_scores = []
    bleu2_scores = []

    for ref, cand in zip(reference_reports, candidate_reports):
        result = compute_smoothed_bleu4(ref, cand, smoothing_method)
        all_results.append(result)
        bleu4_scores.append(result.get('bleu4', 0.0))
        bleu2_scores.append(result.get('bleu2', 0.0))

    # Calculate aggregate statistics
    mean_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0
    mean_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0.0

    # Standard deviation
    std_bleu4 = math.sqrt(sum((x - mean_bleu4)**2 for x in bleu4_scores) / len(bleu4_scores)) if bleu4_scores else 0.0
    std_bleu2 = math.sqrt(sum((x - mean_bleu2)**2 for x in bleu2_scores) / len(bleu2_scores)) if bleu2_scores else 0.0

    # Quality distribution analysis
    excellent_bleu4 = sum(1 for s in bleu4_scores if s >= 0.30)
    good_bleu4 = sum(1 for s in bleu4_scores if 0.20 <= s < 0.30)
    fair_bleu4 = sum(1 for s in bleu4_scores if 0.10 <= s < 0.20)
    poor_bleu4 = sum(1 for s in bleu4_scores if 0.05 <= s < 0.10)
    very_poor_bleu4 = sum(1 for s in bleu4_scores if s < 0.05)

    # Smoothing analysis:
    # - 'smoothing_applied' is set when a smoothing rule was used to avoid zero precision
    #   for higher-order n-grams (e.g., when no 3- or 4-gram matches exist).
    # - Counting how often smoothing is applied indicates how frequently strict
    #   n-gram matching would yield zero BLEU-4 scores for medical reports; a high
    #   count implies many cases are too short or too different to form exact 4-grams.
    # - Common smoothing methods implemented in this module:
    #     * 'epsilon'     : insert a tiny epsilon (e.g. 1e-7) to avoid log(0)
    #     * 'add_one'     : Laplace-style add-one smoothing on n-gram counts
    #     * 'chen_cherry'  : heuristic scaling often used for BLEU smoothing
    # - Smoothing makes BLEU-4 comparisons more stable (non-zero but small scores)
    #   and allows aggregate statistics (means/std) to be computed without being
    #   dominated by exact-match sparsity. The 'smoothing_method' field in the
    #   returned dictionary records which technique was used for the run.

    smoothing_count = sum(1 for r in all_results if r.get('smoothing_applied'))

    return {
        'mean_bleu4': mean_bleu4,
        'mean_bleu2': mean_bleu2,
        'std_bleu4': std_bleu4,
        'std_bleu2': std_bleu2,
        'individual_results': all_results,
        'bleu4_scores': bleu4_scores,
        'bleu2_scores': bleu2_scores,
        'quality_distribution': {
            'excellent': excellent_bleu4,
            'good': good_bleu4,
            'fair': fair_bleu4,
            'poor': poor_bleu4,
            'very_poor': very_poor_bleu4
        },
        'smoothing_applied_count': smoothing_count,
        'num_samples': len(reference_reports),
        'smoothing_method': smoothing_method
    }
