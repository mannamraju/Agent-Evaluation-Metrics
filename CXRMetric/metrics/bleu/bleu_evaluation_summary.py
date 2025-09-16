#!/usr/bin/env python3
"""
BLEU Evaluation Summary Logger

This script runs BLEU evaluation and logs timestamped results
to track performance over time and compare BLEU-2 vs BLEU-4.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import Counter
import math
import re

# Add project root to Python path if running directly
if __name__ == "__main__":
    project_root = Path(__file__).parents[4]
    sys.path.insert(0, str(project_root))

def tokenize_simple(text: str) -> List[str]:
    """Simple tokenization for BLEU calculation."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from tokens."""
    if len(tokens) < n:
        return Counter()
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)

def compute_bleu_score(reference: str, candidate: str, max_n: int = 4) -> float:
    """Compute BLEU score for given texts."""
    ref_tokens = tokenize_simple(reference)
    cand_tokens = tokenize_simple(candidate)
    
    if not cand_tokens:
        return 0.0
    
    # Compute precision for each n-gram level
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(ref_tokens, n)
        cand_ngrams = get_ngrams(cand_tokens, n)
        
        if not cand_ngrams:
            precisions.append(0.0)
            continue
            
        # Count matches
        matches = 0
        for ngram in cand_ngrams:
            matches += min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0))
        
        precision = matches / sum(cand_ngrams.values())
        precisions.append(precision)
    
    # Brevity penalty
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    log_precision_sum = sum(math.log(p) for p in precisions)
    geometric_mean = math.exp(log_precision_sum / len(precisions))
    
    return bp * geometric_mean

def log_bleu_evaluation(results: Dict[str, Any], config: Dict[str, Any] = None):
    """Log BLEU evaluation results with timestamp.
    
    Args:
        results: Dictionary containing evaluation results
        config: Configuration used for the evaluation
    """
    
    # Get the bleu folder path
    bleu_folder = Path(__file__).parent.parent
    summary_file = bleu_folder / "bleu_evaluation_summary.json"
    
    # Create timestamp
    timestamp = datetime.now().isoformat()
    
    # Prepare log entry
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
    
    # Load existing logs or create new list
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Add new entry
    logs.append(log_entry)
    
    # Save updated logs
    with open(summary_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"✅ Logged BLEU evaluation to {summary_file}")
    return log_entry

def run_bleu_evaluation_with_logging():
    """Run BLEU evaluation and log results."""
    
    print("📊 BLEU Evaluation with Logging")
    print("=" * 50)
    
    # Configuration
    config = {
        "bleu_variants": ["bleu2", "bleu4"],
        "implementation": "custom_fallback",
        "tokenization": "simple_regex"
    }
    
    # Load sample radiology reports from metrics data directory
    data_dir = Path(__file__).resolve()
    while data_dir.name != 'metrics' and data_dir.parent != data_dir:
        data_dir = data_dir.parent
    from CXRMetric.metrics.data_loader import load_consolidated
    data = load_consolidated()
    test_cases = data['bleu']
    
    print(f"📊 Evaluating {len(test_cases)} report pairs")
    print(f"⚙️  Computing both BLEU-2 and BLEU-4 scores")
    
    # Calculate BLEU scores
    bleu2_scores = []
    bleu4_scores = []
    
    for case in test_cases:
        bleu2 = compute_bleu_score(case['reference'], case['candidate'], max_n=2)
        bleu4 = compute_bleu_score(case['reference'], case['candidate'], max_n=4)
        
        bleu2_scores.append(bleu2)
        bleu4_scores.append(bleu4)
    
    # Calculate statistics
    mean_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
    mean_bleu4 = sum(bleu4_scores) / len(bleu4_scores)
    
    std_bleu2 = math.sqrt(sum((x - mean_bleu2)**2 for x in bleu2_scores) / len(bleu2_scores))
    std_bleu4 = math.sqrt(sum((x - mean_bleu4)**2 for x in bleu4_scores) / len(bleu4_scores))
    
    # Quality thresholds
    high_quality_bleu2 = sum(1 for score in bleu2_scores if score > 0.3)
    high_quality_bleu4 = sum(1 for score in bleu4_scores if score > 0.1)
    
    # BLEU-4 to BLEU-2 ratio
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
    
    # Display results
    print(f"\n📈 BLEU Results:")
    print(f"  Mean BLEU-2: {mean_bleu2:.4f} ± {std_bleu2:.4f}")
    print(f"  Mean BLEU-4: {mean_bleu4:.4f} ± {std_bleu4:.4f}")
    print(f"  BLEU-2/BLEU-4 Ratio: {mean_ratio:.2f}x (BLEU-4 is stricter)")
    
    print(f"\n📊 Quality Distribution:")
    print(f"  High BLEU-2 (>0.3): {high_quality_bleu2}/{len(test_cases)} ({100*high_quality_bleu2/len(test_cases):.1f}%)")
    print(f"  High BLEU-4 (>0.1): {high_quality_bleu4}/{len(test_cases)} ({100*high_quality_bleu4/len(test_cases):.1f}%)")
    print(f"  Zero BLEU-4 scores: {results['zero_bleu4_count']}/{len(test_cases)} ({100*results['zero_bleu4_count']/len(test_cases):.1f}%)")
    
    # Show some examples
    print(f"\n📝 Sample Comparisons:")
    print("-" * 60)
    
    # Sort by BLEU-4 score for examples
    sorted_indices = sorted(range(len(bleu4_scores)), key=lambda i: bleu4_scores[i], reverse=True)
    
    print("🏆 Best BLEU-4 Score:")
    best_idx = sorted_indices[0]
    print(f"  BLEU-2: {bleu2_scores[best_idx]:.4f}, BLEU-4: {bleu4_scores[best_idx]:.4f}")
    print(f"  Ref: {test_cases[best_idx]['reference'][:60]}...")
    print(f"  Can: {test_cases[best_idx]['candidate'][:60]}...")
    
    print("\n📉 Worst BLEU-4 Score:")
    worst_idx = sorted_indices[-1]
    print(f"  BLEU-2: {bleu2_scores[worst_idx]:.4f}, BLEU-4: {bleu4_scores[worst_idx]:.4f}")
    print(f"  Ref: {test_cases[worst_idx]['reference'][:60]}...")
    print(f"  Can: {test_cases[worst_idx]['candidate'][:60]}...")
    
    # Log results
    log_entry = log_bleu_evaluation(results, config)
    
    return results, log_entry

def display_bleu_evaluation_history():
    """Display historical BLEU evaluation results."""
    
    bleu_folder = Path(__file__).parent.parent
    summary_file = bleu_folder / "bleu_evaluation_summary.json"
    
    if not summary_file.exists():
        print("📝 No BLEU evaluation history found")
        return
    
    with open(summary_file, 'r') as f:
        logs = json.load(f)
    
    print(f"\n📚 BLEU Evaluation History ({len(logs)} entries)")
    print("=" * 60)
    
    for i, log in enumerate(logs[-5:], 1):  # Show last 5 entries
        timestamp = log['timestamp']
        stats = log['summary_stats']
        
        print(f"\n{i}. {timestamp}")
        print(f"   BLEU-2: {stats.get('mean_bleu2', 0):.4f}")
        print(f"   BLEU-4: {stats.get('mean_bleu4', 0):.4f}")
        print(f"   Ratio: {stats.get('bleu4_bleu2_ratio', 0):.2f}x")
        print(f"   Samples: {stats.get('num_samples', 0)}")

def analyze_bleu_strictness():
    """Analyze BLEU-4 strictness with controlled examples."""
    
    print(f"\n🎯 BLEU Strictness Analysis")
    print("=" * 50)
    
    # Load strictness analysis cases from metrics data
    data_dir = Path(__file__).resolve()
    while data_dir.name != 'metrics' and data_dir.parent != data_dir:
        data_dir = data_dir.parent
    from CXRMetric.metrics.data_loader import load_consolidated
    data = load_consolidated()
    strictness_cases = data['bleu_strictness']
    
    strictness_results = []
    
    for case in strictness_cases:
        bleu2 = compute_bleu_score(case['reference'], case['candidate'], max_n=2)
        bleu4 = compute_bleu_score(case['reference'], case['candidate'], max_n=4)
        
        result = {
            'description': case['description'],
            'reference': case['reference'],
            'candidate': case['candidate'],
            'bleu2': bleu2,
            'bleu4': bleu4,
            'impact': 'High' if bleu4 == 0 else 'Medium' if bleu4 < bleu2/2 else 'Low'
        }
        strictness_results.append(result)
        
        print(f"\n• {case['description']}:")
        print(f"  BLEU-2: {bleu2:.4f}")
        print(f"  BLEU-4: {bleu4:.4f}")
        print(f"  Impact: {result['impact']}")
    
    # Log strictness analysis
    analysis_results = {
        "analysis_type": "strictness_analysis",
        "test_cases": strictness_results
    }
    
    log_bleu_evaluation(analysis_results, {"evaluation_type": "strictness_analysis"})
    
    return strictness_results

if __name__ == "__main__":
    print("📊 BLEU Evaluation Summary & Logging")
    print("=" * 60)
    
    # Run standard evaluation
    results, log_entry = run_bleu_evaluation_with_logging()
    
    # Show evaluation history
    display_bleu_evaluation_history()
    
    # Analyze BLEU strictness
    analyze_bleu_strictness()
    
    print(f"\n" + "="*60)
    if results:
        print("✅ BLEU evaluation completed and logged!")
        print(f"📊 Mean BLEU-2: {results['mean_bleu2']:.4f}")
        print(f"📊 Mean BLEU-4: {results['mean_bleu4']:.4f}")
        print("📁 Results saved to bleu_evaluation_summary.json")
    else:
        print("⚠️  BLEU evaluation completed with limitations")
    
    print("\n💡 Summary Features:")
    print("• Timestamped BLEU evaluation logs")
    print("• BLEU-2 vs BLEU-4 comparison tracking")
    print("• Strictness analysis with medical examples")
    print("• Performance trends over time")
