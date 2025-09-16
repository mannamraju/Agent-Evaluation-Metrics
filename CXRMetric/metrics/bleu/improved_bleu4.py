#!/usr/bin/env python3
"""
Improved BLEU-4 Implementation for Medical Text

This implementation includes smoothing techniques specifically designed
for short medical texts where exact 4-gram matches are rare.
"""

import math
import re
from typing import List, Dict, Tuple
from collections import Counter
from pathlib import Path
import json

def tokenize_medical(text: str) -> List[str]:
    """Medical-aware tokenization that preserves clinical terms."""
    # Convert to lowercase but preserve medical abbreviations
    text = text.lower()
    
    # Use regex to split on word boundaries, preserving hyphens in medical terms
    tokens = re.findall(r'\b[\w-]+\b', text)
    
    return tokens

def get_ngrams_with_padding(tokens: List[str], n: int, padding: bool = True) -> Counter:
    """Extract n-grams with optional sentence boundary padding."""
    if len(tokens) == 0:
        return Counter()
    
    if padding and n > 1:
        # Add sentence boundary markers for better n-gram coverage
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

def compute_smoothed_bleu4(reference: str, candidate: str, 
                          smoothing_method: str = 'epsilon') -> Dict[str, float]:
    """
    Compute BLEU-4 with various smoothing methods for medical text.
    
    Args:
        reference: Reference text
        candidate: Candidate text  
        smoothing_method: 'epsilon', 'add_one', or 'chen_cherry'
    
    Returns:
        Dictionary with BLEU scores and component metrics
    """
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
    
    # Calculate precision for n-grams 1 through 4
    precisions = []
    smoothing_applied = False
    
    for n in range(1, 5):
        ref_ngrams = get_ngrams_with_padding(ref_tokens, n, padding=(n > 1))
        cand_ngrams = get_ngrams_with_padding(cand_tokens, n, padding=(n > 1))
        
        if not cand_ngrams:
            if smoothing_method == 'epsilon':
                precisions.append(1e-7)  # Small epsilon to avoid log(0)
                smoothing_applied = True
            elif smoothing_method == 'add_one':
                precisions.append(1.0 / (len(cand_tokens) + 1))
                smoothing_applied = True
            else:
                precisions.append(0.0)
            continue
        
        # Count matches
        matches = 0
        total_cand_ngrams = sum(cand_ngrams.values())
        
        for ngram in cand_ngrams:
            matches += min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0))
        
        precision = matches / total_cand_ngrams
        
        # Apply smoothing if precision is 0 and we're looking at higher n-grams
        if precision == 0.0 and n >= 3:
            if smoothing_method == 'epsilon':
                precision = 1e-7
                smoothing_applied = True
            elif smoothing_method == 'add_one':
                precision = 1.0 / (total_cand_ngrams + 1)
                smoothing_applied = True
            elif smoothing_method == 'chen_cherry':
                # Chen & Cherry (2014) smoothing
                precision = 1.0 / (2 ** n)
                smoothing_applied = True
        
        precisions.append(precision)
    
    # Brevity penalty
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    
    if cand_len > ref_len:
        bp = 1.0
    elif cand_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len / cand_len)
    
    # Calculate BLEU scores for different n-gram levels
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

def evaluate_medical_reports_bleu4(reference_reports: List[str], 
                                  candidate_reports: List[str],
                                  smoothing_method: str = 'epsilon') -> Dict:
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
        bleu4_scores.append(result['bleu4'])
        bleu2_scores.append(result['bleu2'])
    
    # Calculate aggregate statistics
    mean_bleu4 = sum(bleu4_scores) / len(bleu4_scores)
    mean_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
    
    # Standard deviation
    std_bleu4 = math.sqrt(sum((x - mean_bleu4)**2 for x in bleu4_scores) / len(bleu4_scores))
    std_bleu2 = math.sqrt(sum((x - mean_bleu2)**2 for x in bleu2_scores) / len(bleu2_scores))
    
    # Quality distribution analysis
    excellent_bleu4 = sum(1 for s in bleu4_scores if s >= 0.30)
    good_bleu4 = sum(1 for s in bleu4_scores if 0.20 <= s < 0.30)
    fair_bleu4 = sum(1 for s in bleu4_scores if 0.10 <= s < 0.20)
    poor_bleu4 = sum(1 for s in bleu4_scores if 0.05 <= s < 0.10)
    very_poor_bleu4 = sum(1 for s in bleu4_scores if s < 0.05)
    
    # Smoothing analysis
    smoothing_count = sum(1 for r in all_results if r['smoothing_applied'])
    
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

# Test with your current medical examples
if __name__ == "__main__":
    print("🏥 Improved BLEU-4 for Medical Reports")
    print("=" * 50)
    
    # Load test cases for improved BLEU from metrics data
    data_dir = Path(__file__).resolve()
    while data_dir.name != 'metrics' and data_dir.parent != data_dir:
        data_dir = data_dir.parent
    from CXRMetric.metrics.data_loader import load_consolidated
    data = load_consolidated()
    test_cases = data['bleu_improved']
    
    references = [case['reference'] for case in test_cases]
    candidates = [case['candidate'] for case in test_cases]
    
    # Test different smoothing methods
    for method in ['epsilon', 'add_one', 'chen_cherry']:
        print(f"\n📊 Results with {method} smoothing:")
        print("-" * 40)
        
        results = evaluate_medical_reports_bleu4(references, candidates, method)
        
        print(f"Mean BLEU-4: {results['mean_bleu4']:.4f} ± {results['std_bleu4']:.4f}")
        print(f"Mean BLEU-2: {results['mean_bleu2']:.4f} ± {results['std_bleu2']:.4f}")
        print(f"Smoothing applied: {results['smoothing_applied_count']}/{results['num_samples']} cases")
        
        dist = results['quality_distribution']
        print(f"Quality: Excellent:{dist['excellent']} Good:{dist['good']} Fair:{dist['fair']} Poor:{dist['poor']} VeryPoor:{dist['very_poor']}")
        
        # Show individual scores for first few examples
        print("\nIndividual BLEU-4 scores:")
        for i, score in enumerate(results['bleu4_scores'][:3]):
            print(f"  Example {i+1}: {score:.4f}")
