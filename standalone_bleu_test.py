#!/usr/bin/env python3
"""
Standalone BLEU-4 Test

Direct implementation test without package imports.
"""

import pandas as pd
from collections import Counter
import math

def tokenize_simple(text):
    """Simple tokenization for BLEU calculation."""
    # Convert to lowercase and split on whitespace/punctuation
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def get_ngrams(tokens, n):
    """Extract n-grams from tokens."""
    if len(tokens) < n:
        return Counter()
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return Counter(ngrams)

def compute_bleu_score(reference, candidate, max_n=4):
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

def test_bleu_standalone():
    """Test BLEU implementation standalone."""
    
    print("🧪 Standalone BLEU-4 Test")
    print("=" * 40)
    
    test_cases = [
        {
            'gt': "The heart is normal in size.",
            'pred': "The heart is normal in size.",
            'label': 'Perfect match'
        },
        {
            'gt': "The heart is normal in size and shape.",
            'pred': "Heart is normal in size and shape.",
            'label': 'Near perfect (missing "The")'
        },
        {
            'gt': "No acute cardiopulmonary abnormalities are identified.",
            'pred': "No acute abnormalities identified.",
            'label': 'Good match (missing words)'
        },
        {
            'gt': "The lungs are clear bilaterally without infiltrate.",
            'pred': "Chest X-ray shows normal findings.",
            'label': 'Different phrasing'
        }
    ]
    
    print("\n📊 BLEU Score Comparison:")
    print("-" * 70)
    print(f"{'Case':<30} {'BLEU-2':<10} {'BLEU-4':<10} {'Ratio':<10}")
    print("-" * 70)
    
    bleu2_scores = []
    bleu4_scores = []
    
    for case in test_cases:
        bleu2 = compute_bleu_score(case['gt'], case['pred'], max_n=2)
        bleu4 = compute_bleu_score(case['gt'], case['pred'], max_n=4)
        ratio = bleu2 / bleu4 if bleu4 > 0 else float('inf')
        
        bleu2_scores.append(bleu2)
        bleu4_scores.append(bleu4)
        
        ratio_str = f"{ratio:.2f}x" if ratio != float('inf') else "∞"
        print(f"{case['label']:<30} {bleu2:<10.4f} {bleu4:<10.4f} {ratio_str:<10}")
        
        # Show detailed analysis
        print(f"  GT:   '{case['gt']}'")
        print(f"  Pred: '{case['pred']}'")
        
        # Show n-gram analysis
        gt_tokens = tokenize_simple(case['gt'])
        pred_tokens = tokenize_simple(case['pred'])
        
        gt_2grams = get_ngrams(gt_tokens, 2)
        pred_2grams = get_ngrams(pred_tokens, 2)
        gt_4grams = get_ngrams(gt_tokens, 4)
        pred_4grams = get_ngrams(pred_tokens, 4)
        
        # Count overlaps
        overlap_2 = sum(min(pred_2grams[ng], gt_2grams.get(ng, 0)) for ng in pred_2grams)
        overlap_4 = sum(min(pred_4grams[ng], gt_4grams.get(ng, 0)) for ng in pred_4grams)
        
        total_2 = sum(pred_2grams.values())
        total_4 = sum(pred_4grams.values())
        
        print(f"  2-gram overlaps: {overlap_2}/{total_2}")
        print(f"  4-gram overlaps: {overlap_4}/{total_4}")
        print()
    
    # Summary
    print(f"📈 Summary:")
    print(f"  Mean BLEU-2: {sum(bleu2_scores)/len(bleu2_scores):.4f}")
    print(f"  Mean BLEU-4: {sum(bleu4_scores)/len(bleu4_scores):.4f}")
    valid_ratios = [b2/b4 for b2, b4 in zip(bleu2_scores, bleu4_scores) if b4 > 0]
    if valid_ratios:
        print(f"  Mean Ratio:  {sum(valid_ratios)/len(valid_ratios):.2f}x")
    
    print("\n💡 Key Insights:")
    print("• BLEU-4 requires longer exact matches")
    print("• Medical reports often paraphrase, reducing 4-gram overlap")
    print("• BLEU-2 is more forgiving for content similarity")
    print("• Both metrics are useful for different evaluation aspects")
    
    return bleu2_scores, bleu4_scores

if __name__ == "__main__":
    print("This demo was moved to tools/bleu_demo.py. Run: python -m tools.bleu_demo")
