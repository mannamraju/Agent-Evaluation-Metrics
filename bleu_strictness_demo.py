#!/usr/bin/env python3
"""
BLEU-4 vs BLEU-2 Comparison Test

Demonstrates why BLEU-4 is much stricter than BLEU-2 for medical reports.
"""

import pandas as pd
from collections import Counter
import math

def tokenize_simple(text):
    """Simple tokenization for BLEU calculation."""
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

def show_ngram_details(gt_text, pred_text, n):
    """Show detailed n-gram analysis."""
    gt_tokens = tokenize_simple(gt_text)
    pred_tokens = tokenize_simple(pred_text)
    
    gt_ngrams = get_ngrams(gt_tokens, n)
    pred_ngrams = get_ngrams(pred_tokens, n)
    
    print(f"    {n}-grams in GT:   {list(gt_ngrams.keys())}")
    print(f"    {n}-grams in Pred: {list(pred_ngrams.keys())}")
    
    # Find overlaps
    overlaps = []
    for ng in pred_ngrams:
        if ng in gt_ngrams:
            overlaps.append(ng)
    
    print(f"    Overlapping {n}-grams: {overlaps}")
    overlap_count = sum(min(pred_ngrams[ng], gt_ngrams[ng]) for ng in overlaps)
    total_pred = sum(pred_ngrams.values())
    precision = overlap_count / total_pred if total_pred > 0 else 0
    print(f"    {n}-gram precision: {overlap_count}/{total_pred} = {precision:.4f}")
    
    return precision

def test_bleu_strictness():
    """Demonstrate BLEU-4 strictness."""
    
    print("🎯 BLEU-4 Strictness Demonstration")
    print("=" * 50)
    
    # Test case that shows the difference clearly
    gt = "No acute cardiopulmonary abnormalities are identified."
    pred = "No acute abnormalities are identified."
    
    print("📝 Test Case:")
    print(f"  GT:   '{gt}'")
    print(f"  Pred: '{pred}'")
    print(f"  Difference: Missing 'cardiopulmonary'")
    
    bleu2 = compute_bleu_score(gt, pred, max_n=2)
    bleu4 = compute_bleu_score(gt, pred, max_n=4)
    
    print(f"\n📊 BLEU Scores:")
    print(f"  BLEU-2: {bleu2:.4f}")
    print(f"  BLEU-4: {bleu4:.4f}")
    print(f"  Ratio:  {bleu2/bleu4 if bleu4 > 0 else 'inf'}x stricter")
    
    print(f"\n🔍 Detailed N-gram Analysis:")
    
    # 2-gram analysis
    print(f"\n  📋 2-gram Analysis:")
    p2 = show_ngram_details(gt, pred, 2)
    
    # 4-gram analysis
    print(f"\n  📋 4-gram Analysis:")
    p4 = show_ngram_details(gt, pred, 4)
    
    print(f"\n💡 Why BLEU-4 is Zero:")
    gt_tokens = tokenize_simple(gt)
    pred_tokens = tokenize_simple(pred)
    
    print(f"  • GT has {len(gt_tokens)} tokens, Pred has {len(pred_tokens)} tokens")
    print(f"  • GT 4-grams: {len(gt_tokens)-3} possible")
    print(f"  • Pred 4-grams: {len(pred_tokens)-3} possible")
    print(f"  • Missing 'cardiopulmonary' breaks all 4-gram matches")
    print(f"  • BLEU-4 requires exact 4-token sequences")
    
    # Show the exact 4-grams
    gt_4grams = get_ngrams(gt_tokens, 4)
    pred_4grams = get_ngrams(pred_tokens, 4)
    
    print(f"\n  📝 Exact 4-grams:")
    for i, (gt_4g, pred_4g) in enumerate(zip(gt_4grams, pred_4grams)):
        print(f"    Position {i+1}: {gt_4g} vs {pred_4g}")
        if gt_4g != pred_4g:
            print(f"      ❌ Mismatch!")
        else:
            print(f"      ✅ Match!")
    
    print(f"\n📈 Medical Report Implications:")
    print(f"  • Medical reports often paraphrase conditions")
    print(f"  • BLEU-4 punishes clinical synonym usage")
    print(f"  • BLEU-2 captures content similarity better")
    print(f"  • Both metrics provide complementary insights")
    
    # Test with a case that has good BLEU-4
    print(f"\n" + "="*50)
    print(f"🎯 Counter-example: Good BLEU-4 Score")
    
    gt2 = "The heart size is within normal limits."
    pred2 = "The heart size is within normal limits and shape."
    
    print(f"  GT:   '{gt2}'")
    print(f"  Pred: '{pred2}'")
    
    bleu2_good = compute_bleu_score(gt2, pred2, max_n=2)
    bleu4_good = compute_bleu_score(gt2, pred2, max_n=4)
    
    print(f"  BLEU-2: {bleu2_good:.4f}")
    print(f"  BLEU-4: {bleu4_good:.4f}")
    print(f"  💡 Both scores are high due to long exact matches")

if __name__ == "__main__":
    test_bleu_strictness()
    print("\n✅ BLEU strictness demonstration completed!")
