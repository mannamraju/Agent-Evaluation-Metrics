#!/usr/bin/env python3
"""
BLEU-4 Comparison: Original vs Improved Implementation
"""

import sys
from pathlib import Path
import json

# Add the project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3]))

from improved_bleu4 import evaluate_medical_reports_bleu4, compute_smoothed_bleu4

def compare_bleu_implementations():
    """Compare original vs improved BLEU-4 implementations."""
    
    print("🔬 BLEU-4 Implementation Comparison")
    print("=" * 60)
    
    # Load comparison test cases from metrics data
    data_dir = Path(__file__).resolve()
    while data_dir.name != 'metrics' and data_dir.parent != data_dir:
        data_dir = data_dir.parent
    from CXRMetric.metrics.data_loader import load_consolidated
    data = load_consolidated()
    test_cases = data['bleu_comparison']
    
    print(f"📊 Testing {len(test_cases)} medical report pairs\n")
    
    # Import original compute_bleu_score function
    try:
        from bleu_evaluation_summary import compute_bleu_score as original_bleu
        original_available = True
    except ImportError:
        print("⚠️  Original BLEU implementation not available for comparison")
        original_available = False
    
    for i, case in enumerate(test_cases, 1):
        print(f"🔍 Example {i}: {case['description']}")
        print(f"   Reference: {case['reference'][:50]}...")
        print(f"   Candidate: {case['candidate'][:50]}...")
        
        # Original implementation (if available)
        if original_available:
            orig_bleu4 = original_bleu(case['reference'], case['candidate'], max_n=4)
            orig_bleu2 = original_bleu(case['reference'], case['candidate'], max_n=2)
            print(f"   Original:  BLEU-2: {orig_bleu2:.4f}, BLEU-4: {orig_bleu4:.4f}")
        
        # Improved implementation with different smoothing
        for method in ['add_one', 'chen_cherry']:
            result = compute_smoothed_bleu4(case['reference'], case['candidate'], method)
            smoothed = "✓" if result['smoothing_applied'] else "✗"
            print(f"   {method:10}: BLEU-2: {result['bleu2']:.4f}, BLEU-4: {result['bleu4']:.4f} (smoothed: {smoothed})")
        
        print()
    
    # Overall comparison
    references = [case['reference'] for case in test_cases]
    candidates = [case['candidate'] for case in test_cases]
    
    print("\n📈 Overall Performance Comparison:")
    print("-" * 50)
    
    for method in ['add_one', 'chen_cherry']:
        results = evaluate_medical_reports_bleu4(references, candidates, method)
        print(f"{method:12}: Mean BLEU-4: {results['mean_bleu4']:.4f}")
        print(f"{'':12}  Non-zero scores: {sum(1 for s in results['bleu4_scores'] if s > 0.001)}/{len(results['bleu4_scores'])}")
    
    if original_available:
        orig_scores = [original_bleu(ref, cand, max_n=4) for ref, cand in zip(references, candidates)]
        orig_mean = sum(orig_scores) / len(orig_scores)
        print(f"{'Original':12}: Mean BLEU-4: {orig_mean:.4f}")
        print(f"{'':12}  Non-zero scores: {sum(1 for s in orig_scores if s > 0.001)}/{len(orig_scores)}")

def recommend_bleu4_targets():
    """Provide BLEU-4 target recommendations for medical text."""
    
    print("\n🎯 BLEU-4 Target Recommendations for Medical Reports")
    print("=" * 60)
    
    print("📊 Performance Tiers:")
    print("  🏆 Excellent (≥0.25):  High-quality paraphrasing with strong semantic preservation")
    print("  ✅ Good (0.15-0.24):   Acceptable clinical accuracy with reasonable linguistic overlap")
    print("  ⚠️  Fair (0.10-0.14):   Minimal but meaningful overlap, may need clinical review")
    print("  ❌ Poor (<0.10):       Low overlap, likely inadequate for clinical use")
    
    print("\n🏥 Medical Text Considerations:")
    print("  • Short reports (10-50 words) make BLEU-4 inherently challenging")
    print("  • Medical terminology synonyms may not share n-grams")
    print("  • Clinical accuracy matters more than exact wording")
    print("  • Smoothing is essential for meaningful BLEU-4 scores")
    
    print("\n🛠️  Recommended Implementation:")
    print("  • Use 'add_one' or 'chen_cherry' smoothing")
    print("  • Target mean BLEU-4 ≥ 0.15 for good performance")
    print("  • Accept ≥60% of reports scoring >0.10")
    print("  • Combine with clinical accuracy metrics (RadGraph, etc.)")
    
    print("\n📋 Quality Control Thresholds:")
    print("  • Individual report BLEU-4 < 0.05: Flag for manual review")
    print("  • System mean BLEU-4 < 0.10: Investigate model/data issues")
    print("  • >80% zero scores: Implementation problem likely")

if __name__ == "__main__":
    compare_bleu_implementations()
    recommend_bleu4_targets()
