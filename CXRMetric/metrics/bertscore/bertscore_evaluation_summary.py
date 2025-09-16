#!/usr/bin/env python3
"""
BERTScore Evaluation Summary Logger

This script runs BERTScore evaluation and logs timestamped results
to track performance over time and compare different configurations.
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to Python path if running directly
if __name__ == "__main__":
    project_root = Path(__file__).parents[4]
    sys.path.insert(0, str(project_root))

def log_bertscore_evaluation(results: Dict[str, Any], config: Dict[str, Any] = None):
    """Log BERTScore evaluation results with timestamp.
    
    Args:
        results: Dictionary containing evaluation results
        config: Configuration used for the evaluation
    """
    
    # Get the bertscore folder path
    bertscore_folder = Path(__file__).parent.parent
    summary_file = bertscore_folder / "bertscore_evaluation_summary.json"
    
    # Create timestamp
    timestamp = datetime.now().isoformat()
    
    # Prepare log entry
    log_entry = {
        "timestamp": timestamp,
        "evaluation_type": "bertscore",
        "configuration": config or {},
        "results": results,
        "summary_stats": {
            "mean_f1": results.get("mean_f1", None),
            "mean_precision": results.get("mean_precision", None),
            "mean_recall": results.get("mean_recall", None),
            "std_f1": results.get("std_f1", None),
            "num_samples": results.get("num_samples", None),
            "high_similarity_count": results.get("high_similarity_count", None),
            "medium_similarity_count": results.get("medium_similarity_count", None),
            "low_similarity_count": results.get("low_similarity_count", None)
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
    
    print(f"✅ Logged BERTScore evaluation to {summary_file}")
    return log_entry

def run_bertscore_evaluation_with_logging():
    """Run BERTScore evaluation and log results."""
    
    print("🤖 BERTScore Evaluation with Logging")
    print("=" * 50)
    
    try:
        from bert_score import BERTScorer
        
        # Configuration
        config = {
            "model_type": "distilroberta-base",
            "batch_size": 32,
            "lang": "en",
            "rescale_with_baseline": True,
            "use_idf": False
        }
        
        # Sample radiology reports
        references = [
            "The chest X-ray shows clear lungs bilaterally with no acute cardiopulmonary abnormalities.",
            "Heart size is within normal limits. No pleural effusion or pneumothorax identified.",
            "There is mild cardiomegaly. No acute pulmonary edema.",
            "Lungs demonstrate hyperinflation consistent with COPD. No pneumonia.",
            "Small right pleural effusion noted. Otherwise unremarkable.",
            "No acute cardiopulmonary abnormalities identified on chest radiograph.",
            "Bilateral lower lobe atelectasis present. No consolidation.",
            "Normal cardiac silhouette and clear lung fields bilaterally."
        ]
        
        candidates = [
            "Bilateral lungs are clear without acute cardiopulmonary findings.",
            "Normal cardiac silhouette. No effusion present.",
            "Heart is mildly enlarged. No signs of pulmonary edema.",
            "Hyperinflated lungs suggesting COPD. No infiltrates present.",
            "Minor right-sided pleural fluid. Rest of study normal.",
            "No acute abnormalities seen on chest X-ray.",
            "Atelectasis in both lower lobes. No infiltrate identified.",
            "Heart and lungs appear normal on radiograph."
        ]
        
        print(f"📊 Evaluating {len(references)} report pairs")
        print(f"⚙️  Configuration: {config['model_type']}")
        
        # Initialize scorer
        scorer = BERTScorer(**config)
        
        # Compute scores
        precision, recall, f1 = scorer.score(candidates, references)
        
        # Calculate statistics
        results = {
            "mean_f1": float(f1.mean()),
            "mean_precision": float(precision.mean()),
            "mean_recall": float(recall.mean()),
            "std_f1": float(f1.std()),
            "num_samples": len(f1),
            "individual_f1_scores": [float(score) for score in f1],
            "individual_precision_scores": [float(score) for score in precision],
            "individual_recall_scores": [float(score) for score in recall]
        }
        
        # Score distribution
        high_similarity = (f1 > 0.8).sum().item()
        medium_similarity = ((f1 >= 0.6) & (f1 <= 0.8)).sum().item()
        low_similarity = (f1 < 0.6).sum().item()
        
        results.update({
            "high_similarity_count": high_similarity,
            "medium_similarity_count": medium_similarity,
            "low_similarity_count": low_similarity,
            "high_similarity_pct": 100 * high_similarity / len(f1),
            "medium_similarity_pct": 100 * medium_similarity / len(f1),
            "low_similarity_pct": 100 * low_similarity / len(f1)
        })
        
        # Display results
        print(f"\n📈 BERTScore Results:")
        print(f"  Mean F1:        {results['mean_f1']:.4f}")
        print(f"  Mean Precision: {results['mean_precision']:.4f}")
        print(f"  Mean Recall:    {results['mean_recall']:.4f}")
        print(f"  Std F1:         {results['std_f1']:.4f}")
        
        print(f"\n📊 Score Distribution:")
        print(f"  High (>0.8):     {high_similarity:2d} ({results['high_similarity_pct']:5.1f}%)")
        print(f"  Medium (0.6-0.8): {medium_similarity:2d} ({results['medium_similarity_pct']:5.1f}%)")
        print(f"  Low (<0.6):      {low_similarity:2d} ({results['low_similarity_pct']:5.1f}%)")
        
        # Log results
        log_entry = log_bertscore_evaluation(results, config)
        
        return results, log_entry
        
    except ImportError as e:
        print(f"❌ BERTScore package not available: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None, None

def display_evaluation_history():
    """Display historical BERTScore evaluation results."""
    
    bertscore_folder = Path(__file__).parent.parent
    summary_file = bertscore_folder / "bertscore_evaluation_summary.json"
    
    if not summary_file.exists():
        print("📝 No evaluation history found")
        return
    
    with open(summary_file, 'r') as f:
        logs = json.load(f)
    
    print(f"\n📚 BERTScore Evaluation History ({len(logs)} entries)")
    print("=" * 60)
    
    for i, log in enumerate(logs[-5:], 1):  # Show last 5 entries
        timestamp = log['timestamp']
        stats = log['summary_stats']
        config = log['configuration']
        
        print(f"\n{i}. {timestamp}")
        print(f"   Model: {config.get('model_type', 'N/A')}")
        print(f"   Mean F1: {stats.get('mean_f1', 0):.4f}")
        print(f"   Samples: {stats.get('num_samples', 0)}")
        
        high_count = stats.get('high_similarity_count', 0)
        total = stats.get('num_samples', 1)
        print(f"   High Quality: {high_count}/{total} ({100*high_count/total:.1f}%)")

def compare_bertscore_models():
    """Compare BERTScore performance across different models."""
    
    print(f"\n🔬 Multi-Model BERTScore Comparison")
    print("=" * 50)
    
    models = [
        "distilroberta-base",
        "roberta-base", 
        # "emilyalsentzer/Bio_ClinicalBERT",  # Uncomment if available
    ]
    
    # Sample test case
    reference = "The chest X-ray shows clear lungs bilaterally with no acute cardiopulmonary abnormalities."
    candidate = "Bilateral lungs are clear without acute cardiopulmonary findings."
    
    results = {}
    
    for model in models:
        try:
            from bert_score import BERTScorer
            
            print(f"\n🧪 Testing {model}...")
            
            scorer = BERTScorer(
                model_type=model,
                lang="en",
                rescale_with_baseline=True
            )
            
            precision, recall, f1 = scorer.score([candidate], [reference])
            
            results[model] = {
                "f1": float(f1[0]),
                "precision": float(precision[0]),
                "recall": float(recall[0])
            }
            
            print(f"   F1: {f1[0]:.4f}, P: {precision[0]:.4f}, R: {recall[0]:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[model] = {"error": str(e)}
    
    # Log comparison results
    comparison_results = {
        "comparison_type": "multi_model",
        "test_case": {"reference": reference, "candidate": candidate},
        "model_results": results
    }
    
    # Find best performing model
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best_model = max(valid_results.keys(), key=lambda k: valid_results[k]["f1"])
        print(f"\n🏆 Best performing model: {best_model} (F1: {valid_results[best_model]['f1']:.4f})")
        
        comparison_results["best_model"] = best_model
        comparison_results["best_f1"] = valid_results[best_model]["f1"]
    
    log_bertscore_evaluation(comparison_results, {"evaluation_type": "model_comparison"})
    
    return results

if __name__ == "__main__":
    print("🤖 BERTScore Evaluation Summary & Logging")
    print("=" * 60)
    
    # Run standard evaluation
    results, log_entry = run_bertscore_evaluation_with_logging()
    
    # Show evaluation history
    display_evaluation_history()
    
    # Compare models (optional)
    print(f"\n" + "="*60)
    compare_bertscore_models()
    
    print(f"\n" + "="*60)
    if results:
        print("✅ BERTScore evaluation completed and logged!")
        print(f"📊 Mean F1 Score: {results['mean_f1']:.4f}")
        print("📁 Results saved to bertscore_evaluation_summary.json")
    else:
        print("⚠️  BERTScore evaluation completed with limitations")
    
    print("\n💡 Summary Features:")
    print("• Timestamped evaluation logs")
    print("• Configuration tracking")
    print("• Performance comparison across runs")
    print("• Multi-model evaluation support")
