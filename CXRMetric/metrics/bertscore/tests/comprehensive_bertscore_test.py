#!/usr/bin/env python3
"""
Comprehensive BERTScore Evaluation Suite

This script combines BERTScore testing, evaluation, logging, and analysis.
It provides both standalone testing functionality and timestamped logging
for tracking performance over time.

Features:
- Package availability testing
- BERTScore evaluation with sample data
- Timestamped result logging
- Historical performance tracking
- Multi-model comparison
- Educational demonstrations

Run from the project root directory:
python CXRMetric/metrics/bertscore/tests/comprehensive_bertscore_test.py
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to Python path if running directly
if __name__ == "__main__":
    project_root = Path(__file__).parents[4]
    sys.path.insert(0, str(project_root))

class BERTScoreTestSuite:
    """Comprehensive BERTScore testing and evaluation suite."""
    
    def __init__(self):
        self.bertscore_folder = Path(__file__).parent.parent
        self.summary_file = self.bertscore_folder / "bertscore_evaluation_summary.json"
        self.bert_scorer = None
        
    def test_package_availability(self) -> tuple[bool, Optional[str]]:
        """Test if BERTScore package is available."""
        
        print("🤖 BERTScore Package Availability Test")
        print("=" * 50)
        
        try:
            from bert_score import BERTScorer
            import torch
            print("✅ bert-score package is available!")
            print("✅ PyTorch is available!")
            
            # Test basic initialization
            scorer = BERTScorer(model_type="distilroberta-base", lang="en")
            print("✅ BERTScorer initialized successfully!")
            
            return True, "distilroberta-base"
            
        except ImportError as e:
            missing_pkg = str(e).split("'")[1] if "'" in str(e) else "unknown"
            print(f"❌ Missing package: {missing_pkg}")
            print("\n💡 Installation Instructions:")
            
            if "bert_score" in str(e):
                print("   pip install bert-score")
            elif "torch" in str(e):
                print("   pip install torch")
            else:
                print("   pip install bert-score torch")
                
            print("\n📝 BERTScore Features (when available):")
            print("   • Uses contextual BERT embeddings")
            print("   • Better semantic similarity than BLEU")
            print("   • Supports multiple pre-trained models")
            print("   • Provides precision, recall, and F1 scores")
            
            return False, None
            
        except Exception as e:
            print(f"❌ Error testing BERTScore: {e}")
            print(f"Error type: {type(e).__name__}")
            return False, None
    
    def run_basic_evaluation(self, model_type: str = "distilroberta-base") -> Optional[Dict[str, Any]]:
        """Run basic BERTScore evaluation with sample medical reports."""
        
        print(f"\n📊 Basic BERTScore Evaluation")
        print("=" * 50)
        
        try:
            from bert_score import BERTScorer
            
            # Load BERTScore test cases from metrics data directory
            data_dir = Path(__file__).resolve()
            while data_dir.name != 'metrics' and data_dir.parent != data_dir:
                data_dir = data_dir.parent
            try:
                from CXRMetric.metrics.data_loader import load_metric_cases
                test_cases = load_metric_cases('bertscore')
            except Exception:
                consolidated = data_dir / 'data' / 'metrics_test_cases.json'
                with open(consolidated, 'r', encoding='utf-8') as _f:
                    data = json.load(_f)
                    test_cases = data.get('bertscore', [])
            
            references = [case['reference'] for case in test_cases]
            candidates = [case['candidate'] for case in test_cases]
            
            print(f"📝 Evaluating {len(test_cases)} report pairs")
            print(f"🧠 Model: {model_type}")
            
            # Initialize scorer
            scorer = BERTScorer(
                model_type=model_type,
                batch_size=32,
                lang="en",
                rescale_with_baseline=True
            )
            
            # Compute scores
            precision, recall, f1 = scorer.score(candidates, references)
            
            # Display individual results
            print("\n📈 Individual Results:")
            print("-" * 80)
            
            for i, case in enumerate(test_cases):
                print(f"\n{i+1}. {case['expected']}:")
                print(f"   Reference: {case['reference'][:65]}{'...' if len(case['reference']) > 65 else ''}")
                print(f"   Candidate: {case['candidate'][:65]}{'...' if len(case['candidate']) > 65 else ''}")
                print(f"   F1: {f1[i]:.4f} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f}")
            
            # Calculate statistics
            results = {
                "model_type": model_type,
                "mean_f1": float(f1.mean()),
                "mean_precision": float(precision.mean()),
                "mean_recall": float(recall.mean()),
                "std_f1": float(f1.std()),
                "num_samples": len(f1),
                "individual_scores": {
                    "f1": [float(score) for score in f1],
                    "precision": [float(score) for score in precision],
                    "recall": [float(score) for score in recall]
                },
                "test_cases": test_cases
            }
            
            # Score distribution analysis
            high_similarity = (f1 > 0.8).sum().item()
            medium_similarity = ((f1 >= 0.6) & (f1 <= 0.8)).sum().item()
            low_similarity = (f1 < 0.6).sum().item()
            
            results.update({
                "score_distribution": {
                    "high_count": high_similarity,
                    "medium_count": medium_similarity,
                    "low_count": low_similarity,
                    "high_pct": 100 * high_similarity / len(f1),
                    "medium_pct": 100 * medium_similarity / len(f1),
                    "low_pct": 100 * low_similarity / len(f1)
                }
            })
            
            # Display summary
            print(f"\n📊 Summary Statistics:")
            print(f"   Mean F1:        {results['mean_f1']:.4f}")
            print(f"   Mean Precision: {results['mean_precision']:.4f}")
            print(f"   Mean Recall:    {results['mean_recall']:.4f}")
            print(f"   Std F1:         {results['std_f1']:.4f}")
            
            print(f"\n📊 Score Distribution:")
            print(f"   High (>0.8):     {high_similarity:2d} ({results['score_distribution']['high_pct']:5.1f}%)")
            print(f"   Medium (0.6-0.8): {medium_similarity:2d} ({results['score_distribution']['medium_pct']:5.1f}%)")
            print(f"   Low (<0.6):      {low_similarity:2d} ({results['score_distribution']['low_pct']:5.1f}%)")
            
            return results
            
        except ImportError:
            print("❌ BERTScore package not available for evaluation")
            return None
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return None
    
    def demonstrate_bertscore_advantages(self):
        """Demonstrate BERTScore advantages over traditional metrics."""
        
        print(f"\n🎯 BERTScore vs Traditional Metrics Comparison")
        print("=" * 60)
        
        # Load comparison demonstration cases from consolidated metrics data
        data_dir = Path(__file__).resolve()
        while data_dir.name != 'metrics' and data_dir.parent != data_dir:
            data_dir = data_dir.parent
        consolidated = data_dir / 'data' / 'metrics_test_cases.json'
        with open(consolidated, 'r', encoding='utf-8') as _f:
            data = json.load(_f)
            comparison_cases = data.get('bertscore_comparison', [])
        
        try:
            from CXRMetric.metrics.data_loader import load_metric_cases
            comparison_cases = load_metric_cases('bertscore_comparison')
        except Exception:
            comparison_cases = data.get('bertscore_comparison', [])
        
        print("🔍 Demonstration Cases:")
        for i, case in enumerate(comparison_cases, 1):
            print(f"\n{i}. {case['case']}:")
            print(f"   Reference: {case['reference']}")
            print(f"   Candidate: {case['candidate']}")
            print(f"   BLEU Expected:      {case['bleu_expected']}")
            print(f"   BERTScore Expected: {case['bertscore_expected']}")
        
        print(f"\n💡 Key Advantages of BERTScore:")
        print("   • Captures semantic meaning beyond surface text")
        print("   • Understands medical terminology relationships")
        print("   • Less sensitive to word order variations")
        print("   • Better correlation with human judgments")
        print("   • Handles paraphrasing and synonyms effectively")
        print("   • Provides both precision and recall metrics")
    
    def compare_models(self) -> Optional[Dict[str, Any]]:
        """Compare BERTScore performance across different models."""
        
        print(f"\n🔬 Multi-Model BERTScore Comparison")
        print("=" * 50)
        
        models = [
            ("distilroberta-base", "Fast, lightweight model"),
            ("roberta-base", "Standard RoBERTa model"),
            # Clinical models (uncomment if available)
            # ("emilyalsentzer/Bio_ClinicalBERT", "Clinical domain BERT"),
            # ("allenai/scibert_scivocab_uncased", "Scientific BERT"),
        ]
        
        # Test case for comparison
        reference = "The chest X-ray shows clear lungs bilaterally with no acute cardiopulmonary abnormalities."
        candidate = "Bilateral lungs are clear without acute cardiopulmonary findings."
        
        results = {}
        
        print(f"🧪 Test Case:")
        print(f"   Reference: {reference}")
        print(f"   Candidate: {candidate}")
        print(f"\n📊 Model Comparison Results:")
        
        for model_name, description in models:
            try:
                from bert_score import BERTScorer
                
                print(f"\n🧠 Testing {model_name}...")
                print(f"   Description: {description}")
                
                scorer = BERTScorer(
                    model_type=model_name,
                    lang="en",
                    rescale_with_baseline=True
                )
                
                precision, recall, f1 = scorer.score([candidate], [reference])
                
                results[model_name] = {
                    "f1": float(f1[0]),
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    "description": description
                }
                
                print(f"   F1: {f1[0]:.4f} | Precision: {precision[0]:.4f} | Recall: {recall[0]:.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[model_name] = {"error": str(e), "description": description}
        
        # Determine best model
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            best_model = max(valid_results.keys(), key=lambda k: valid_results[k]["f1"])
            print(f"\n🏆 Best performing model: {best_model}")
            print(f"   F1 Score: {valid_results[best_model]['f1']:.4f}")
            print(f"   Description: {valid_results[best_model]['description']}")
        
        return results
    
    def log_results(self, results: Dict[str, Any], config: Dict[str, Any] = None):
        """Log evaluation results with timestamp."""
        
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "evaluation_type": "comprehensive_bertscore",
            "configuration": config or {},
            "results": results
        }
        
        # Load existing logs
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add new entry
        logs.append(log_entry)
        
        # Save updated logs
        with open(self.summary_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"✅ Results logged to {self.summary_file}")
        return log_entry
    
    def display_evaluation_history(self):
        """Display historical evaluation results."""
        
        if not self.summary_file.exists():
            print("📝 No evaluation history found")
            return
        
        with open(self.summary_file, 'r') as f:
            logs = json.load(f)
        
        print(f"\n📚 Evaluation History ({len(logs)} entries)")
        print("=" * 60)
        
        # Show last 5 entries
        recent_logs = logs[-5:] if len(logs) > 5 else logs
        
        for i, log in enumerate(recent_logs, 1):
            timestamp = log['timestamp']
            results = log.get('results', {})
            config = log.get('configuration', {})
            
            print(f"\n{i}. {timestamp}")
            
            if 'model_type' in results:
                print(f"   Model: {results['model_type']}")
            
            if 'mean_f1' in results:
                print(f"   Mean F1: {results['mean_f1']:.4f}")
                
            if 'num_samples' in results:
                print(f"   Samples: {results['num_samples']}")
                
            if 'score_distribution' in results:
                dist = results['score_distribution']
                high_count = dist.get('high_count', 0)
                total = results.get('num_samples', 1)
                print(f"   High Quality: {high_count}/{total} ({100*high_count/total:.1f}%)")
    
    def run_comprehensive_test(self):
        """Run the complete BERTScore test suite."""
        
        print("🤖 Comprehensive BERTScore Evaluation Suite")
        print("=" * 70)
        
        # Test package availability
        available, default_model = self.test_package_availability()
        
        results = {}
        
        if available:
            # Run basic evaluation
            eval_results = self.run_basic_evaluation(default_model)
            if eval_results:
                results.update(eval_results)
            
            # Compare models
            model_comparison = self.compare_models()
            if model_comparison:
                results['model_comparison'] = model_comparison
            
            # Log results
            config = {
                "test_type": "comprehensive",
                "default_model": default_model,
                "timestamp": datetime.now().isoformat()
            }
            self.log_results(results, config)
        
        # Educational demonstrations (always run)
        self.demonstrate_bertscore_advantages()
        
        # Show history
        self.display_evaluation_history()
        
        print(f"\n" + "=" * 70)
        if available and results:
            print("✅ Comprehensive BERTScore evaluation completed!")
            if 'mean_f1' in results:
                print(f"📊 Mean F1 Score: {results['mean_f1']:.4f}")
            print("📁 Results logged for historical tracking")
        else:
            print("⚠️  BERTScore evaluation completed with limitations")
            print("💡 Install bert-score package for full functionality")
        
        print("\n🎯 Summary:")
        print("• BERTScore provides semantic similarity beyond surface text")
        print("• Ideal for medical report evaluation with paraphrasing")
        print("• Multiple model options available for different needs")
        print("• Historical tracking enables performance monitoring")
        
        return results

def main():
    """Main execution function."""
    suite = BERTScoreTestSuite()
    return suite.run_comprehensive_test()

if __name__ == "__main__":
    main()
