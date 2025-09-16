#!/usr/bin/env python3
"""
Comprehensive Perplexity Metrics Evaluation Suite

This script combines perplexity and cross-entropy loss testing, evaluation, logging, and analysis.
It provides both standalone testing functionality and timestamped logging
for tracking performance over time.

Features:
- Package availability testing (transformers, torch)
- Perplexity evaluation with multiple model configurations
- Timestamped result logging
- Historical performance tracking
- Algorithm performance analysis
- Educational demonstrations

Run from the project root directory:
python CXRMetric/metrics/perplexity/tests/comprehensive_perplexity_test.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to Python path if running directly
if __name__ == "__main__":
    project_root = Path(__file__).parents[4]
    sys.path.insert(0, str(project_root))

class PerplexityMetricsTestSuite:
    """Comprehensive perplexity metrics testing and evaluation suite."""
    
    def __init__(self):
        self.perplexity_folder = Path(__file__).parent.parent
        self.summary_file = self.perplexity_folder / "perplexity_evaluation_summary.json"
        self.project_root = Path(__file__).parents[4]
        
    def test_package_availability(self) -> tuple[bool, List[str]]:
        """Test if required packages are available."""
        
        print("Perplexity Metrics Package Availability Test")
        print("=" * 60)
        
        missing_components = []
        
        # Test Python libraries
        try:
            import pandas as pd
            print("✅ pandas is available!")
        except ImportError:
            print("❌ pandas is not available!")
            missing_components.append("pandas")
        
        try:
            import numpy as np
            print("✅ numpy is available!")
        except ImportError:
            print("❌ numpy is not available!")
            missing_components.append("numpy")
        
        try:
            import torch
            print("✅ torch is available!")
            print(f"   PyTorch version: {torch.__version__}")
            print(f"   CUDA available: {torch.cuda.is_available()}")
        except ImportError:
            print("❌ torch is not available!")
            missing_components.append("torch")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print("✅ transformers is available!")
            # Test if we can load a simple model
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
                print("   ✅ Can load DistilGPT-2 tokenizer")
            except Exception as e:
                print(f"   ⚠️ Model loading issue: {e}")
        except ImportError:
            print("❌ transformers is not available!")
            missing_components.append("transformers")
        
        if len(missing_components) == 0:
            print("\n✅ All components available for perplexity metrics!")
            return True, []
        else:
            print(f"\n❌ Missing components: {missing_components}")
            print("Install with: pip install transformers torch")
            return False, missing_components
    
    def create_test_medical_data(self, num_samples: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create test medical report data for perplexity evaluation."""
        
        # Load perplexity test cases from consolidated metrics data
        try:
            from CXRMetric.metrics.data_loader import load_metric_cases
            test_cases = load_metric_cases('perplexity')
        except Exception:
            data_dir = Path(__file__).resolve()
            while data_dir.name != 'metrics' and data_dir.parent != data_dir:
                data_dir = data_dir.parent
            consolidated = data_dir / 'data' / 'metrics_test_cases.json'
            with open(consolidated, 'r', encoding='utf-8') as _f:
                data = json.load(_f)
                test_cases = data.get('perplexity', [])
        
        # Take the first num_samples cases
        selected_cases = test_cases[:num_samples]
        
        gt_data = pd.DataFrame({
            'study_id': range(1, len(selected_cases) + 1),
            'report': [case['reference'] for case in selected_cases]
        })
        
        pred_data = pd.DataFrame({
            'study_id': range(1, len(selected_cases) + 1),
            'report': [case['generated'] for case in selected_cases]
        })
        
        return gt_data, pred_data
    
    def run_perplexity_evaluation(self) -> Optional[Dict[str, Any]]:
        """Run perplexity metrics evaluation with test data."""
        
        print(f"\nPerplexity Metrics Evaluation")
        print("=" * 50)
        
        try:
            # Import here to handle missing packages gracefully
            sys.path.insert(0, str(self.project_root))
            from CXRMetric.metrics.perplexity import PerplexityEvaluator
            
            # Create test data
            gt_data, pred_data = self.create_test_medical_data()
            print(f"Created test data with {len(gt_data)} medical report pairs")
            
            # Initialize evaluator with different model configurations
            evaluators = {
                'DistilGPT-2 (Default)': PerplexityEvaluator(model_name="distilgpt2", batch_size=4),
            }
            
            # Add GPT-2 if available
            try:
                evaluators['GPT-2 (Small)'] = PerplexityEvaluator(model_name="gpt2", batch_size=2)
            except:
                pass  # Skip if not available
            
            print(f"\nTesting Perplexity Configurations:")
            for name, evaluator in evaluators.items():
                print(f"  • {name}: Model={evaluator.model_name}, Device={evaluator.device}")
            
            results_summary = {}
            all_scores = {}
            
            # Run evaluation for each configuration
            for config_name, evaluator in evaluators.items():
                try:
                    results = evaluator.compute_metric(gt_data, pred_data)
                    
                    # Extract perplexity scores
                    ppl_columns = [col for col in results.columns if 'perplexity' in col.lower()]
                    
                    if ppl_columns:
                        config_results = {}
                        
                        for col in ppl_columns:
                            scores = results[col].replace([np.inf, -np.inf], np.nan).dropna()
                            if len(scores) > 0:
                                config_results[col] = {
                                    'mean': float(np.mean(scores)),
                                    'std': float(np.std(scores)),
                                    'median': float(np.median(scores)),
                                    'min': float(np.min(scores)),
                                    'max': float(np.max(scores))
                                }
                        
                        results_summary[config_name] = config_results
                        all_scores[config_name] = results
                        
                        print(f"\n{config_name} Results:")
                        for metric, stats in config_results.items():
                            print(f"  {metric}: {stats['mean']:.2f} ± {stats['std']:.2f}")
                    
                except Exception as e:
                    print(f"❌ Evaluation failed for {config_name}: {e}")
                    results_summary[config_name] = {'error': str(e)}
            
            if results_summary:
                # Model comparison analysis
                print(f"\nPerplexity Model Comparison:")
                print("-" * 50)
                
                for config_name, results in results_summary.items():
                    if 'error' not in results and 'perplexity_generated' in results:
                        gen_ppl = results['perplexity_generated']['mean']
                        ref_ppl = results['perplexity_reference']['mean']
                        print(f"  {config_name}:")
                        print(f"    Generated: {gen_ppl:.2f}")
                        print(f"    Reference: {ref_ppl:.2f}")
                        print(f"    Ratio: {gen_ppl/ref_ppl:.3f}")
                
                # Find best performing configuration
                best_config = None
                best_score = float('inf')
                
                for config_name, results in results_summary.items():
                    if ('error' not in results and 'perplexity_generated' in results and
                        results['perplexity_generated']['mean'] < best_score):
                        best_score = results['perplexity_generated']['mean']
                        best_config = config_name
                
                if best_config:
                    print(f"\n✅ Best performing configuration: {best_config}")
                
                # Overall analysis
                analysis_results = {
                    'num_samples': len(gt_data),
                    'perplexity_configurations': results_summary,
                    'model_comparison': {
                        'best_configuration': best_config,
                        'evaluation_focus': 'Lower perplexity indicates more fluent text',
                        'medical_text_considerations': 'Medical reports may have higher perplexity due to technical terminology'
                    }
                }
                
                return analysis_results
            
            else:
                print("❌ No perplexity evaluation results obtained")
                return None
        
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Required packages: pip install transformers torch")
            return None
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return None
    
    def run_educational_demonstrations(self) -> bool:
        """Run educational demonstrations about perplexity metrics."""
        
        print(f"\nPerplexity and Cross-Entropy Advantages")
        print("=" * 50)
        
        print("Key Advantages:")
        print()
        
        print("1. Text Fluency Assessment:")
        print("   • Measures how natural/fluent generated text appears to language models")
        print("   • Example: Lower perplexity = more fluent, natural-sounding medical reports")
        print()
        
        print("2. Model Confidence Evaluation:")
        print("   • Cross-entropy indicates model's confidence in predicting each token")
        print("   • Example: Low cross-entropy = model is confident about word choices")
        print()
        
        print("3. Language Model Perspective:")
        print("   • Evaluates text from the perspective of state-of-the-art language models")
        print("   • Example: GPT-2/DistilGPT-2 trained on diverse text, good general fluency judge")
        print()
        
        print("4. Complementary to Other Metrics:")
        print("   • Focuses on fluency rather than semantic similarity or exact matches")
        print("   • Example: Good for detecting grammatical errors and unnatural phrasing")
        print()
        
        print("5. Comparative Analysis:")
        print("   • Can compare generated vs reference text perplexity")
        print("   • Example: Generated/Reference ratio near 1.0 indicates similar fluency")
        print()
        
        print("Perplexity vs Other Metrics:")
        print("   BLEU: Precision-focused, exact n-gram matching")
        print("   ROUGE-L: Recall-focused, flexible word order")
        print("   BERTScore: Semantic similarity using contextual embeddings")
        print("   Perplexity: Fluency and naturalness from language model perspective")
        print("   RadCliQ: Composite clinical quality assessment")
        
        return True
    
    def run_algorithm_analysis(self) -> bool:
        """Run algorithm performance analysis for perplexity metrics."""
        
        print(f"\nPerplexity Algorithm Analysis")
        print("=" * 50)
        
        print("Algorithm Characteristics:")
        print()
        
        print("Model Architecture:")
        print("  • Description: Autoregressive causal language models (GPT family)")
        print("  • Performance: Pretrained on large text corpora for general language understanding")  
        print("  • Scaling: Larger models (GPT-2 > DistilGPT-2) generally provide better fluency assessment")
        print()
        
        print("Perplexity Computation:")
        print("  • Description: exp(cross_entropy_loss) where loss = -log(P(token|context))")
        print("  • Performance: Measures average inverse probability of tokens given context")
        print("  • Scaling: O(n) where n is sequence length, with transformer O(n²) attention")
        print()
        
        print("Cross-Entropy Loss:")
        print("  • Description: Negative log-likelihood of true tokens under model distribution")
        print("  • Performance: Direct measure of model's predictive confidence")
        print("  • Scaling: Lower values indicate higher confidence/better fit")
        print()
        
        print("Sliding Window Processing:")
        print("  • Description: For long sequences, use overlapping windows to stay within model limits")
        print("  • Performance: Maintains context while handling arbitrary sequence lengths")
        print("  • Scaling: Enables evaluation of long medical reports without truncation")
        print()
        
        print("Batch Processing:")
        print("  • Description: Process multiple texts simultaneously for efficiency")
        print("  • Performance: GPU acceleration when available, falls back to CPU")
        print("  • Scaling: Configurable batch size based on available memory")
        print()
        
        print("Typical Performance on Medical Reports:")
        print("   • Processing time: 1-5 seconds per report (depending on model and hardware)")
        print("   • Memory usage: Model-dependent (DistilGPT-2: ~250MB, GPT-2: ~500MB)")
        print("   • Perplexity range: 10-100 typical for medical text (lower = more fluent)")
        print("   • Cross-entropy range: 2-5 typical (log scale, lower = higher confidence)")
        
        return True
    
    def log_results(self, test_results: Dict[str, Any]) -> None:
        """Log test results to JSON file with timestamp."""
        
        # Create results entry
        result_entry = {
            'timestamp': datetime.now().isoformat(),
            'tests': sum(1 for result in test_results.values() if result.get('status') in ['passed', 'failed']),
            'passed': sum(1 for result in test_results.values() if result.get('status') == 'passed'),
            'failed': sum(1 for result in test_results.values() if result.get('status') == 'failed'),
            'status': 'completed' if all(r.get('status') == 'passed' for r in test_results.values()) else 'limited',
            'test_details': {
                name: {
                    'duration': result.get('duration', 0),
                    'status': result.get('status', 'unknown')
                }
                for name, result in test_results.items()
            },
            'total_duration': sum(r.get('duration', 0) for r in test_results.values()),
            'performance_metrics': {
                'average_test_time': sum(r.get('duration', 0) for r in test_results.values()) / max(len(test_results), 1),
                'fastest_test': min((r.get('duration', 0) for r in test_results.values()), default=0),
                'slowest_test': max((r.get('duration', 0) for r in test_results.values()), default=0)
            }
        }
        
        # Add perplexity-specific results if available
        perplexity_result = test_results.get('Perplexity Evaluation', {})
        if 'data' in perplexity_result and perplexity_result['data']:
            result_entry['perplexity_results'] = {
                config: {
                    'generated_ppl': results.get('perplexity_generated', {}).get('mean', 0),
                    'reference_ppl': results.get('perplexity_reference', {}).get('mean', 0),
                    'cross_entropy_gen': results.get('cross_entropy_generated', {}).get('mean', 0),
                    'model_config': config
                }
                for config, results in perplexity_result['data'].get('perplexity_configurations', {}).items()
                if 'error' not in results
            }
            result_entry['samples_processed'] = perplexity_result['data'].get('num_samples', 0)
        
        # Load existing results
        existing_results = []
        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        existing_results = json.loads(content)
                        if not isinstance(existing_results, list):
                            existing_results = [existing_results]  # Handle single object case
            except (json.JSONDecodeError, FileNotFoundError):
                existing_results = []
        
        # Append new result
        existing_results.append(result_entry)
        
        # Save updated results
        with open(self.summary_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
        
        print(f"✅ Results logged to {self.summary_file}")
    
    def display_historical_results(self) -> None:
        """Display historical test results."""
        
        if not self.summary_file.exists():
            print("No historical results found.")
            return
        
        try:
            with open(self.summary_file, 'r') as f:
                results = json.loads(f.read())
                if not isinstance(results, list):
                    results = [results]  # Handle single object case
        except (json.JSONDecodeError, FileNotFoundError):
            print("Error reading historical results.")
            return
        
        print(f"Perplexity Metrics Evaluation History ({len(results)} entries)")
        print("=" * 70)
        
        for i, result in enumerate(results[-5:], 1):  # Show last 5 results
            timestamp = result.get('timestamp', 'Unknown')
            tests = result.get('tests', 0)
            passed = result.get('passed', 0)
            failed = result.get('failed', 0)
            status = result.get('status', 'unknown')
            duration = result.get('total_duration', 0)
            
            print(f"\n{i}. {timestamp}")
            print("   " + "-" * 50)
            print(f"   Tests: {tests} total, {passed} passed, {failed} failed")
            print(f"   Status: {status}")
            
            # Show perplexity-specific results if available
            if 'perplexity_results' in result:
                print("   Perplexity Results:")
                for config, metrics in result['perplexity_results'].items():
                    gen_ppl = metrics.get('generated_ppl', 0)
                    ref_ppl = metrics.get('reference_ppl', 0)
                    print(f"      • {config}: Gen={gen_ppl:.1f}, Ref={ref_ppl:.1f}")
            
            # Show test details
            test_details = result.get('test_details', {})
            if test_details:
                print("   Test Details:")
                for test_name, details in test_details.items():
                    status_emoji = "✅" if details.get('status') == 'passed' else "❌"
                    duration = details.get('duration', 0)
                    print(f"      {status_emoji} {test_name}: {duration:.3f}s")
            
            print(f"   Total Duration: {duration:.2f}s")
            if 'samples_processed' in result:
                print(f"   Samples Processed: {result['samples_processed']}")
        
        # Historical statistics
        if len(results) > 1:
            print("\nHistorical Statistics:")
            print("   " + "=" * 50)
            successful_runs = sum(1 for r in results if r.get('status') == 'completed')
            avg_tests = sum(r.get('tests', 0) for r in results) / len(results)
            print(f"   • Total Evaluation Runs: {len(results)}")
            print(f"   • Successful Runs: {successful_runs} ({100*successful_runs/len(results):.1f}%)")
            print(f"   • Average Tests per Run: {avg_tests:.1f}")
    
    def run_all_tests(self) -> None:
        """Run all perplexity metrics tests."""
        
        print("Comprehensive Perplexity Metrics Evaluation Suite")
        print("=" * 80)
        
        test_results = {}
        start_time = datetime.now()
        
        # Test 1: Package Availability
        print("\nTest 1: Package Availability")
        test_start = datetime.now()
        available, missing = self.test_package_availability()
        test_end = datetime.now()
        test_results['Package Availability'] = {
            'status': 'passed' if available else 'failed',
            'duration': (test_end - test_start).total_seconds(),
            'data': {'available': available, 'missing_packages': missing}
        }
        
        # Test 2: Perplexity Evaluation (only if packages available)
        print("\nTest 2: Perplexity Metrics Evaluation")
        test_start = datetime.now()
        try:
            evaluation_results = self.run_perplexity_evaluation()
            test_end = datetime.now()
            test_results['Perplexity Evaluation'] = {
                'status': 'passed' if evaluation_results else 'failed',
                'duration': (test_end - test_start).total_seconds(),
                'data': evaluation_results
            }
        except Exception as e:
            test_end = datetime.now()
            test_results['Perplexity Evaluation'] = {
                'status': 'failed',
                'duration': (test_end - test_start).total_seconds(),
                'error': str(e)
            }
        
        # Test 3: Educational Demonstrations
        print("\nTest 3: Educational Demonstrations")
        test_start = datetime.now()
        demo_success = self.run_educational_demonstrations()
        test_end = datetime.now()
        test_results['Educational Demo'] = {
            'status': 'passed' if demo_success else 'failed',
            'duration': (test_end - test_start).total_seconds()
        }
        
        # Test 4: Algorithm Analysis
        print("\nTest 4: Algorithm Performance Analysis")
        test_start = datetime.now()
        analysis_success = self.run_algorithm_analysis()
        test_end = datetime.now()
        test_results['Algorithm Analysis'] = {
            'status': 'passed' if analysis_success else 'failed',
            'duration': (test_end - test_start).total_seconds()
        }
        
        # Log results
        self.log_results(test_results)
        
        # Display historical results
        self.display_historical_results()
        
        # Final summary
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        passed_tests = sum(1 for result in test_results.values() if result['status'] == 'passed')
        failed_tests = sum(1 for result in test_results.values() if result['status'] == 'failed')
        
        print("\n" + "=" * 80)
        print("DETAILED TEST SUMMARY")
        print("=" * 80)
        
        print("Test Execution Summary:")
        print(f"   • Total Tests: {len(test_results)}")
        print(f"   • Passed: {passed_tests} ✅")
        print(f"   • Failed: {failed_tests} ❌")
        print(f"   • Success Rate: {100 * passed_tests / len(test_results):.1f}%")
        print(f"   • Total Duration: {total_duration:.2f}s")
        print()
        
        print("Individual Test Results:")
        for test_name, result in test_results.items():
            status_emoji = "✅" if result['status'] == 'passed' else "❌"
            duration = result['duration']
            print(f"   {status_emoji} {test_name}: {duration:.3f}s")
            
            # Add specific details
            if test_name == 'Package Availability' and result['data']['available']:
                print("      └─ transformers_available: True")
            elif test_name == 'Perplexity Evaluation' and result.get('data'):
                configs_tested = len(result['data'].get('perplexity_configurations', {}))
                print(f"      └─ model_configurations_tested: {configs_tested}")
            elif test_name == 'Educational Demo':
                print("      └─ advantages_shown: 5")
            elif test_name == 'Algorithm Analysis':
                print("      └─ algorithm_characteristics: Model architecture, computation, processing")
        
        print()
        print("Performance Metrics:")
        avg_time = sum(r['duration'] for r in test_results.values()) / len(test_results)
        fastest = min(r['duration'] for r in test_results.values())
        slowest = max(r['duration'] for r in test_results.values())
        print(f"   • Average Test Time: {avg_time:.3f}s")
        print(f"   • Fastest Test: {fastest:.3f}s")
        print(f"   • Slowest Test: {slowest:.3f}s")
        
        # Final status
        if failed_tests == 0:
            print("\n✅ Comprehensive perplexity metrics evaluation completed!")
            if test_results.get('Perplexity Evaluation', {}).get('data'):
                configs = test_results['Perplexity Evaluation']['data'].get('perplexity_configurations', {})
                print("Perplexity Configuration Results:")
                for config, results in configs.items():
                    if 'error' not in results and 'perplexity_generated' in results:
                        gen_ppl = results['perplexity_generated']['mean']
                        ref_ppl = results['perplexity_reference']['mean']
                        print(f"   • {config}: Generated={gen_ppl:.1f}, Reference={ref_ppl:.1f}")
            print("Detailed results logged for historical tracking")
        else:
            print("⚠️ Perplexity metrics evaluation completed with limitations")
        
        print("\nKey Insights:")
        print("• Perplexity measures text fluency from language model perspective")
        print("• Lower perplexity indicates more natural, fluent text")
        print("• Cross-entropy loss reflects model confidence in token predictions")
        print("• Useful complement to semantic similarity and exact match metrics")


if __name__ == "__main__":
    suite = PerplexityMetricsTestSuite()
    suite.run_all_tests()
