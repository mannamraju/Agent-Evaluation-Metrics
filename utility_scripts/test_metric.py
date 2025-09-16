import config
# Import both evaluation approaches for compatibility testing
from CXRMetric.run_eval import calc_metric as original_calc_metric
from CXRMetric.modular_evaluation import calc_metric as modular_calc_metric

gt_reports = config.GT_REPORTS
predicted_reports = config.PREDICTED_REPORTS
out_file = config.OUT_FILE
use_idf = config.USE_IDF

if __name__ == "__main__":
    print("🚀 Testing CXR-Report-Metric evaluation...")
    print(f"Ground truth: {gt_reports}")
    print(f"Predictions: {predicted_reports}")
    print(f"Output: {out_file}")
    print(f"Use IDF: {use_idf}")
    
    try:
        print("\n📊 Running modular evaluation (recommended)...")
        modular_summary = modular_calc_metric(gt_reports, predicted_reports, out_file, use_idf)
        print("✅ Modular evaluation completed successfully!")
        
        # Print key results
        if 'mean_metrics' in modular_summary:
            print("\n📈 Key Results:")
            for metric, value in modular_summary['mean_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        # Print CheXpert results if available
        if 'chexpert' in modular_summary:
            print(f"\n🏥 CheXpert Micro-F1: {modular_summary['chexpert']['micro_f1']:.4f}")
        
        # Print bounding box results if available
        if 'box_precision' in modular_summary and 'box_recall' in modular_summary:
            print(f"\n📦 Bounding Box - Precision: {modular_summary['box_precision']:.4f}, Recall: {modular_summary['box_recall']:.4f}")
        
    except Exception as e:
        print(f"❌ Modular evaluation failed: {e}")
        print("\n🔄 Falling back to original evaluation...")
        
        try:
            original_summary = original_calc_metric(gt_reports, predicted_reports, out_file, use_idf)
            print("✅ Original evaluation completed successfully!")
        except Exception as e2:
            print(f"❌ Original evaluation also failed: {e2}")
            raise
