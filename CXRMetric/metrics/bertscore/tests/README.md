# BERTScore Evaluation Tests

This directory contains comprehensive testing and evaluation tools for BERTScore metrics in the CXR Report evaluation system.

## Files

### `comprehensive_bertscore_test.py` ⭐ **Main Test Suite**
Complete BERTScore evaluation suite combining testing, logging, and analysis.

**Features:**
- Package availability testing
- BERTScore evaluation with medical report samples
- Timestamped result logging (JSON format)
- Historical performance tracking
- Multi-model comparison (DistilRoBERTa, RoBERTa, Clinical BERT)
- Educational demonstrations vs BLEU

**Usage:**
```bash
# Run from project root
python CXRMetric/metrics/bertscore/tests/comprehensive_bertscore_test.py
```

### ~~`bertscore_standalone_test.py`~~ **[DEPRECATED]**
Replaced by `comprehensive_bertscore_test.py`

### ~~`test_bertscore.py`~~ **[DEPRECATED]** 
Replaced by `comprehensive_bertscore_test.py`

## Key Features

### 🧪 **Package Testing**
- Checks bert-score and PyTorch availability
- Provides installation instructions if missing
- Tests basic BERTScorer initialization

### 📊 **Comprehensive Evaluation**
- 8 medical report test cases covering different scenarios:
  - Semantic equivalence
  - Medical paraphrasing  
  - Clinical terminology
  - Word order variations
  - Technical vs descriptive language

### 📈 **Performance Analysis**
- F1, Precision, Recall metrics
- Score distribution analysis (High/Medium/Low)
- Statistical summaries (mean, std deviation)
- Individual case breakdowns

### 🔬 **Model Comparison**
- DistilRoBERTa (fast, lightweight)
- RoBERTa-base (standard performance)
- Clinical BERT (domain-specific, when available)
- SciBERT (scientific text, when available)

### 📚 **Historical Tracking**
- Timestamped evaluation logs
- Performance trends over time
- Configuration tracking
- JSON-formatted results storage

### 🎯 **Educational Content**
- BERTScore vs BLEU comparison
- Medical domain advantages
- Semantic similarity demonstrations
- Installation and usage guidance

## Sample Results

### Typical Performance
```
Mean F1:        0.4465
Mean Precision: 0.4850
Mean Recall:    0.4083
```

### Score Distribution
- **High (>0.8)**: 0% (Rare for paraphrased medical content)
- **Medium (0.6-0.8)**: 12.5% (Good semantic matches)
- **Low (<0.6)**: 87.5% (Typical for medical paraphrasing)

### Model Comparison
- **RoBERTa-base**: F1 0.6170 (Best overall)
- **DistilRoBERTa**: F1 0.6132 (Fastest, nearly equivalent)

## Understanding BERTScore for Medical Reports

### Why BERTScore Works Well for Medical Text
- **Semantic Understanding**: Captures meaning beyond word overlap
- **Medical Terminology**: Understands clinical synonyms and paraphrasing
- **Contextual Embeddings**: Uses BERT's deep language understanding
- **Multiple Metrics**: Precision, recall, and F1 for comprehensive evaluation

### Score Interpretation
- **F1 > 0.8**: Excellent semantic similarity (rare)
- **F1 0.6-0.8**: Good similarity with meaningful overlap
- **F1 0.4-0.6**: Moderate similarity, some semantic connection
- **F1 < 0.4**: Limited similarity

### Best Practices
1. **Use F1 Score**: Harmonic mean of precision and recall
2. **Medical Models**: Consider clinical BERT when available
3. **Baseline Rescaling**: Always enabled for normalized scores
4. **Batch Processing**: Efficient for multiple evaluations

## Installation Requirements

```bash
pip install bert-score torch transformers
```

## Running Tests

```bash
# From project root
cd /path/to/CXR-Report-Metric

# Run comprehensive test
python CXRMetric/metrics/bertscore/tests/comprehensive_bertscore_test.py

# Results logged to:
# CXRMetric/metrics/bertscore/bertscore_evaluation_summary.json
```

## Integration with Main Evaluation System

The BERTScore evaluator integrates with the main CXRMetric system:

```python
from CXRMetric.metrics.bertscore import BERTScoreEvaluator

evaluator = BERTScoreEvaluator(model_type="distilroberta-base")
results = evaluator.compute_metric(gt_df, pred_df)
```

## Next Steps

1. **Install Clinical Models**: Test with medical domain BERT models
2. **Compare with BLEU**: Run side-by-side with BLEU evaluation
3. **Tune Thresholds**: Adjust score interpretation for your use case
4. **Track Performance**: Use logged results to monitor evaluation trends
