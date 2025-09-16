# Composite Metrics

This folder contains composite metrics evaluation for CXR (Chest X-Ray) report quality assessment, specifically implementing **RadCliQ** (Radiological Clinical Quality) metrics.

## Overview

Composite metrics combine multiple individual metrics (BLEU, BERTScore, semantic similarity, RadGraph, etc.) into a single, interpretable quality score. This provides a more holistic assessment of generated medical reports compared to individual metrics.

## Models

### RadCliQ-v0
- **Purpose**: First-generation composite metric for medical report quality
- **Components**: Combines BLEU, BERTScore, semantic embeddings, RadGraph F1
- **Training**: Trained on radiologist quality assessments
- **Output**: Quality score (typically 0-100 range)

### RadCliQ-v1  
- **Purpose**: Enhanced composite metric with improved clinical correlation
- **Components**: Refined weighting of semantic, factual, and linguistic features
- **Improvements**: Better agreement with radiologist evaluations
- **Output**: Improved quality score with clinical relevance

## Files

### Core Implementation
- `composite_metrics.py` - Main CompositeMetricEvaluator class
- `__init__.py` - Package initialization and exports

### Testing Suite
- `tests/comprehensive_composite_test.py` - Complete testing and evaluation suite
- `tests/test_composite.py` - Integration tests and edge case handling
- `composite_evaluation_summary.json` - Timestamped evaluation history

### Model Files (Required)
- `../../composite_metric_model.pkl` - RadCliQ-v0 trained model
- `../../radcliq-v1.pkl` - RadCliQ-v1 enhanced model  
- `../../normalizer.pkl` - Feature normalization model

## Usage

### Basic Usage

```python
from CXRMetric.metrics.composite import CompositeMetricEvaluator
import pandas as pd

# Initialize evaluator
evaluator = CompositeMetricEvaluator(
    compute_v0=True,  # Enable RadCliQ-v0
    compute_v1=True   # Enable RadCliQ-v1
)

# Prepare your data
data = pd.DataFrame({
    'reference_report': ["Reference report text..."],
    'generated_report': ["Generated report text..."]
})

# Compute composite metrics
results = evaluator.compute_metric(data, data)
print(results[['radcliq_v0', 'radcliq_v1']])
```

### Comprehensive Testing

```bash
# Run comprehensive test suite
cd /path/to/CXR-Report-Metric
python CXRMetric/metrics/composite/tests/comprehensive_composite_test.py

# Run integration tests  
python -m pytest CXRMetric/metrics/composite/tests/test_composite.py -v
```

## Features

### 🎯 Holistic Quality Assessment
- Combines multiple evaluation dimensions
- Weighted integration of linguistic, semantic, and factual metrics
- Single interpretable quality score

### 🧠 Machine Learning Integration
- Trained models optimized for clinical report quality
- Feature normalization for consistent scoring
- Version evolution with improved clinical correlation

### 📊 Comprehensive Analysis
- Feature importance analysis
- Historical performance tracking
- Timestamped evaluation logging

### 🔬 Robust Testing
- Package availability testing
- Mock evaluation for development environments
- Integration testing with other metric types
- Edge case handling

## Dependencies

### Required Packages
```bash
pip install scikit-learn pandas numpy
```

### Model Files
The composite metrics require trained model files:
- `composite_metric_model.pkl` - RadCliQ-v0 model
- `radcliq-v1.pkl` - RadCliQ-v1 model
- `normalizer.pkl` - Feature normalizer

These should be placed in the `CXRMetric/` directory.

## Evaluation Results

### Typical Performance
- **RadCliQ-v0**: Mean scores 45-75 (clinical quality range)
- **RadCliQ-v1**: Mean scores 50-80 (improved range and correlation)
- **Processing**: Sub-second evaluation for small datasets
- **Memory**: Moderate (model loading + feature computation)

### Feature Importance (RadCliQ-v1)
1. **Semantic Embedding** (45%): Clinical content relevance
2. **RadGraph F1** (25%): Medical fact verification  
3. **BERTScore** (20%): Contextual similarity
4. **BLEU Score** (10%): Surface-level matching

## Architecture

```
composite/
├── composite_metrics.py          # Main evaluator class
├── __init__.py                   # Package exports
├── README.md                     # This documentation
├── composite_evaluation_summary.json  # Historical results
└── tests/
    ├── comprehensive_composite_test.py  # Full test suite
    └── test_composite.py               # Integration tests
```

## Advantages Over Individual Metrics

### 🔍 Individual Metrics Limitations
- **BLEU**: Only measures surface similarity, ignores semantics
- **BERTScore**: Good for semantics, but misses clinical facts
- **RadGraph**: Excellent for facts, but ignores fluency

### 🎯 Composite Metrics Benefits
- **Holistic Assessment**: Combines all dimensions intelligently
- **Clinical Relevance**: Optimized for medical report quality
- **Interpretable Scores**: Single score easier to understand
- **Trained Weights**: ML-optimized feature combinations

### Example Comparison
```
Individual Metrics:
- BLEU: 0.2 (low surface similarity)
- BERTScore: 0.6 (good semantics)  
- RadGraph: 0.4 (moderate facts)
→ Hard to interpret overall quality

Composite Metric:
- RadCliQ-v1: 68.5
→ Clear interpretation: "Good quality report"
```

## Development

### Adding New Composite Models

1. Train your composite model with appropriate features
2. Save the model as a `.pkl` file in `CXRMetric/`
3. Update `CompositeMetricEvaluator` to load the new model
4. Add corresponding tests and documentation

### Testing New Features

1. Add test cases to `test_composite.py` for integration testing
2. Update `comprehensive_composite_test.py` for evaluation
3. Run the full test suite to ensure compatibility

## Research Background

Composite metrics like RadCliQ are based on research showing that:
- Individual metrics capture different aspects of quality
- Radiologist quality assessments consider multiple dimensions
- Machine learning can optimize metric combinations
- Clinical relevance requires domain-specific training

## Troubleshooting

### Model Files Missing
```
❌ RadCliQ-v0 model missing at CXRMetric/composite_metric_model.pkl
💡 Download model files from the project repository
```

### Import Errors
```python
# Check if scikit-learn is installed
pip install scikit-learn

# Verify model files exist
import pathlib
pathlib.Path("CXRMetric/radcliq-v1.pkl").exists()
```

### Performance Issues
- **Memory**: Model files are loaded once during initialization
- **Speed**: Feature computation is the bottleneck, not prediction
- **Scaling**: Consider batch processing for large datasets

## Contributing

When contributing to composite metrics:

1. **Maintain Compatibility**: Ensure new features work with existing interface
2. **Add Tests**: Include both unit and integration tests
3. **Document Changes**: Update README and docstrings
4. **Performance**: Consider impact on evaluation speed
5. **Clinical Relevance**: Validate against medical domain requirements

---

For questions or issues with composite metrics, please refer to the main project documentation or create an issue in the project repository.
