# ROUGE Metrics

This folder contains ROUGE-L (Longest Common Subsequence) evaluation metrics for CXR (Chest X-Ray) report quality assessment.

## Overview

ROUGE-L is a recall-oriented evaluation metric that measures the longest common subsequence (LCS) between reference and generated texts. Unlike BLEU which requires exact n-gram matches, ROUGE-L allows for flexible word ordering while still capturing content overlap.

## Algorithm

### ROUGE-L (Longest Common Subsequence)
- **Purpose**: Measures content preservation with flexible word ordering
- **Method**: Uses dynamic programming to find longest common subsequence
- **Advantage**: Handles paraphrasing and sentence restructuring well
- **Time Complexity**: O(m × n) where m, n are text lengths
- **Space Complexity**: O(m × n) for dynamic programming table

## Files

### Core Implementation
- `rouge_metrics.py` - Main ROUGEEvaluator class with LCS algorithm
- `__init__.py` - Package initialization and exports

### Testing Suite
- `tests/comprehensive_rouge_test.py` - Complete testing and evaluation suite
- `tests/test_rouge.py` - Integration tests and edge case handling
- `rouge_evaluation_summary.json` - Timestamped evaluation history

## Usage

### Basic Usage

```python
from CXRMetric.metrics.rouge import ROUGEEvaluator
import pandas as pd

# Initialize evaluator
evaluator = ROUGEEvaluator(beta=1.2)  # Default beta for slight recall focus

# Prepare your data
data = pd.DataFrame({
    'reference_report': ["The chest X-ray shows clear lungs with no acute findings."],
    'generated_report': ["Chest radiograph demonstrates clear lung fields without abnormalities."]
})

# Compute ROUGE-L scores
results = evaluator.compute_metric(data, data)
print(results['rouge_l'])
```

### Beta Parameter Tuning

```python
# Different beta values for different evaluation focuses
evaluator_precision = ROUGEEvaluator(beta=0.5)   # Precision-focused
evaluator_balanced = ROUGEEvaluator(beta=1.0)    # Balanced F1
evaluator_recall = ROUGEEvaluator(beta=1.2)      # Recall-focused (default)
evaluator_strong_recall = ROUGEEvaluator(beta=2.0)  # Strong recall focus
```

### Comprehensive Testing

```bash
# Run comprehensive test suite
cd /path/to/CXR-Report-Metric
python CXRMetric/metrics/rouge/tests/comprehensive_rouge_test.py

# Run integration tests  
python CXRMetric/metrics/rouge/tests/test_rouge.py
```

## Features

### No External Dependencies
- Pure Python implementation using only standard libraries
- Fast and reliable - no model downloads or network requests
- Works offline and in constrained environments

### Flexible Sequence Matching
- Handles paraphrasing: "lung clear" → "clear lung fields"
- Captures content preservation even with word reordering
- Good for medical reports with varied sentence structures

### Configurable Beta Parameter
- Controls precision vs recall balance in F1 calculation
- β = 1.0: Balanced precision and recall
- β > 1.0: Emphasizes recall (content coverage)
- β < 1.0: Emphasizes precision (accuracy of matches)

### Comprehensive Analysis
- Multiple beta configurations tested simultaneously  
- Historical performance tracking
- Algorithm performance characteristics analysis
- Medical text specific optimizations

## Algorithm Details

### Longest Common Subsequence (LCS)
The core of ROUGE-L is finding the LCS between reference and generated text token sequences:

1. **Tokenization**: Split texts into token sequences
2. **DP Table**: Build m×n dynamic programming table
3. **LCS Computation**: Fill table using recurrence relation
4. **Score Calculation**: Compute precision, recall, and F1

### Scoring Formula
```
Precision = LCS_length / generated_length
Recall = LCS_length / reference_length
F1 = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

### Performance Characteristics
- **Time**: O(m × n) per text pair
- **Space**: O(m × n) for DP table
- **Typical Performance**: Sub-millisecond per medical report pair
- **Memory Usage**: Minimal (few KB per evaluation)

## Evaluation Results

### Typical Performance on Medical Reports
- **Processing Time**: < 1ms per report pair
- **Score Range**: 0.0 to 1.0 (higher is better)
- **Medical Text Scores**: Typically 0.3-0.8 for paraphrased content
- **Identical Text Score**: ~1.0 (perfect match)

### Beta Parameter Effects
- **β=0.5**: Emphasizes precision, good for accuracy assessment
- **β=1.0**: Balanced, standard F1 score
- **β=1.2**: Default, slight recall emphasis for content coverage
- **β=2.0**: Strong recall focus, good for comprehensiveness evaluation

## Architecture

```
rouge/
├── rouge_metrics.py               # Main evaluator class
├── __init__.py                    # Package exports  
├── README.md                      # This documentation
├── rouge_evaluation_summary.json # Historical results
└── tests/
    ├── comprehensive_rouge_test.py  # Full test suite
    └── test_rouge.py               # Integration tests
```

## Advantages Over Other Metrics

### vs BLEU
- **BLEU**: Requires exact n-gram matches, precision-focused
- **ROUGE-L**: Allows flexible ordering, recall-focused
- **Use Case**: ROUGE better for content coverage, BLEU for fluency

### vs BERTScore  
- **BERTScore**: Semantic similarity using large neural models
- **ROUGE-L**: Lexical similarity using lightweight algorithm
- **Use Case**: ROUGE faster and more interpretable

### vs Exact Match
- **Exact Match**: Requires identical text
- **ROUGE-L**: Handles paraphrasing and restructuring
- **Use Case**: ROUGE better for natural language variation

## Medical Text Advantages

### Clinical Terminology Preservation
- Captures medical terms even when sentence structure changes
- Good for evaluating content accuracy in clinical reports
- Handles abbreviation variations and synonyms well

### Paraphrasing Robustness
```
Reference: "Bilateral infiltrates consistent with pulmonary edema are present"
Generated: "There are bilateral infiltrates suggesting pulmonary edema"
ROUGE-L: High score due to preserved medical content
```

## Development

### Adding New Features

1. **Algorithm Variants**: Could add ROUGE-1, ROUGE-2 for n-gram versions
2. **Preprocessing**: Add medical text normalization
3. **Optimization**: Implement space-optimized LCS for very long texts
4. **Multi-language**: Extend tokenization for other languages

### Testing New Features

1. Add test cases to `test_rouge.py` for integration testing
2. Update `comprehensive_rouge_test.py` for evaluation
3. Run the full test suite to ensure compatibility

## Troubleshooting

### Import Errors
```python
# Check if the package is properly installed
import sys
sys.path.append('/path/to/CXR-Report-Metric')
from CXRMetric.metrics.rouge import ROUGEEvaluator
```

### Performance Issues
- **Large Texts**: Consider text truncation for very long reports
- **Batch Processing**: Process multiple reports in batches
- **Memory**: Algorithm uses O(m×n) space - monitor for very long texts

### Score Interpretation
- **Low Scores**: May indicate significant paraphrasing or content changes
- **High Scores**: Indicate good content preservation
- **Perfect Scores**: Usually indicate identical or near-identical texts

## Contributing

When contributing to ROUGE metrics:

1. **Maintain Compatibility**: Ensure changes work with existing interface
2. **Add Tests**: Include both unit and integration tests  
3. **Document Changes**: Update README and docstrings
4. **Performance**: Consider impact on evaluation speed
5. **Medical Focus**: Validate against medical text requirements

---

For questions or issues with ROUGE metrics, please refer to the main project documentation or create an issue in the project repository.
