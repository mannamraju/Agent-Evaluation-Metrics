# Perplexity Metrics

This folder contains perplexity and cross-entropy loss evaluation metrics for CXR (Chest X-Ray) report quality assessment.

## Overview

Perplexity is a measurement of how well a probability model predicts text. It uses pretrained language models to evaluate the fluency and naturalness of generated text by measuring the model's confidence in predicting each token given the context.

## Algorithm

### Perplexity Computation
- **Purpose**: Measures text fluency and naturalness from a language model perspective
- **Method**: `perplexity = exp(cross_entropy_loss)` where loss is negative log-likelihood
- **Advantage**: Provides insight into text quality that complements semantic similarity metrics
- **Model Architecture**: Uses autoregressive causal language models (GPT family)

### Cross-Entropy Loss
- **Purpose**: Direct measure of model's predictive confidence
- **Method**: `-log(P(token|context))` averaged over all tokens
- **Range**: Lower values indicate higher confidence (typical range: 2-5 for medical text)
- **Interpretation**: Raw confidence score before exponential transformation

## Files

### Core Implementation
- `perplexity_metrics.py` - Main PerplexityEvaluator class with model integration
- `__init__.py` - Package initialization and exports

### Testing Suite
- `tests/comprehensive_perplexity_test.py` - Complete testing and evaluation suite
- `tests/test_perplexity.py` - Integration tests and edge case handling
- `perplexity_evaluation_summary.json` - Timestamped evaluation history

## Usage

### Basic Usage

```python
from CXRMetric.metrics.perplexity import PerplexityEvaluator
import pandas as pd

# Initialize evaluator
evaluator = PerplexityEvaluator(model_name="distilgpt2")  # Default model

# Prepare your data. gt_data is Ground Truth data 
gt_data = pd.DataFrame({
    'study_id': [1, 2, 3],
    'report': [
        "The chest X-ray shows clear lungs with no acute findings.",
        "There is evidence of pneumonia in the right lower lobe.", 
        "Cardiac silhouette is enlarged consistent with cardiomegaly."
    ]
})

#pred_data is Predicted data
pred_data = pd.DataFrame({
    'study_id': [1, 2, 3],
    'report': [
        "Chest radiograph demonstrates clear lung fields without abnormalities.",
        "Right lower lobe pneumonia is present with inflammatory changes.",
        "The heart size is increased suggesting cardiac enlargement."
    ]
})

# Compute perplexity scores
results = evaluator.compute_metric(gt_data, pred_data)
print(f"Generated perplexity: {results['perplexity_generated'].mean():.2f}")
print(f"Reference perplexity: {results['perplexity_reference'].mean():.2f}")
```

### Model Configuration

```python
# Different model options
evaluator_small = PerplexityEvaluator(model_name="distilgpt2")    # Fast, smaller model
evaluator_standard = PerplexityEvaluator(model_name="gpt2")       # Standard GPT-2
evaluator_custom = PerplexityEvaluator(
    model_name="microsoft/DialoGPT-medium",  # Conversation-focused
    batch_size=4,          # Adjust based on available memory
    max_length=512,        # Maximum sequence length
    stride=256             # Overlap for sliding window
)
```

### Advanced Configuration

```python
# GPU/CPU configuration
evaluator_gpu = PerplexityEvaluator(
    model_name="gpt2",
    device="cuda",         # Force GPU usage
    batch_size=8           # Larger batch size for GPU
)

evaluator_cpu = PerplexityEvaluator(
    model_name="distilgpt2",
    device="cpu",          # Force CPU usage
    batch_size=2           # Smaller batch size for CPU
)
```

### Comprehensive Testing

```bash
# Run comprehensive test suite
cd /path/to/CXR-Report-Metric
python CXRMetric/metrics/perplexity/tests/comprehensive_perplexity_test.py

# Run integration tests
python CXRMetric/metrics/perplexity/tests/test_perplexity.py
```

## Features

### Multiple Model Support
- **DistilGPT-2**: Faster, smaller model good for quick evaluation
- **GPT-2**: Standard model with better language understanding
- **Custom Models**: Support for any HuggingFace causal language model
- **Automatic Fallback**: Mock implementation when models unavailable

### Efficient Processing
- **Sliding Window**: Handles long sequences beyond model limits
- **Batch Processing**: Process multiple texts simultaneously
- **GPU Acceleration**: Automatic CUDA utilization when available
- **Memory Management**: Configurable batch sizes for different hardware

### Comprehensive Metrics
- **Generated Text Perplexity**: Fluency of model-generated reports
- **Reference Text Perplexity**: Baseline fluency of ground truth
- **Perplexity Ratio**: Relative fluency comparison (Generated/Reference)
- **Cross-Entropy Difference**: Raw confidence comparison
- **Token Counts**: Text length analysis

### Medical Text Optimization
- Handles medical terminology and technical language
- Appropriate perplexity ranges for clinical text (10-100 typical)
- Robust to varied sentence structures in radiology reports
- Complementary analysis to semantic similarity metrics

## Algorithm Details

### Model Architecture
Perplexity evaluation uses autoregressive causal language models:

1. **Tokenization**: Text split into subword tokens using model tokenizer
2. **Context Processing**: Each token predicted given all previous tokens
3. **Loss Computation**: Negative log-likelihood of true tokens
4. **Perplexity Calculation**: Exponential of average loss

### Sliding Window Processing
For sequences longer than model limits:

```
Text: [token1, token2, ..., token1000]
Window 1: tokens 1-512    -> loss1
Window 2: tokens 257-768  -> loss2  (256 token stride)
Window 3: tokens 513-1000 -> loss3
Final: weighted average of losses
```

### Score Interpretation
```python
# Perplexity ranges for medical text
perplexity < 20:     Very fluent, natural text
perplexity 20-50:    Good fluency, typical medical reports  
perplexity 50-100:   Acceptable fluency, some awkward phrasing
perplexity > 100:    Poor fluency, unnatural or garbled text

# Cross-entropy ranges
cross_entropy < 3:   High model confidence
cross_entropy 3-4:   Moderate confidence (typical)
cross_entropy > 4:   Low confidence, unexpected word choices

# Ratio interpretation
ratio ~1.0:          Generated text has similar fluency to reference
ratio < 1.0:         Generated text more fluent than reference
ratio > 1.0:         Generated text less fluent than reference
```

## Performance Characteristics

### Model Comparison
| Model | Size | Speed | Quality | Memory |
|-------|------|-------|---------|--------|
| DistilGPT-2 | 82M | Fast | Good | ~250MB |
| GPT-2 Small | 124M | Medium | Better | ~500MB |
| GPT-2 Medium | 355M | Slow | Best | ~1.4GB |

### Hardware Requirements
- **CPU Only**: Works with any model, slower processing
- **GPU Recommended**: 4GB+ VRAM for GPT-2, 2GB+ for DistilGPT-2
- **Memory**: Model size + batch processing overhead
- **Processing Speed**: 0.1-5 seconds per report (hardware dependent)

### Optimization Tips
```python
# For speed (CPU environments)
evaluator = PerplexityEvaluator(
    model_name="distilgpt2",
    batch_size=1,
    max_length=256
)

# For accuracy (GPU environments)  
evaluator = PerplexityEvaluator(
    model_name="gpt2",
    batch_size=8,
    max_length=512,
    device="cuda"
)
```

## Architecture

```
perplexity/
├── perplexity_metrics.py              # Main evaluator class
├── __init__.py                         # Package exports
├── README.md                          # This documentation  
├── perplexity_evaluation_summary.json # Historical results
└── tests/
    ├── comprehensive_perplexity_test.py # Full test suite
    └── test_perplexity.py              # Integration tests
```

## Advantages Over Other Metrics

### vs Exact Matching (BLEU)
- **BLEU**: Requires exact n-gram matches, focuses on precision
- **Perplexity**: Evaluates overall fluency and naturalness
- **Use Case**: Perplexity better for assessing readability and flow

### vs Semantic Similarity (BERTScore)
- **BERTScore**: Measures meaning preservation using embeddings
- **Perplexity**: Measures text quality from language model perspective  
- **Use Case**: Perplexity complements semantic metrics with fluency assessment

### vs Sequence Matching (ROUGE)
- **ROUGE**: Measures content overlap with flexible word order
- **Perplexity**: Measures text naturalness independent of reference
- **Use Case**: Perplexity good for absolute quality assessment

### vs Human Evaluation
- **Human Evaluation**: Subjective, expensive, slow
- **Perplexity**: Objective, automatic, fast
- **Use Case**: Perplexity provides scalable proxy for text quality

## Medical Text Applications

### Clinical Report Assessment
```python
# Example: Evaluating report fluency
medical_reports = [
    "Patient presents with acute dyspnea and chest pain.",           # Natural
    "Presenting patient acute dyspnea with and chest pain.",        # Awkward  
    "The patient acute presents dyspnea chest pain with.",          # Unnatural
]

for report in medical_reports:
    result = evaluator._compute_perplexity_single(report)
    print(f"Report: '{report}'")
    print(f"Perplexity: {result['perplexity']:.2f}")
```

### Quality Benchmarking
- Compare different text generation models
- Track improvement over training iterations
- Evaluate post-processing effects on fluency
- Assess domain adaptation effectiveness

### Error Detection
```python
# High perplexity may indicate issues
if perplexity > 100:
    print("⚠️ Potential issues: grammatical errors, unnatural phrasing")
elif perplexity > 50:
    print("⚠️ Review recommended: somewhat awkward phrasing")  
else:
    print("✅ Good fluency: natural, readable text")
```

## Development

### Adding New Models

```python
# Test custom model compatibility
evaluator = PerplexityEvaluator(model_name="your-custom-model")

# Requirements for custom models:
# 1. HuggingFace AutoModelForCausalLM compatibility
# 2. Appropriate tokenizer available
# 3. Reasonable context length for medical text
```

### Extending Functionality

1. **Multi-language Support**: Add tokenizers for other languages
2. **Domain Adaptation**: Fine-tune models on medical text
3. **Ensemble Methods**: Combine multiple model perspectives
4. **Custom Metrics**: Add domain-specific perplexity variants

### Performance Monitoring

```python
# Track evaluation performance
results = evaluator.get_summary_stats(predictions_df)
performance_info = results['perplexity_analysis']

print(f"Model: {performance_info['model_name']}")
print(f"Device: {performance_info['device']}")
print(f"Avg Generated PPL: {performance_info['generated_analysis']['median_perplexity']}")
```

## Troubleshooting

### Installation Issues
```bash
# Install required dependencies
pip install transformers torch

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
```python
# Reduce memory usage
evaluator = PerplexityEvaluator(
    model_name="distilgpt2",  # Smaller model
    batch_size=1,             # Reduce batch size
    max_length=256            # Shorter sequences
)
```

### Model Loading Issues
```python
# Check model availability
try:
    evaluator = PerplexityEvaluator(model_name="gpt2")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    # Falls back to mock implementation
```

### Performance Issues
- **Slow Processing**: Use DistilGPT-2, reduce batch size, shorter max_length
- **High Memory**: Reduce batch size, use CPU instead of GPU
- **Low Scores**: Check for very technical/specialized text that may be unfamiliar to general language models

## Contributing

When contributing to perplexity metrics:

1. **Model Compatibility**: Ensure new models work with existing interface
2. **Medical Validation**: Test on medical text specifically
3. **Performance Testing**: Benchmark speed and memory usage
4. **Documentation**: Update README and docstrings
5. **Error Handling**: Robust fallbacks for model loading issues

---

For questions or issues with perplexity metrics, please refer to the main project documentation or create an issue in the project repository.
