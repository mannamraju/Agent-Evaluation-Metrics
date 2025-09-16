# Setting up Semantic Embedding Evaluation (CheXbert)

The semantic embedding evaluation is a crucial metric that measures semantic similarity between radiology reports using the CheXbert model. This metric provides domain-specific understanding of medical content.

## Why Semantic Embedding is Important

- **Medical Domain Understanding**: CheXbert is trained specifically on chest X-ray reports
- **Semantic Similarity**: Captures meaning beyond word overlap (unlike BLEU/ROUGE)
- **Clinical Accuracy**: Correlates well with radiologist assessments
- **Content Preservation**: Measures how well clinical information is preserved

## Setup Instructions

### 1. Download CheXbert Model

The CheXbert model checkpoint is required but not included due to size (~500MB). You can obtain it from:

**Option A: Official CheXbert Repository**
```bash
# Clone the official CheXbert repository
git clone https://github.com/stanfordmlgroup/CheXbert.git
# The model file will be at: CheXbert/chexbert.pth
```

**Option B: Direct Download** (if available)
```bash
# Download directly to the correct location
wget -O CXRMetric/CheXbert/chexbert.pth [MODEL_URL]
```

### 2. Update Configuration

Update `config.py` with the correct path:

```python
# Update this path to point to your CheXbert model
CHEXBERT_PATH = "CXRMetric/CheXbert/chexbert.pth"  # or your custom path
```

### 3. Install Additional Dependencies

Some CheXbert dependencies may not be installed:

```bash
# Install in your virtual environment
pip install statsmodels transformers
```

### 4. Verify Setup

Test that semantic embedding evaluation works:

```python
from CXRMetric.modular_evaluation import evaluate_reports

# Test with semantic embedding
results, summary = evaluate_reports(
    'reports/gt_reports.csv',
    'reports/predicted_reports.csv', 
    metrics=['semantic_embedding']
)

print("Semantic embedding scores:", results['semb_score'].mean())
```

## Understanding Semantic Embedding Scores

- **Range**: [-1, 1] where 1 indicates identical semantic content
- **Interpretation**:
  - > 0.8: Very high semantic similarity
  - 0.6-0.8: Good semantic preservation
  - 0.4-0.6: Moderate similarity
  - < 0.4: Poor semantic alignment

## Troubleshooting

### Common Issues

1. **"CheXbert model not found"**
   - Download the model checkpoint as described above
   - Update `config.py` with correct path

2. **"ModuleNotFoundError: No module named 'statsmodels'"**
   ```bash
   pip install statsmodels
   ```

3. **"PyTorch not available"** (VS Code only)
   - This is just a linting issue in VS Code
   - Code will run correctly at runtime
   - Ensure VS Code is using the correct Python interpreter

4. **Zero Scores**
   - Check that model path is correct
   - Verify model file is not corrupted
   - Ensure input reports have meaningful content

## Alternative: Skip Semantic Embedding

If you cannot set up CheXbert but want to run other metrics:

```python
# Run evaluation without semantic embedding
results, summary = evaluate_reports(
    'gt.csv', 'pred.csv',
    metrics=['bleu', 'rouge', 'bertscore', 'radgraph']  # Exclude semantic_embedding
)
```

## Performance Notes

- First run will be slower (model loading + embedding generation)
- Embeddings are cached for subsequent runs
- GPU acceleration recommended for large datasets
- Memory usage: ~2GB for model + embeddings

## Integration with Azure

The semantic embedding evaluation is fully supported in Azure deployments:

```bash
# Azure deployment includes CheXbert setup (use the script under azure_deployment)
azure_deployment/deploy-aci.ps1 -VMType gpu
```

For Azure deployments, ensure the CheXbert model is included in your container image or mounted as a volume.
