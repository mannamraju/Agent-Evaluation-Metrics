# Environment Setup Guide

This guide will help you set up the development environment for the Agent-Evaluation-Metrics repository.

## Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- Git

## Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/mannamraju/Agent-Evaluation-Metrics.git
cd Agent-Evaluation-Metrics
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements-updated.txt
```

### 4. Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"
```

### 5. Test Installation
```bash
python test_environment.py
```

## Using the Activation Script

For convenience, you can use the provided activation script:

```bash
chmod +x activate_env.sh
./activate_env.sh
```

This script will:
- Activate the virtual environment
- Display Python and package versions
- Show you're ready to work

## Key Dependencies

- **PyTorch**: CPU version for deep learning models
- **Transformers**: Hugging Face transformers library
- **BERTScore**: For semantic similarity evaluation
- **ROUGE**: For text summarization evaluation
- **NLTK**: Natural language processing toolkit
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation and analysis

## Hardware Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for dependencies and models
- **GPU**: Optional (CPU version of PyTorch is installed)

## Troubleshooting

### Common Issues

1. **Permission denied when installing packages**:
   ```bash
   pip install --user -r requirements-updated.txt
   ```

2. **NLTK download issues**:
   ```bash
   python -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; import nltk; nltk.download('punkt')"
   ```

3. **Memory issues during installation**:
   ```bash
   pip install --no-cache-dir -r requirements-updated.txt
   ```

### Environment Verification

To verify your environment is working correctly:

```bash
python -c "
import torch
import transformers
import bert_score
import nltk
import pandas as pd
from CXRMetric.modular_evaluation import ModularEvaluationRunner
print('✓ All dependencies imported successfully')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ Transformers version: {transformers.__version__}')
"
```

## Development Workflow

1. Always activate the virtual environment before working:
   ```bash
   source venv/bin/activate
   ```

2. Install new dependencies:
   ```bash
   pip install <package-name>
   pip freeze > requirements-updated.txt
   ```

3. Run tests:
   ```bash
   python test_environment.py
   ```

4. Deactivate when done:
   ```bash
   deactivate
   ```

## Important Notes

- The `venv/` directory is ignored by git and should not be committed
- Use `requirements-updated.txt` for the latest compatible dependencies
- The original `requirements.txt` contains outdated PyTorch versions
- All evaluation metrics work with CPU-only PyTorch installation

## Need Help?

If you encounter issues:
1. Check this guide first
2. Verify your Python version: `python --version`
3. Check if you're in the virtual environment: `which python`
4. Try reinstalling dependencies: `pip install --force-reinstall -r requirements-updated.txt`