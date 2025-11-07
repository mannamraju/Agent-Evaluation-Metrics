#!/bin/bash
# Environment setup script for Agent-Evaluation-Metrics

echo "Activating Python virtual environment..."
source venv/bin/activate

echo "Python environment ready!"
echo "Python version: $(python --version)"
echo "Virtual environment: $VIRTUAL_ENV"

echo ""
echo "Available evaluation metrics:"
echo "- BLEU-4: N-gram overlap for report generation quality"
echo "- BERTScore: Semantic similarity using contextualized embeddings"
echo "- ROUGE-L: Longest common subsequence overlap"
echo "- Composite RadCliQ: Specialized radiology report quality metric"
echo ""
echo "Example usage:"
echo "python -c \"from CXRMetric.modular_evaluation import ModularEvaluationRunner; print('Ready to run evaluations!')\""