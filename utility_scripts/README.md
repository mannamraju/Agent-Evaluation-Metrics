# Utility Scripts

This folder contains various utility scripts for the CXR-Report-Metric project.

## Scripts Overview

### Core Evaluation Scripts
- **`test_metric.py`** - Main testing script for CXR report metrics
- **`evaluate_modular.py`** - Modular evaluation framework for different metrics
- **`examples_modular.py`** - Example usage demonstrations for modular metrics

### Model Access and Information
- **`azure_model_access.py`** - Guide for accessing Azure AI models

### API and Server Scripts
- **`api_server.py`** - REST API server for CXR metrics evaluation

### GPU and Model Testing
- **`gpu_test.py`** - GPU performance testing for perplexity metrics
- **`medical_model_alternatives.py`** - Information about local medical imaging models
- **`azure_model_access.py`** - Guide for accessing Azure AI models

## Usage

### Running Tests
```bash
# Run main metric tests
python utility_scripts/test_metric.py

# Run modular evaluation
python utility_scripts/evaluate_modular.py

# Test GPU performance
python utility_scripts/gpu_test.py
```

### Azure Deployment
```powershell
# Deploy to Azure Container Instances (scripts moved to azure_deployment/)
# See azure_deployment/ folder for deployment scripts and documentation
```

### API Server
```bash
# Start the REST API server
python utility_scripts/api_server.py
```

## Dependencies

Most scripts require the main project dependencies. Additional requirements may include:
- Flask/FastAPI (for API server)
- PyTorch with CUDA (for GPU testing)
- Azure deployment dependencies are in azure_deployment/ folder

## Notes

- Make sure to adjust import paths when running scripts from the utility_scripts folder
- Some scripts may need environment variables or configuration files
- Azure scripts require proper Azure authentication and subscriptions
