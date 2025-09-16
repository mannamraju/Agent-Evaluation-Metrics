# Azure Deployment

This folder contains all files and scripts related to deploying the CXR-Report-Metric to Azure cloud services.

## Files Overview

### Documentation
- **`AZURE_DEPLOYMENT.md`** - Comprehensive Azure deployment guide
- **`AZURE_GPU_REQUIREMENTS.md`** - GPU requirements and configuration for Azure

### Configuration Files
- **`Dockerfile`** - Docker containerization configuration
- **`Procfile`** - Process file for Heroku/Azure App Service
- **`requirements-azure.txt`** - Python dependencies specific to Azure deployment

### Deployment Scripts
- **`deploy-aci.ps1`** - PowerShell script for Azure Container Instances deployment
- **`azure_ml_deploy.py`** - Azure Machine Learning deployment utilities
- **`azure_batch_processor.py`** - Batch processing utilities for Azure
- **`azure_gpu_config.py`** - GPU configuration for Azure deployments

## Deployment Options

### 1. Azure Container Instances (ACI)
```powershell
# Deploy to ACI using PowerShell
.\deploy-aci.ps1
```

### 2. Azure Machine Learning
```bash
# Deploy to Azure ML
python azure_ml_deploy.py
```

### 3. Azure App Service (Docker)
```bash
# Build and deploy Docker container
docker build -t cxr-metrics .
# Deploy to Azure App Service
```

## Prerequisites

- Azure CLI installed and configured
- Docker (for containerized deployments)
- Azure subscription with appropriate permissions
- Python environment with Azure SDK

## Configuration

1. Set up Azure credentials:
   ```bash
   az login
   az account set --subscription YOUR_SUBSCRIPTION_ID
   ```

2. Configure environment variables:
   ```bash
   export AZURE_RESOURCE_GROUP=your-resource-group
   export AZURE_REGION=eastus
   ```

3. Install Azure-specific dependencies:
   ```bash
   pip install -r requirements-azure.txt
   ```

## GPU Support

For GPU-accelerated deployments, refer to:
- `AZURE_GPU_REQUIREMENTS.md` for detailed GPU requirements
- `azure_gpu_config.py` for GPU configuration scripts

## Monitoring and Scaling

- Use Azure Monitor for application insights
- Configure auto-scaling based on demand
- Set up health checks and alerts

## Cost Optimization

- Use Azure Spot Instances for non-critical workloads
- Implement automatic shutdown for development environments
- Monitor resource usage and optimize accordingly
