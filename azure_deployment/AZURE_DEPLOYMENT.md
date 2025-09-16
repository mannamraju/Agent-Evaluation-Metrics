# CXR-Report-Metric Azure Deployment Guide

## 🔥 GPU-Optimized Azure Deployment
This guide covers deploying the **GPU-optimized** CXR-Report-Metric system to Azure Cloud with comprehensive performance optimization for medical report evaluation.

## ⚡ Quick Start - GPU Deployment

```powershell
# Deploy GPU-optimized container (Recommended)
./deploy-aci.ps1 -VMType gpu

# Expected: Tesla V100 (16GB), ~$1.80/hour, 10-25 reports/second
```

## 📊 GPU Requirements Summary

| **Tier** | **GPU** | **Memory** | **Azure VM** | **Cost/Hr** | **Performance** |
|-----------|---------|------------|--------------|-------------|-----------------|
| **Minimum** | Tesla K80 | 8GB | `Standard_NC6` | $0.90 | 2-5 reports/sec |
| **Recommended** | Tesla V100 | 16GB | `Standard_NC6s_v3` | $1.80 | 10-25 reports/sec |
| **High-Perf** | 2x V100 | 32GB | `Standard_NC12s_v3` | $3.60 | 25-50 reports/sec |
| **Maximum** | A100 | 40GB+ | `Standard_NC24ads_A100_v4` | $27.20 | 50-100+ reports/sec |

## 🎯 Key Optimizations Implemented

### 1. **Automatic GPU Detection & Optimization**
- Auto-detects available GPU resources
- Dynamically adjusts batch sizes based on GPU memory
- Intelligent memory management with OOM recovery

### 2. **Medical Domain Models**
- **BERTScore**: PubMedBERT for medical text understanding
- **Embedding Models**: Medical-domain optimized transformers
- **Model Caching**: Pre-downloaded models for faster startup

### 3. **Performance Features**
- **Mixed Precision**: 2x memory efficiency with float16
- **Batch Processing**: Optimized batching for GPU utilization  
- **Memory Cleanup**: Automatic GPU memory management
- **Smart Caching**: Intelligent result and model caching

### 4. **Azure-Specific Optimizations**
- **Container Optimization**: Multi-stage builds with model pre-caching
- **Health Checks**: Kubernetes/ACI-compatible endpoints
- **Auto-scaling**: Memory and GPU-aware scaling policies
- **Monitoring**: Built-in performance and resource monitoring

## Prerequisites
- Azure subscription
- Azure CLI installed
- Docker installed (for containerization)
- Python 3.7+ environment

## Deployment Options

### 1. Azure Container Instances (Quick Start)

**Best for**: Development, testing, small-scale evaluation

```powershell
# Run the deployment script
.\deploy-aci.ps1
```

**Manual steps**:
```bash
# Build and push image
az acr build --registry cxrmetricregistry --image cxr-metric:latest .

# Deploy container
az container create \
    --resource-group cxr-metric-rg \
    --name cxr-report-metric \
    --image cxrmetricregistry.azurecr.io/cxr-metric:latest \
    --cpu 4 --memory 8 \
    --ports 8000 \
    --environment-variables PYTHONPATH=/app
```

### 2. Azure App Service (Web API)

**Best for**: Production web API, REST endpoint access

```bash
# Deploy web service
az webapp create --resource-group cxr-metric-rg --plan cxr-metric-plan --name cxr-metric-api --deployment-container-image-name cxrmetricregistry.azurecr.io/cxr-metric:latest

# Configure app settings
az webapp config appsettings set --resource-group cxr-metric-rg --name cxr-metric-api --settings PYTHONPATH=/app PORT=8000
```

**Access**: `https://cxr-metric-api.azurewebsites.net/`

### 3. Azure Machine Learning

**Best for**: ML experiments, model management, MLOps

```python
# Run Azure ML deployment
python azure_ml_deploy.py
```

Features:
- Model versioning and tracking
- Automated scaling
- A/B testing capabilities
- Integration with Azure ML Studio

### 4. Azure Batch (Large-scale Processing)

**Best for**: Processing thousands of reports in parallel

```python
# Run batch processing
python azure_batch_processor.py
```

Features:
- Parallel processing of large datasets
- Auto-scaling compute nodes
- Cost-effective for batch workloads
- Integration with Azure Storage

### 5. Azure Kubernetes Service (AKS)

**Best for**: Production workloads, microservices, high availability

```yaml
# Apply Kubernetes configuration
kubectl apply -f k8s-deployment.yaml
```

## Model Files and Storage

### Azure Blob Storage Integration

```python
# Configure model storage
CHEXBERT_PATH = "https://yourstorageaccount.blob.core.windows.net/models/chexbert.pth"
RADGRAPH_PATH = "https://yourstorageaccount.blob.core.windows.net/models/radgraph_model.tar.gz"
```

### Model Download on Startup

```python
# Add to Dockerfile or startup script
RUN python -c "
from azure.storage.blob import BlobServiceClient
import os

# Download models from Azure Blob Storage
blob_client = BlobServiceClient(account_url='https://youraccount.blob.core.windows.net')
blob_client.download_blob('models/chexbert.pth').download_to_stream(open('/app/models/chexbert.pth', 'wb'))
"
```

## Configuration Management

### Environment Variables
Set these in your Azure service:

```bash
PYTHONPATH=/app
CHEXBERT_PATH=/app/models/chexbert.pth
RADGRAPH_PATH=/app/models/radgraph_model.tar.gz
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_KEYVAULT_URL=https://your-keyvault.vault.azure.net/
```

### Azure Key Vault Integration
```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Access secrets
credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://your-keyvault.vault.azure.net/", credential=credential)
storage_key = client.get_secret("storage-account-key").value
```

## Performance Optimization

### GPU Support
For CUDA-enabled models:

```dockerfile
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
# Use GPU-enabled base image
```

Azure VM sizes with GPU:
- `Standard_NC6` (1 GPU, 6 cores, 56GB RAM)
- `Standard_NC12` (2 GPUs, 12 cores, 112GB RAM)
- `Standard_ND6s` (1 Tesla P40, 6 cores, 112GB RAM)

### Scaling Configuration

**Container Instances**:
```bash
--cpu 8 --memory 16  # Scale up resources
```

**App Service**:
```bash
az appservice plan update --sku P3V2  # Premium tier with more resources
```

**AML Compute**:
```python
compute_target = ComputeTarget.create(
    workspace=ws,
    name="gpu-cluster",
    provisioning_configuration=AmlCompute.provisioning_configuration(
        vm_size="Standard_NC6",
        min_nodes=0,
        max_nodes=10,
        idle_seconds_before_scaledown=300
    )
)
```

## Monitoring and Logging

### Application Insights
```python
from applicationinsights import TelemetryClient
tc = TelemetryClient(os.getenv('APPINSIGHTS_INSTRUMENTATION_KEY'))

# Log evaluation metrics
tc.track_metric('evaluation_time', evaluation_duration)
tc.track_metric('reports_processed', num_reports)
```

### Azure Monitor Integration
```bash
# Enable diagnostic logs
az monitor diagnostic-settings create \
    --resource cxr-metric-api \
    --logs '[{"category": "AppServiceHTTPLogs", "enabled": true}]' \
    --metrics '[{"category": "AllMetrics", "enabled": true}]'
```

## Cost Optimization

### Resource Tiers by Use Case

| Use Case | Service | Configuration | Est. Monthly Cost |
|----------|---------|--------------|-------------------|
| Development | Container Instances | 2 vCPU, 4GB RAM | $50-100 |
| Small Production | App Service | P1V2 tier | $150-300 |
| ML Experiments | Azure ML | Pay-per-use compute | $200-500 |
| Large Batch | Azure Batch | Auto-scaling nodes | $100-1000 |
| Enterprise | AKS | 3-node cluster | $500-1500 |

### Auto-scaling Policies
```bash
# App Service auto-scaling
az monitor autoscale create \
    --resource-group cxr-metric-rg \
    --resource cxr-metric-api \
    --min-count 1 \
    --max-count 10 \
    --count 2
```

## Security Considerations

### Network Security
- Use Azure Private Endpoints for database connections
- Configure Network Security Groups (NSG)
- Enable Azure DDoS protection

### Authentication & Authorization
```python
from azure.identity import DefaultAzureCredential
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

# Add authentication to API endpoints
security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Verify Azure AD token
    pass
```

### Data Privacy
- Use Azure Private Link for storage access
- Enable encryption at rest and in transit
- Implement HIPAA compliance if processing medical data

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase container memory or use batch processing
2. **Model Loading Errors**: Ensure model files are accessible and paths are correct
3. **Timeout Issues**: Increase timeout settings for large evaluations
4. **CUDA Errors**: Verify GPU availability and CUDA compatibility

### Debugging Commands
```bash
# Check container logs
az container logs --resource-group cxr-metric-rg --name cxr-report-metric

# Monitor App Service logs
az webapp log tail --resource-group cxr-metric-rg --name cxr-metric-api

# Check resource usage
az monitor metrics list --resource /subscriptions/your-sub/resourceGroups/cxr-metric-rg/providers/Microsoft.ContainerInstance/containerGroups/cxr-report-metric
```

## Support and Maintenance

### Automated Updates
```yaml
# GitHub Actions for CI/CD
name: Deploy to Azure
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Build and deploy
      run: |
        az acr build --registry cxrmetricregistry --image cxr-metric:latest .
        az container restart --resource-group cxr-metric-rg --name cxr-report-metric
```

### Health Checks
The API includes health check endpoints:
- `/health` - Basic health status
- `/ready` - Readiness check for load balancers
- `/metrics` - Available evaluation metrics

---

## Quick Start Commands

```bash
# 1. Build and deploy to Azure Container Instances
git clone your-repo
cd CXR-Report-Metric
az group create --name cxr-metric-rg --location eastus
az acr create --resource-group cxr-metric-rg --name cxrmetricregistry --sku Basic
az acr build --registry cxrmetricregistry --image cxr-metric:latest .
.\deploy-aci.ps1

# 2. Access your deployed API
curl https://cxr-metric-api.eastus.azurecontainer.io:8000/health
```

**Your CXR Report Metric system is now running in Azure!** 🚀
