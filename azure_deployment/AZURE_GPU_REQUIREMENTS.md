# CXR-Report-Metric: Azure GPU Requirements & Performance Guide

## 🔥 GPU Requirements for Efficient Evaluation Pipeline

### Minimum GPU Specifications

| Performance Tier | GPU Model | Memory | Azure VM Types | Est. Cost/Hour | Use Case |
|------------------|-----------|---------|---------------|----------------|----------|
| **Basic** | Tesla K80 | 8GB | `Standard_NC6` | $0.90 | Development/Testing |
| **Recommended** | Tesla V100 | 16GB | `Standard_NC6s_v3` | $1.80 | Production |
| **High-Performance** | Tesla V100 (2x) | 32GB | `Standard_NC12s_v3` | $3.60 | Batch Processing |
| **Maximum** | A100 (40GB) | 40GB+ | `Standard_NC24ads_A100_v4` | $27.20 | Research/Large-scale |

### 📊 Performance Benchmarks by GPU Tier

#### Basic Tier (Tesla K80, 8GB VRAM)
```
Reports/Second: 2-5
Batch Size: 4-8
Metrics Supported: All (with reduced batch sizes)
Limitations: Slower processing, may need CPU fallback for large models
```

#### Recommended Tier (Tesla V100, 16GB VRAM)
```
Reports/Second: 10-25
Batch Size: 16-32  
Metrics Supported: All metrics at optimal speed
Sweet Spot: Best price/performance ratio
```

#### High-Performance Tier (Dual V100, 32GB VRAM)
```
Reports/Second: 25-50
Batch Size: 32-64
Concurrent Processing: Multiple metric types simultaneously
Ideal for: Batch processing 1000+ reports
```

#### Maximum Tier (A100, 40GB+ VRAM)
```
Reports/Second: 50-100+
Batch Size: 64-128
Advanced Features: Mixed precision, tensor cores
Best for: Research, real-time processing
```

## 🛠️ Model-Specific GPU Requirements

### BERTScore Evaluation
- **Minimum**: 6GB VRAM (batch_size=8)
- **Recommended**: 12GB VRAM (batch_size=24)
- **Optimal**: 16GB+ VRAM (batch_size=32+)
- **Model**: PubMedBERT for medical domain

### RadGraph Evaluation  
- **Minimum**: 8GB VRAM (batch_size=4)
- **Recommended**: 16GB VRAM (batch_size=12)
- **Notes**: More memory-intensive due to graph processing

### CheXbert Evaluation
- **Minimum**: 4GB VRAM (batch_size=12)
- **Recommended**: 8GB VRAM (batch_size=32)
- **Notes**: Lighter model, good GPU utilization

### Semantic Embedding
- **Minimum**: 6GB VRAM (batch_size=16)
- **Recommended**: 12GB VRAM (batch_size=48)
- **Notes**: Can benefit from larger batch sizes

## 🚀 Azure VM Selection Guide

### Development Environment
**Recommended**: `Standard_NC6s_v3`
- 1x Tesla V100 (16GB)
- 6 vCPUs, 56GB RAM
- $1.80/hour
- Perfect for development and moderate production loads

```bash
# Deploy development environment
./deploy-aci.ps1 -VMType gpu -ResourceGroup mydev-rg
```

### Production Environment  
**Recommended**: `Standard_NC12s_v3`
- 2x Tesla V100 (32GB total)
- 12 vCPUs, 224GB RAM  
- $3.60/hour
- High availability, can handle multiple concurrent requests

```bash
# Deploy production environment
./deploy-aci.ps1 -VMType high-performance -ResourceGroup myprod-rg
```

### Batch Processing
**Recommended**: `Standard_ND40rs_v2` 
- 8x Tesla V100 (128GB total)
- 40 vCPUs, 672GB RAM
- ~$14.40/hour
- Process thousands of reports in parallel

## 🔧 GPU Optimization Features

### Automatic Batch Size Optimization
Our system automatically adjusts batch sizes based on available GPU memory:

```python
from azure_gpu_config import AzureGPUConfig

config = AzureGPUConfig()
optimal_batch_size = config.get_optimal_batch_size("bert")  # Auto-calculated
```

### Memory Management
- **Automatic cleanup**: GPU memory cleared after each batch
- **OOM handling**: Automatic batch size reduction on memory errors
- **Mixed precision**: Uses float16 when possible for 2x memory efficiency

### Smart Caching
- **Model caching**: Pre-load models during container startup
- **Result caching**: Cache evaluation results to avoid recomputation
- **Persistent storage**: Models cached across container restarts

## 📈 Performance Optimization Tips

### 1. Batch Size Tuning
```python
# Optimal batch sizes by GPU memory
gpu_memory_gb = 16
if gpu_memory_gb >= 32:
    batch_size = 64  # High-end GPUs
elif gpu_memory_gb >= 16:
    batch_size = 32  # Standard production
elif gpu_memory_gb >= 8:
    batch_size = 16  # Basic tier
else:
    batch_size = 8   # Minimum viable
```

### 2. Multi-GPU Utilization
```python
# Automatically uses all available GPUs
evaluator = GPUOptimizedBERTScoreEvaluator(
    device="auto",  # Automatically selects best GPU
    batch_size="auto"  # Automatically optimizes batch size
)
```

### 3. Memory-Efficient Processing
- Process reports in streaming fashion for large datasets
- Use gradient checkpointing for very large models
- Enable CUDA memory mapping for efficient data loading

## 💰 Cost Optimization Strategies

### 1. Right-Sizing
- **Development**: Use `Standard_NC6s_v3` ($1.80/hr)
- **Production**: Scale up to `Standard_NC12s_v3` ($3.60/hr) only when needed
- **Batch Jobs**: Use `Standard_ND40rs_v2` ($14.40/hr) for large workloads

### 2. Auto-Scaling
```yaml
# Azure Container Apps auto-scaling
minReplicas: 0  # Scale to zero when idle
maxReplicas: 10
targetConcurrentRequests: 50
scaleRule: http
```

### 3. Spot Instances
- Use Azure Spot VMs for batch processing (up to 90% savings)
- Suitable for non-critical workloads
- Automatic retry on interruption

### 4. Reserved Instances
- Save 40-60% with 1-3 year commitments
- Ideal for consistent production workloads

## 🔍 Monitoring & Troubleshooting

### Performance Monitoring
```python
from azure_gpu_config import print_system_info

# Print comprehensive system information
system_info = print_system_info()
```

Output includes:
- GPU utilization and memory usage
- Batch size recommendations  
- Performance bottlenecks
- Cost optimization suggestions

### Common Issues & Solutions

#### Issue: Out of Memory (OOM)
```
Solution: Automatic batch size reduction implemented
- Starts with optimal batch size
- Reduces by 50% on OOM
- Continues until processing succeeds
```

#### Issue: Slow Model Loading
```
Solution: Model pre-caching during container startup
- Models downloaded during Docker build
- Cached in persistent storage
- Warm-up requests sent during initialization
```

#### Issue: Poor GPU Utilization
```
Solution: Batch size optimization
- Monitor GPU usage with nvidia-smi
- Increase batch size if memory allows
- Use multiple metrics concurrently
```

## 🏗️ Deployment Examples

### Basic GPU Deployment
```bash
# Deploy with automatic GPU optimization
./deploy-aci.ps1 -VMType gpu -Location "East US"

# Expected: ~$1.80/hour, 10-25 reports/second
```

### High-Performance Batch Processing
```bash
# Deploy for large-scale evaluation
./deploy-aci.ps1 -VMType high-performance -Replicas 3

# Expected: ~$10.80/hour total, 75-150 reports/second
```

### Cost-Optimized Development
```bash
# Deploy CPU-only for development
./deploy-aci.ps1 -VMType cpu

# Expected: ~$0.20/hour, 1-3 reports/second
```

## 🚦 Quick Start Commands

### 1. Deploy GPU-Optimized Container
```bash
# PowerShell
./deploy-aci.ps1 -VMType gpu

# Bash equivalent
./deploy.sh --vm-type gpu --region eastus
```

### 2. Test GPU Performance
```bash
curl -X POST http://your-endpoint:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "gt_reports": ["No acute findings."] * 100,
    "pred_reports": ["Heart and lungs normal."] * 100,
    "metrics": ["bertscore", "radgraph", "chexbert"]
  }'
```

### 3. Monitor Performance
```bash
# Check GPU usage
curl http://your-endpoint:8000/system-info

# Check processing stats  
curl http://your-endpoint:8000/performance-stats
```

## 📊 Expected Performance Metrics

### Single GPU (V100, 16GB)
- **BERTScore**: 20-30 reports/second
- **RadGraph**: 8-12 reports/second  
- **CheXbert**: 40-60 reports/second
- **Combined**: 6-10 reports/second (all metrics)

### Dual GPU (2x V100, 32GB)
- **BERTScore**: 40-60 reports/second
- **RadGraph**: 15-25 reports/second
- **CheXbert**: 80-120 reports/second  
- **Combined**: 12-20 reports/second (all metrics)

### A100 (40GB)
- **BERTScore**: 60-100 reports/second
- **RadGraph**: 25-40 reports/second
- **CheXbert**: 120-200 reports/second
- **Combined**: 20-35 reports/second (all metrics)

---

## 🎯 Summary

**For most production workloads, we recommend:**
- **VM Type**: `Standard_NC6s_v3` (Tesla V100, 16GB)
- **Cost**: ~$1.80/hour  
- **Performance**: 10-25 reports/second
- **Batch Size**: 16-32 (automatically optimized)

This provides excellent price/performance ratio with sufficient GPU memory for all evaluation metrics. Scale up to dual V100 or A100 instances only for high-throughput batch processing requirements.

The system includes comprehensive GPU optimization, automatic batch sizing, and intelligent memory management to ensure efficient resource utilization regardless of the chosen instance type.
