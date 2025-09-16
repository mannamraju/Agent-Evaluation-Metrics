"""
GPU-optimized configuration for CXR Report Metric evaluation on Azure
"""
import torch
import os
import psutil

class AzureGPUConfig:
    """Configuration class for optimal GPU performance on Azure"""
    
    # Minimum GPU requirements for efficient evaluation pipeline
    MIN_GPU_SPECS = {
        "recommended": {
            "name": "NVIDIA Tesla V100 or better",
            "memory_gb": 16,
            "compute_capability": 7.0,
            "azure_vm_types": ["Standard_NC6s_v3", "Standard_NC12s_v3", "Standard_NC24s_v3"],
            "description": "Optimal performance for all evaluation metrics"
        },
        "minimum": {
            "name": "NVIDIA Tesla K80 or equivalent",
            "memory_gb": 8,
            "compute_capability": 3.7,
            "azure_vm_types": ["Standard_NC6", "Standard_NC12", "Standard_NC24"],
            "description": "Basic functionality, slower processing"
        },
        "high_performance": {
            "name": "NVIDIA A100 or H100",
            "memory_gb": 40,
            "compute_capability": 8.0,
            "azure_vm_types": ["Standard_ND96asr_v4", "Standard_ND40rs_v2", "Standard_NC24ads_A100_v4"],
            "description": "Maximum throughput for large-scale batch processing"
        }
    }
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.gpu_memory_gb = self._get_gpu_memory()
        self.cpu_count = psutil.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
    def _get_optimal_device(self):
        """Determine the best device for computation"""
        if torch.cuda.is_available():
            # Use the GPU with most memory
            max_memory = 0
            best_device = 0
            
            for i in range(torch.cuda.device_count()):
                memory = torch.cuda.get_device_properties(i).total_memory
                if memory > max_memory:
                    max_memory = memory
                    best_device = i
            
            return f"cuda:{best_device}"
        return "cpu"
    
    def _get_gpu_memory(self):
        """Get available GPU memory in GB"""
        if "cuda" in self.device:
            device_id = int(self.device.split(":")[1])
            return torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        return 0
    
    def get_optimal_batch_size(self, model_type="bert"):
        """Calculate optimal batch size based on available GPU memory"""
        if self.gpu_memory_gb == 0:
            return 1  # CPU fallback
            
        # Empirical batch sizes for different GPU memory sizes
        batch_size_mapping = {
            "bert": {
                8: 8,    # 8GB GPU
                16: 16,  # 16GB GPU  
                32: 32,  # 32GB GPU
                40: 48   # 40GB+ GPU
            },
            "radgraph": {
                8: 4,    # RadGraph is more memory intensive
                16: 8,
                32: 16,
                40: 24
            },
            "chexbert": {
                8: 12,
                16: 24,
                32: 48,
                40: 64
            }
        }
        
        # Find closest memory size
        memory_sizes = sorted(batch_size_mapping[model_type].keys())
        closest_memory = min(memory_sizes, key=lambda x: abs(x - self.gpu_memory_gb))
        
        return batch_size_mapping[model_type][closest_memory]
    
    def get_worker_config(self):
        """Get optimal worker configuration for multi-processing"""
        if self.gpu_memory_gb >= 16:
            return {
                "num_workers": min(8, self.cpu_count // 2),
                "pin_memory": True,
                "prefetch_factor": 2
            }
        elif self.gpu_memory_gb >= 8:
            return {
                "num_workers": min(4, self.cpu_count // 3),
                "pin_memory": True,
                "prefetch_factor": 1
            }
        else:
            return {
                "num_workers": 2,
                "pin_memory": False,
                "prefetch_factor": 1
            }
    
    def apply_memory_optimizations(self):
        """Apply PyTorch memory optimizations"""
        if "cuda" in self.device:
            # Enable memory efficient attention
            torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory fraction to prevent OOM
            if self.gpu_memory_gb <= 8:
                torch.cuda.set_per_process_memory_fraction(0.8)
            elif self.gpu_memory_gb <= 16:
                torch.cuda.set_per_process_memory_fraction(0.9)
            else:
                torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Enable CUDNN benchmark mode for consistent input sizes
            torch.backends.cudnn.benchmark = True
            
            # Enable CUDNN deterministic for reproducibility (disable in production)
            if os.getenv("ENABLE_DETERMINISTIC", "false").lower() == "true":
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
    
    def get_azure_vm_recommendation(self, workload_type="balanced"):
        """Recommend Azure VM types based on workload"""
        recommendations = {
            "development": {
                "vm_types": ["Standard_NC6s_v3", "Standard_NC6s_v2"],
                "description": "Cost-effective for development and testing",
                "estimated_cost_per_hour": "$0.90 - $1.80"
            },
            "production": {
                "vm_types": ["Standard_NC12s_v3", "Standard_NC24s_v3"],
                "description": "Balanced performance and cost for production workloads",
                "estimated_cost_per_hour": "$1.80 - $3.60"
            },
            "batch_processing": {
                "vm_types": ["Standard_ND40rs_v2", "Standard_NC24ads_A100_v4"],
                "description": "High-throughput processing for large datasets",
                "estimated_cost_per_hour": "$3.60 - $27.20"
            },
            "research": {
                "vm_types": ["Standard_ND96asr_v4", "Standard_NC96ads_A100_v4"],
                "description": "Maximum GPU memory and compute for research workloads",
                "estimated_cost_per_hour": "$27.20 - $36.00"
            }
        }
        
        return recommendations.get(workload_type, recommendations["balanced"])
    
    def validate_environment(self):
        """Validate the current environment meets minimum requirements"""
        validation_results = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_count": self.cpu_count,
            "total_memory_gb": self.total_memory_gb,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__
        }
        
        # Check against minimum requirements
        recommendations = []
        
        if not validation_results["gpu_available"]:
            recommendations.append("⚠️ No GPU detected. Performance will be severely degraded.")
        elif validation_results["gpu_memory_gb"] < 8:
            recommendations.append("⚠️ GPU memory < 8GB. Consider upgrading to Standard_NC6s_v3 or higher.")
        elif validation_results["gpu_memory_gb"] >= 16:
            recommendations.append("✅ GPU memory adequate for optimal performance.")
        
        if validation_results["total_memory_gb"] < 16:
            recommendations.append("⚠️ System RAM < 16GB. May cause issues with large datasets.")
        
        if validation_results["cpu_count"] < 4:
            recommendations.append("⚠️ CPU count < 4. Data loading may be slow.")
        
        validation_results["recommendations"] = recommendations
        return validation_results
    
    def print_system_info(self):
        """Print comprehensive system information"""
        validation = self.validate_environment()
        
        print("🔥 CXR Report Metric - Azure GPU Configuration")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"GPU Available: {validation['gpu_available']}")
        print(f"GPU Count: {validation['gpu_count']}")
        print(f"GPU Memory: {validation['gpu_memory_gb']:.1f} GB")
        print(f"CPU Cores: {validation['cpu_count']}")
        print(f"System RAM: {validation['total_memory_gb']:.1f} GB")
        print(f"CUDA Version: {validation['cuda_version']}")
        print(f"PyTorch Version: {validation['pytorch_version']}")
        
        print("\n📊 Optimal Batch Sizes:")
        print(f"BERT Models: {self.get_optimal_batch_size('bert')}")
        print(f"RadGraph: {self.get_optimal_batch_size('radgraph')}")
        print(f"CheXbert: {self.get_optimal_batch_size('chexbert')}")
        
        worker_config = self.get_worker_config()
        print(f"\n⚡ Worker Configuration:")
        print(f"Num Workers: {worker_config['num_workers']}")
        print(f"Pin Memory: {worker_config['pin_memory']}")
        print(f"Prefetch Factor: {worker_config['prefetch_factor']}")
        
        if validation['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in validation['recommendations']:
                print(f"  {rec}")
        
        print(f"\n🏗️ Minimum Azure VM Requirements:")
        for level, specs in self.MIN_GPU_SPECS.items():
            print(f"{level.title()}: {specs['azure_vm_types'][0]} ({specs['memory_gb']}GB GPU)")
        
        return validation

# Global configuration instance
azure_config = AzureGPUConfig()

# Convenience functions
def get_device():
    """Get the optimal compute device"""
    return azure_config.device

def get_optimal_batch_size(model_type="bert"):
    """Get optimal batch size for model type"""
    return azure_config.get_optimal_batch_size(model_type)

def apply_optimizations():
    """Apply all GPU optimizations"""
    azure_config.apply_memory_optimizations()

def print_system_info():
    """Print system information"""
    return azure_config.print_system_info()

if __name__ == "__main__":
    print_system_info()
