"""
GPU-optimized base evaluator for Azure deployment
"""
import torch
import numpy as np
import pandas as pd
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import psutil
import gc

# Import Azure GPU configuration
try:
    from azure_gpu_config import azure_config, get_device, apply_optimizations
    AZURE_GPU_CONFIG_AVAILABLE = True
except ImportError:
    AZURE_GPU_CONFIG_AVAILABLE = False
    def get_device():
        return "cuda:0" if torch.cuda.is_available() else "cpu"

class GPUOptimizedBaseEvaluator(ABC):
    """
    GPU-optimized base class for all metric evaluators in Azure deployment
    """
    
    def __init__(self, batch_size: Optional[int] = None, device: Optional[str] = None):
        """
        Initialize GPU-optimized evaluator
        
        Args:
            batch_size: Override automatic batch size calculation
            device: Override automatic device selection
        """
        # Device configuration
        self.device = device if device else get_device()
        
        # Apply GPU optimizations if available
        if AZURE_GPU_CONFIG_AVAILABLE:
            apply_optimizations()
            self.batch_size = batch_size if batch_size else azure_config.get_optimal_batch_size(self.model_type)
            self.worker_config = azure_config.get_worker_config()
        else:
            self.batch_size = batch_size if batch_size else 8
            self.worker_config = {"num_workers": 2, "pin_memory": False}
        
        # Performance monitoring
        self.performance_stats = {
            "total_evaluations": 0,
            "total_time": 0.0,
            "gpu_memory_peak": 0.0,
            "cpu_memory_peak": 0.0
        }
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize model-specific components
        self._initialize_models()
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return model type for batch size optimization"""
        pass
    
    @abstractmethod
    def _initialize_models(self):
        """Initialize model-specific components"""
        pass
    
    @abstractmethod
    def compute_metric(self, gt_reports: List[str], pred_reports: List[str]) -> Dict[str, Any]:
        """Compute the metric between ground truth and predicted reports"""
        pass
    
    @abstractmethod
    def get_metric_columns(self) -> List[str]:
        """Return list of metric column names"""
        pass
    
    def _batch_process(self, data: List[Any], process_fn, batch_size: Optional[int] = None) -> List[Any]:
        """
        Process data in batches for memory efficiency
        
        Args:
            data: Input data to process
            process_fn: Function to process each batch
            batch_size: Override default batch size
        
        Returns:
            List of processed results
        """
        if not data:
            return []
        
        effective_batch_size = batch_size if batch_size else self.batch_size
        results = []
        
        start_time = time.time()
        
        for i in range(0, len(data), effective_batch_size):
            batch = data[i:i + effective_batch_size]
            
            try:
                # Process batch
                batch_results = process_fn(batch)
                results.extend(batch_results)
                
                # Memory cleanup every few batches
                if i % (effective_batch_size * 5) == 0:
                    self._cleanup_memory()
                
                # Log progress for large datasets
                if len(data) > 100 and i % (effective_batch_size * 10) == 0:
                    progress = (i + effective_batch_size) / len(data) * 100
                    elapsed = time.time() - start_time
                    self.logger.info(f"Processed {i + effective_batch_size}/{len(data)} items ({progress:.1f}%) in {elapsed:.1f}s")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"OOM at batch size {effective_batch_size}, reducing to {effective_batch_size // 2}")
                    # Recursive call with smaller batch size
                    remaining_data = data[i:]
                    remaining_results = self._batch_process(remaining_data, process_fn, effective_batch_size // 2)
                    results.extend(remaining_results)
                    break
                else:
                    raise e
        
        return results
    
    def _cleanup_memory(self):
        """Clean up GPU and CPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _monitor_memory(self):
        """Monitor GPU and CPU memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.performance_stats["gpu_memory_peak"] = max(
                self.performance_stats["gpu_memory_peak"], gpu_memory
            )
        
        cpu_memory = psutil.Process().memory_info().rss / 1024**3  # GB
        self.performance_stats["cpu_memory_peak"] = max(
            self.performance_stats["cpu_memory_peak"], cpu_memory
        )
    
    def evaluate_batch(self, gt_reports: List[str], pred_reports: List[str]) -> pd.DataFrame:
        """
        Evaluate a batch of reports with GPU optimization
        
        Args:
            gt_reports: Ground truth reports
            pred_reports: Predicted reports
            
        Returns:
            DataFrame with evaluation results
        """
        if len(gt_reports) != len(pred_reports):
            raise ValueError("Ground truth and predicted reports must have the same length")
        
        start_time = time.time()
        
        try:
            # Compute metrics
            results = self.compute_metric(gt_reports, pred_reports)
            
            # Convert to DataFrame
            if isinstance(results, dict):
                # Single-row result
                df = pd.DataFrame([results])
            elif isinstance(results, list):
                # Multi-row result
                df = pd.DataFrame(results)
            else:
                raise ValueError(f"Unexpected result type: {type(results)}")
            
            # Update performance stats
            elapsed_time = time.time() - start_time
            self.performance_stats["total_evaluations"] += len(gt_reports)
            self.performance_stats["total_time"] += elapsed_time
            
            # Monitor memory usage
            self._monitor_memory()
            
            # Log performance
            if len(gt_reports) > 10:
                self.logger.info(f"Evaluated {len(gt_reports)} reports in {elapsed_time:.2f}s "
                               f"({len(gt_reports)/elapsed_time:.1f} reports/sec)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            # Return empty DataFrame with correct columns
            empty_df = pd.DataFrame(columns=self.get_metric_columns())
            return empty_df
        
        finally:
            # Always clean up memory
            self._cleanup_memory()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        
        if stats["total_time"] > 0:
            stats["reports_per_second"] = stats["total_evaluations"] / stats["total_time"]
        else:
            stats["reports_per_second"] = 0.0
        
        # Add system info
        if AZURE_GPU_CONFIG_AVAILABLE:
            stats["device"] = self.device
            stats["batch_size"] = self.batch_size
            stats["gpu_memory_total"] = azure_config.gpu_memory_gb
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            "total_evaluations": 0,
            "total_time": 0.0,
            "gpu_memory_peak": 0.0,
            "cpu_memory_peak": 0.0
        }
    
    def warm_up(self, sample_texts: Optional[List[str]] = None):
        """
        Warm up the model with sample inputs to optimize CUDA kernels
        
        Args:
            sample_texts: Sample texts for warm-up. If None, uses default samples.
        """
        if not sample_texts:
            sample_texts = [
                "No acute cardiopulmonary abnormality.",
                "Heart size is normal. The lungs are clear.",
                "There is no evidence of pneumonia or other acute findings."
            ]
        
        self.logger.info("Warming up model...")
        
        try:
            # Run a small evaluation to warm up CUDA kernels
            _ = self.evaluate_batch(sample_texts[:2], sample_texts[1:3])
            self.logger.info("Model warm-up completed successfully")
        except Exception as e:
            self.logger.warning(f"Model warm-up failed: {e}")
    
    def __str__(self):
        return f"{self.__class__.__name__}(device={self.device}, batch_size={self.batch_size})"
    
    def __repr__(self):
        return self.__str__()
