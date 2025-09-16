"""
GPU-Optimized FastAPI Web Service for CXR Report Metric Evaluation
Optimized for Azure deployment with comprehensive GPU acceleration
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import json
import tempfile
import os
import time
import logging
import asyncio
from typing import List, Optional, Dict, Any
import psutil
import traceback

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Import GPU configuration and optimized evaluation system
try:
    # Prefer the consolidated implementation under azure_deployment
    from azure_deployment.azure_gpu_config import azure_config, print_system_info
    GPU_CONFIG_AVAILABLE = True
    logger.info("Azure GPU configuration (azure_deployment) loaded successfully")
except ImportError as e:
    GPU_CONFIG_AVAILABLE = False
    logger.warning(f"Azure GPU configuration not available: {e}")

# Import modular evaluation system
try:
    from CXRMetric.modular_evaluation import ModularEvaluationRunner
    MODULAR_EVAL_AVAILABLE = True
    logger.info("Modular evaluation system loaded successfully")
except ImportError as e:
    MODULAR_EVAL_AVAILABLE = False
    logger.error(f"Could not import evaluation modules: {e}")

app = FastAPI(
    title="CXR Report Metric API - GPU Optimized",
    description="GPU-accelerated API for evaluating chest X-ray report quality using multiple metrics",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class EvaluationRequest(BaseModel):
    gt_reports: List[str]
    pred_reports: List[str]
    metrics: Optional[List[str]] = ["bertscore", "rouge", "bleu"]
    use_gpu_optimization: Optional[bool] = True
    
class EvaluationResponse(BaseModel):
    results: Dict[str, Any]
    performance_stats: Dict[str, Any]
    status: str
    message: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    gpu_info: Dict[str, Any]
    available_metrics: List[str]

class SystemInfoResponse(BaseModel):
    gpu_config: Dict[str, Any]
    performance_recommendations: List[str]
    memory_usage: Dict[str, Any]

# Global evaluation runner
evaluation_runner = None
performance_stats = {
    "total_evaluations": 0,
    "total_processing_time": 0.0,
    "average_processing_time": 0.0,
    "successful_evaluations": 0,
    "failed_evaluations": 0
}

async def initialize_evaluator():
    """Initialize the evaluation runner with GPU optimization"""
    global evaluation_runner
    
    try:
        if not MODULAR_EVAL_AVAILABLE:
            logger.error("Modular evaluation system not available")
            return False
            
        evaluation_runner = ModularEvaluationRunner()
        
        # Apply GPU optimizations if available
        if GPU_CONFIG_AVAILABLE:
            azure_config.apply_memory_optimizations()
            logger.info(f"GPU optimizations applied. Device: {azure_config.device}")
        
        logger.info("Evaluation runner initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        logger.error(traceback.format_exc())
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the application with GPU optimizations"""
    logger.info("🚀 Starting CXR Report Metric API with GPU optimization...")
    
    # Print system information
    if GPU_CONFIG_AVAILABLE:
        print_system_info()
    
    # Initialize evaluator
    success = await initialize_evaluator()
    
    if not success:
        logger.warning("⚠️  Evaluator initialization failed. Some endpoints may not work.")
    else:
        logger.info("✅ Application startup completed successfully")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - comprehensive health check with GPU info"""
    available_metrics = [
        "bleu", "rouge", "bertscore", "semantic_embedding", 
        "radgraph", "chexpert", "composite", "bbox"
    ]
    
    gpu_info = {}
    gpu_available = False
    
    if GPU_CONFIG_AVAILABLE:
        try:
            gpu_available = azure_config.device.startswith("cuda")
            gpu_info = {
                "device": azure_config.device,
                "gpu_memory_gb": azure_config.gpu_memory_gb,
                "cuda_available": gpu_available,
                "optimal_batch_size": azure_config.get_optimal_batch_size("bert") if gpu_available else "N/A"
            }
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        gpu_available=gpu_available,
        gpu_info=gpu_info,
        available_metrics=available_metrics
    )

@app.get("/health")
async def health_check():
    """Simple health check endpoint for Azure load balancers"""
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint for Azure container orchestration"""
    if evaluation_runner is None:
        raise HTTPException(status_code=503, detail="Evaluation system not ready")
    return {"status": "ready", "timestamp": pd.Timestamp.now().isoformat()}

@app.get("/system-info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get comprehensive system information and GPU status"""
    
    if not GPU_CONFIG_AVAILABLE:
        raise HTTPException(status_code=503, detail="GPU configuration not available")
    
    try:
        # Get GPU configuration
        validation_results = azure_config.validate_environment()
        
        # Get memory usage
        memory_info = {
            "cpu_memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "cpu_memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_memory_percent": psutil.virtual_memory().percent
        }
        
        if validation_results["gpu_available"]:
            try:
                import torch
                memory_info.update({
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_cached_gb": torch.cuda.memory_reserved() / (1024**3),
                    "gpu_memory_total_gb": azure_config.gpu_memory_gb
                })
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
        
        return SystemInfoResponse(
            gpu_config=validation_results,
            performance_recommendations=validation_results.get("recommendations", []),
            memory_usage=memory_info
        )
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_reports(request: EvaluationRequest):
    """Evaluate reports using specified metrics with GPU optimization"""
    
    if evaluation_runner is None:
        raise HTTPException(status_code=503, detail="Evaluation system not available")
    
    start_time = time.time()
    
    try:
        # Validate input
        if len(request.gt_reports) != len(request.pred_reports):
            raise HTTPException(
                status_code=400, 
                detail=f"Ground truth ({len(request.gt_reports)}) and predicted reports ({len(request.pred_reports)}) must have same length"
            )
        
        if len(request.gt_reports) == 0:
            raise HTTPException(status_code=400, detail="No reports provided for evaluation")
        
        # Log request info
        logger.info(f"Processing evaluation request: {len(request.gt_reports)} reports, metrics: {request.metrics}")
        
        # Create temporary DataFrame
        data = pd.DataFrame({
            'gt_reports': request.gt_reports,
            'pred_reports': request.pred_reports
        })
        
        # Run evaluation with GPU optimization
        results = evaluation_runner.run_evaluation(
            data=data,
            metrics=request.metrics,
            gt_col='gt_reports',
            pred_col='pred_reports'
        )
        
        processing_time = time.time() - start_time
        
        # Update global performance stats
        global performance_stats
        performance_stats["total_evaluations"] += len(request.gt_reports)
        performance_stats["total_processing_time"] += processing_time
        performance_stats["successful_evaluations"] += 1
        performance_stats["average_processing_time"] = (
            performance_stats["total_processing_time"] / performance_stats["successful_evaluations"]
        )
        
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            try:
                if hasattr(value, 'to_dict'):
                    json_results[key] = value.to_dict()
                elif isinstance(value, pd.DataFrame):
                    json_results[key] = value.to_dict('records')
                elif isinstance(value, (int, float, str, bool, list, dict)):
                    json_results[key] = value
                else:
                    json_results[key] = str(value)
            except Exception as e:
                logger.warning(f"Could not serialize result {key}: {e}")
                json_results[key] = f"<non-serializable: {type(value)}>"
        
        # Get performance stats
        perf_stats = {
            "processing_time": processing_time,
            "reports_per_second": len(request.gt_reports) / processing_time,
            "total_reports_processed": performance_stats["total_evaluations"]
        }
        
        # Add GPU stats if available
        if GPU_CONFIG_AVAILABLE and evaluation_runner:
            try:
                # Get evaluator performance stats if available
                for metric_name in request.metrics:
                    evaluator = getattr(evaluation_runner, f"{metric_name}_evaluator", None)
                    if evaluator and hasattr(evaluator, 'get_performance_stats'):
                        perf_stats[f"{metric_name}_stats"] = evaluator.get_performance_stats()
            except Exception as e:
                logger.warning(f"Could not get detailed performance stats: {e}")
        
        return EvaluationResponse(
            results=json_results,
            performance_stats=perf_stats,
            status="success",
            message=f"Evaluation completed for {len(request.gt_reports)} reports using {', '.join(request.metrics)}",
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        performance_stats["failed_evaluations"] += 1
        
        logger.error(f"Evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": f"Evaluation failed: {str(e)}",
                "processing_time": processing_time,
                "metrics_requested": request.metrics
            }
        )

@app.post("/evaluate/upload")
async def evaluate_from_files(
    gt_file: UploadFile = File(...),
    pred_file: UploadFile = File(...),
    metrics: Optional[str] = "bertscore,rouge,bleu"
):
    """Evaluate reports from uploaded CSV files with GPU optimization"""
    
    if evaluation_runner is None:
        raise HTTPException(status_code=503, detail="Evaluation system not available")
    
    start_time = time.time()
    
    try:
        # Validate file types
        if not gt_file.filename.endswith('.csv') or not pred_file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Parse metrics list
        metrics_list = [m.strip() for m in metrics.split(',') if m.strip()]
        
        logger.info(f"Processing file upload: GT={gt_file.filename}, Pred={pred_file.filename}, Metrics={metrics_list}")
        
        # Save uploaded files temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            gt_path = os.path.join(temp_dir, "gt_reports.csv")
            pred_path = os.path.join(temp_dir, "pred_reports.csv")
            
            # Save files
            with open(gt_path, "wb") as f:
                content = await gt_file.read()
                f.write(content)
            with open(pred_path, "wb") as f:
                content = await pred_file.read()
                f.write(content)
            
            # Load data
            try:
                gt_df = pd.read_csv(gt_path)
                pred_df = pd.read_csv(pred_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse CSV files: {str(e)}")
            
            # Validate dataframes
            if len(gt_df) == 0 or len(pred_df) == 0:
                raise HTTPException(status_code=400, detail="Empty CSV files provided")
            
            if len(gt_df) != len(pred_df):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Ground truth file has {len(gt_df)} rows, predicted file has {len(pred_df)} rows. They must match."
                )
            
            # Find text columns (assume first text column contains reports)
            gt_text_cols = gt_df.select_dtypes(include=['object']).columns
            pred_text_cols = pred_df.select_dtypes(include=['object']).columns
            
            if len(gt_text_cols) == 0 or len(pred_text_cols) == 0:
                raise HTTPException(status_code=400, detail="No text columns found in CSV files")
            
            gt_col = gt_text_cols[0]
            pred_col = pred_text_cols[0]
            
            logger.info(f"Using columns: GT='{gt_col}', Pred='{pred_col}'")
            
            # Create evaluation DataFrame
            data = pd.DataFrame({
                'gt_reports': gt_df[gt_col],
                'pred_reports': pred_df[pred_col]
            })
            
            # Run evaluation
            results = evaluation_runner.run_evaluation(
                data=data,
                metrics=metrics_list,
                gt_col='gt_reports',
                pred_col='pred_reports'
            )
            
            processing_time = time.time() - start_time
            
            # Convert to JSON-serializable format
            json_results = {}
            for key, value in results.items():
                try:
                    if hasattr(value, 'to_dict'):
                        json_results[key] = value.to_dict()
                    elif isinstance(value, pd.DataFrame):
                        json_results[key] = value.to_dict('records')
                    else:
                        json_results[key] = value
                except Exception as e:
                    logger.warning(f"Could not serialize result {key}: {e}")
                    json_results[key] = str(value)
            
            return EvaluationResponse(
                results=json_results,
                performance_stats={
                    "processing_time": processing_time,
                    "reports_per_second": len(data) / processing_time,
                    "input_files": {
                        "gt_file": gt_file.filename,
                        "pred_file": pred_file.filename,
                        "gt_column": gt_col,
                        "pred_column": pred_col
                    }
                },
                status="success",
                message=f"File evaluation completed for {len(data)} reports from uploaded files",
                processing_time=processing_time
            )
            
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"File evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": f"File evaluation failed: {str(e)}",
                "processing_time": processing_time
            }
        )

@app.get("/metrics")
async def list_available_metrics():
    """List all available evaluation metrics with detailed information"""
    
    metrics_info = {
        "bleu": {
            "description": "BLEU score for n-gram overlap assessment",
            "output_columns": ["bleu_1", "bleu_2", "bleu_3", "bleu_4"],
            "gpu_optimized": False,
            "recommended_batch_size": "N/A (fast computation)"
        },
        "rouge": {
            "description": "ROUGE scores for text summarization quality",
            "output_columns": ["rouge_1_f", "rouge_2_f", "rouge_l_f"],
            "gpu_optimized": False,
            "recommended_batch_size": "N/A (fast computation)"
        },
        "bertscore": {
            "description": "GPU-optimized BERTScore for semantic similarity using PubMedBERT",
            "output_columns": ["bertscore_precision", "bertscore_recall", "bertscore_f1"],
            "gpu_optimized": True,
            "recommended_batch_size": "Auto-calculated based on GPU memory"
        },
        "semantic_embedding": {
            "description": "GPU-accelerated cosine similarity of sentence embeddings",
            "output_columns": ["embedding_cosine_similarity"],
            "gpu_optimized": True,
            "recommended_batch_size": "Auto-calculated based on GPU memory"
        },
        "radgraph": {
            "description": "RadGraph F1 score for medical entity matching (GPU-optimized)",
            "output_columns": ["radgraph_partial_f1", "radgraph_complete_f1"],
            "gpu_optimized": True,
            "recommended_batch_size": "Smaller batches due to graph processing"
        },
        "chexpert": {
            "description": "GPU-accelerated CheXpert label accuracy for medical conditions",
            "output_columns": ["chexpert_micro_f1", "chexpert_macro_f1"],
            "gpu_optimized": True,
            "recommended_batch_size": "Large batches supported"
        },
        "composite": {
            "description": "Composite metric combining multiple scores",
            "output_columns": ["composite_score"],
            "gpu_optimized": True,
            "recommended_batch_size": "Depends on component metrics"
        },
        "bbox": {
            "description": "Bounding box evaluation for spatial accuracy",
            "output_columns": ["bbox_iou", "bbox_precision", "bbox_recall"],
            "gpu_optimized": False,
            "recommended_batch_size": "N/A (geometric computation)"
        }
    }
    
    # Add performance information if GPU config is available
    performance_info = {}
    if GPU_CONFIG_AVAILABLE:
        performance_info = {
            "gpu_available": azure_config.device.startswith("cuda"),
            "device": azure_config.device,
            "gpu_memory_gb": azure_config.gpu_memory_gb,
            "recommended_metrics_for_gpu": ["bertscore", "semantic_embedding", "chexpert"]
        }
    
    return {
        "available_metrics": list(metrics_info.keys()),
        "metrics_details": metrics_info,
        "performance_info": performance_info,
        "gpu_optimization_available": GPU_CONFIG_AVAILABLE
    }

@app.get("/performance-stats")
async def get_performance_stats():
    """Get comprehensive performance statistics"""
    
    stats = performance_stats.copy()
    
    # Add system information
    if GPU_CONFIG_AVAILABLE:
        stats.update({
            "gpu_device": azure_config.device,
            "gpu_memory_gb": azure_config.gpu_memory_gb,
            "optimal_batch_sizes": {
                "bert": azure_config.get_optimal_batch_size("bert"),
                "radgraph": azure_config.get_optimal_batch_size("radgraph"),
                "chexbert": azure_config.get_optimal_batch_size("chexbert")
            }
        })
    
    # Add memory usage
    memory_info = {
        "cpu_memory_percent": psutil.virtual_memory().percent,
        "cpu_memory_available_gb": psutil.virtual_memory().available / (1024**3)
    }
    
    if GPU_CONFIG_AVAILABLE and azure_config.device.startswith("cuda"):
        try:
            import torch
            memory_info.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_cached_gb": torch.cuda.memory_reserved() / (1024**3)
            })
        except Exception as e:
            logger.warning(f"Could not get GPU memory stats: {e}")
    
    stats["memory_usage"] = memory_info
    stats["timestamp"] = pd.Timestamp.now().isoformat()
    
    return stats

@app.post("/warm-up")
async def warm_up_models(background_tasks: BackgroundTasks):
    """Warm up GPU models for optimal performance"""
    
    if not GPU_CONFIG_AVAILABLE:
        raise HTTPException(status_code=503, detail="GPU optimization not available")
    
    def warm_up_task():
        try:
            logger.info("Starting model warm-up...")
            
            # Sample medical texts for warm-up
            sample_texts = [
                "No acute cardiopulmonary abnormality.",
                "Heart size is normal. The lungs are clear.",
                "There is no evidence of pneumonia or other acute findings.",
                "Mild cardiomegaly with clear lungs.",
                "Bilateral lower lobe opacity consistent with pneumonia."
            ]
            
            if evaluation_runner:
                # Warm up with a small evaluation
                data = pd.DataFrame({
                    'gt_reports': sample_texts[:3],
                    'pred_reports': sample_texts[1:4]
                })
                
                results = evaluation_runner.run_evaluation(
                    data=data,
                    metrics=["bertscore"],  # Start with one GPU metric
                    gt_col='gt_reports',
                    pred_col='pred_reports'
                )
                
                logger.info("Model warm-up completed successfully")
            else:
                logger.warning("No evaluation runner available for warm-up")
                
        except Exception as e:
            logger.error(f"Model warm-up failed: {e}")
    
    background_tasks.add_task(warm_up_task)
    
    return {
        "status": "success",
        "message": "Model warm-up started in background",
        "estimated_time": "30-60 seconds"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
