import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml import command
from azure.identity import DefaultAzureCredential

# Azure ML deployment configuration
def deploy_to_azure_ml():
    """Deploy CXR-Report-Metric to Azure Machine Learning"""
    
    # Initialize ML Client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id="your-subscription-id",
        resource_group_name="your-resource-group",
        workspace_name="your-workspace-name"
    )
    
    # Create custom environment
    env = Environment(
        name="cxr-metric-env",
        description="Environment for CXR Report Metric evaluation",
        build=BuildContext(path=".", dockerfile_path="Dockerfile"),
        inference_config={
            "liveness_route": {"path": "/health", "port": 8000},
            "readiness_route": {"path": "/ready", "port": 8000},
            "scoring_route": {"path": "/evaluate", "port": 8000}
        }
    )
    
    # Create and register environment
    ml_client.environments.create_or_update(env)
    
    # Define the command job for batch evaluation
    job = command(
        inputs={
            "gt_reports": "${{inputs.gt_reports}}",
            "pred_reports": "${{inputs.pred_reports}}",
            "metrics": "${{inputs.metrics}}"
        },
        outputs={
            "results": "${{outputs.results}}"
        },
        code="./",
        command="python evaluate_modular.py --gt_reports ${{inputs.gt_reports}} --pred_reports ${{inputs.pred_reports}} --metrics ${{inputs.metrics}} --output_file ${{outputs.results}}/metrics.json",
        environment="cxr-metric-env@latest",
        compute="cpu-cluster",  # or "gpu-cluster" for CUDA workloads
        display_name="CXR Metric Evaluation Job"
    )
    
    return ml_client.create_or_update(job)

if __name__ == "__main__":
    deploy_to_azure_ml()
