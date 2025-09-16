"""
Azure Batch configuration for large-scale CXR report evaluation
Use this for processing thousands of reports in parallel
"""
import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
from azure.batch import models
from azure.storage.blob import BlobServiceClient
import datetime

class CXRMetricBatchProcessor:
    def __init__(self, batch_account_name, batch_account_key, batch_service_url, 
                 storage_account_name, storage_account_key):
        """Initialize Azure Batch client for CXR metric evaluation"""
        
        # Batch service credentials
        self.batch_credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
        self.batch_client = BatchServiceClient(self.batch_credentials, batch_service_url)
        
        # Storage service client
        self.blob_client = BlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=storage_account_key
        )
        
    def create_pool(self, pool_id, vm_size="Standard_D2_v2", node_count=2):
        """Create a batch pool for CXR metric evaluation"""
        
        # Container configuration for CXR metric image
        container_conf = models.ContainerConfiguration(
            container_image_names=["cxrmetricregistry.azurecr.io/cxr-metric:latest"],
            container_registries=[
                models.ContainerRegistry(
                    registry_server="cxrmetricregistry.azurecr.io",
                    user_name="your-registry-username",
                    password="your-registry-password"
                )
            ]
        )
        
        # VM configuration
        vm_config = models.VirtualMachineConfiguration(
            image_reference=models.ImageReference(
                publisher="microsoft-azure-batch",
                offer="ubuntu-server-container",
                sku="20-04-lts",
                version="latest"
            ),
            container_configuration=container_conf,
            node_agent_sku_id="batch.node.ubuntu 20.04"
        )
        
        # Pool configuration
        pool = models.PoolAddParameter(
            id=pool_id,
            vm_size=vm_size,
            virtual_machine_configuration=vm_config,
            target_dedicated_nodes=node_count,
            start_task=models.StartTask(
                command_line="/bin/bash -c 'echo Pool started'",
                wait_for_success=True,
                user_identity=models.UserIdentity(
                    auto_user=models.AutoUserSpecification(
                        scope=models.AutoUserScope.pool,
                        elevation_level=models.ElevationLevel.admin
                    )
                )
            )
        )
        
        try:
            self.batch_client.pool.add(pool)
            print(f"Pool {pool_id} created successfully")
        except models.BatchErrorException as e:
            if e.error.code == "PoolExists":
                print(f"Pool {pool_id} already exists")
            else:
                raise
    
    def create_job(self, job_id, pool_id):
        """Create a batch job"""
        job = models.JobAddParameter(
            id=job_id,
            pool_info=models.PoolInformation(pool_id=pool_id)
        )
        
        try:
            self.batch_client.job.add(job)
            print(f"Job {job_id} created successfully")
        except models.BatchErrorException as e:
            if e.error.code == "JobExists":
                print(f"Job {job_id} already exists")
            else:
                raise
    
    def add_evaluation_task(self, job_id, task_id, input_blob_path, output_blob_path, 
                          metrics=["bleu", "rouge", "bertscore"]):
        """Add a CXR metric evaluation task"""
        
        metrics_str = ",".join(metrics)
        command = (
            f"/bin/bash -c '"
            f"python /app/evaluate_modular.py "
            f"--input_file {input_blob_path} "
            f"--output_file {output_blob_path} "
            f"--metrics {metrics_str} "
            f"--batch_mode'"
        )
        
        task = models.TaskAddParameter(
            id=task_id,
            command_line=command,
            container_settings=models.TaskContainerSettings(
                image_name="cxrmetricregistry.azurecr.io/cxr-metric:latest",
                container_run_options="--rm"
            ),
            resource_files=[
                models.ResourceFile(
                    http_url=f"https://yourstorageaccount.blob.core.windows.net/{input_blob_path}",
                    file_path="input_data.csv"
                )
            ],
            output_files=[
                models.OutputFile(
                    file_pattern="output_*.json",
                    destination=models.OutputFileDestination(
                        container=models.OutputFileBlobContainerDestination(
                            container_url=f"https://yourstorageaccount.blob.core.windows.net/outputs/{output_blob_path}"
                        )
                    ),
                    upload_options=models.OutputFileUploadOptions(
                        upload_condition=models.OutputFileUploadCondition.task_completion
                    )
                )
            ]
        )
        
        self.batch_client.task.add(job_id, task)
        print(f"Task {task_id} added to job {job_id}")
    
    def monitor_tasks(self, job_id):
        """Monitor task progress"""
        print(f"Monitoring tasks for job {job_id}")
        
        while True:
            tasks = list(self.batch_client.task.list(job_id))
            incomplete_tasks = [task for task in tasks if task.state != models.TaskState.completed]
            
            if not incomplete_tasks:
                print("All tasks completed!")
                break
                
            print(f"{len(incomplete_tasks)} tasks remaining...")
            time.sleep(30)
    
    def cleanup_resources(self, pool_id, job_id):
        """Clean up batch resources"""
        try:
            self.batch_client.job.delete(job_id)
            print(f"Job {job_id} deleted")
        except:
            pass
            
        try:
            self.batch_client.pool.delete(pool_id)
            print(f"Pool {pool_id} deleted")
        except:
            pass

def main():
    """Example usage of batch processing"""
    
    # Azure credentials (use environment variables or Key Vault)
    batch_processor = CXRMetricBatchProcessor(
        batch_account_name=os.getenv("AZURE_BATCH_ACCOUNT_NAME"),
        batch_account_key=os.getenv("AZURE_BATCH_ACCOUNT_KEY"),
        batch_service_url=os.getenv("AZURE_BATCH_SERVICE_URL"),
        storage_account_name=os.getenv("AZURE_STORAGE_ACCOUNT_NAME"),
        storage_account_key=os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    )
    
    pool_id = "cxr-metric-pool"
    job_id = f"cxr-metric-job-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        # Create pool and job
        batch_processor.create_pool(pool_id, vm_size="Standard_D4_v2", node_count=4)
        batch_processor.create_job(job_id, pool_id)
        
        # Add evaluation tasks (example for multiple datasets)
        datasets = ["dataset1.csv", "dataset2.csv", "dataset3.csv"]
        for i, dataset in enumerate(datasets):
            batch_processor.add_evaluation_task(
                job_id=job_id,
                task_id=f"eval-task-{i}",
                input_blob_path=f"inputs/{dataset}",
                output_blob_path=f"results_{i}.json",
                metrics=["bleu", "rouge", "bertscore", "radgraph"]
            )
        
        # Monitor completion
        batch_processor.monitor_tasks(job_id)
        
        print("Batch evaluation completed!")
        
    finally:
        # Cleanup
        batch_processor.cleanup_resources(pool_id, job_id)

if __name__ == "__main__":
    main()
