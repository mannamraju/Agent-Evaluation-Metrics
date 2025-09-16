# Azure Container Instance Deployment Script - GPU Optimized
# Run this in Azure Cloud Shell or with Azure CLI installed

param(
    [string]$ResourceGroup = "cxr-metric-rg",
    [string]$Location = "West US",
    [string]$ContainerName = "cxr-report-metric",
    [string]$ImageName = "cxr-metric:latest",
    [string]$RegistryName = "cxrmetricregistry",
    [string]$VMType = "gpu",  # Options: cpu, gpu, high-performance
    [int]$Replicas = 1
)

Write-Host "🚀 Deploying CXR-Report-Metric to Azure with GPU optimization" -ForegroundColor Green
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Resource Group: $ResourceGroup"
Write-Host "  Location: $Location"
Write-Host "  VM Type: $VMType"
Write-Host "  Container Name: $ContainerName"

# GPU-optimized VM configurations
$vmConfigs = @{
    "cpu" = @{
        "cpu" = 4
        "memory" = 8
        "sku" = "Standard_D4s_v3"
        "cost_per_hour" = "$0.20"
        "description" = "CPU-only deployment (development/testing)"
    }
    "gpu" = @{
        "cpu" = 6
        "memory" = 56
        "gpu" = "V100:1"
        "sku" = "Standard_NC6s_v3"
        "cost_per_hour" = "$1.80"
        "description" = "Single V100 GPU (recommended for production)"
    }
    "high-performance" = @{
        "cpu" = 12
        "memory" = 112
        "gpu" = "V100:2"
        "sku" = "Standard_NC12s_v3"
        "cost_per_hour" = "$3.60"
        "description" = "Dual V100 GPUs (high-throughput processing)"
    }
}

$config = $vmConfigs[$VMType]
if (-not $config) {
    Write-Error "Invalid VM type. Options: cpu, gpu, high-performance"
    exit 1
}

Write-Host "  VM Config: $($config.description)" -ForegroundColor Cyan
Write-Host "  Estimated Cost: $($config.cost_per_hour)/hour" -ForegroundColor Cyan

# Create resource group
Write-Host "`n📦 Creating resource group..." -ForegroundColor Blue
az group create --name $ResourceGroup --location $Location

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create resource group"
    exit 1
}

# Create Azure Container Registry
Write-Host "`n🏗️  Creating Azure Container Registry..." -ForegroundColor Blue
az acr create --resource-group $ResourceGroup --name $RegistryName --sku Standard --location $Location

if ($LASTEXITCODE -ne 0) {
    Write-Warning "Container registry might already exist, continuing..."
}

# Build and push Docker image with GPU optimization
Write-Host "`n🔨 Building and pushing optimized Docker image..." -ForegroundColor Blue
az acr build --registry $RegistryName --image $ImageName --platform linux/amd64 .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build Docker image"
    exit 1
}

# Get registry credentials
Write-Host "`n🔐 Getting registry credentials..." -ForegroundColor Blue
$registryServer = "$RegistryName.azurecr.io"
$registryUsername = az acr credential show --name $RegistryName --query username --output tsv
$registryPassword = az acr credential show --name $RegistryName --query passwords[0].value --output tsv

# Prepare container creation command based on VM type
$containerArgs = @(
    "container", "create",
    "--resource-group", $ResourceGroup,
    "--name", $ContainerName,
    "--image", "$registryServer/$ImageName",
    "--cpu", $config.cpu,
    "--memory", $config.memory,
    "--ports", "8000", "8080", "9090",
    "--dns-name-label", "cxr-metric-api-$(Get-Random -Minimum 1000 -Maximum 9999)",
    "--registry-login-server", $registryServer,
    "--registry-username", $registryUsername,
    "--registry-password", $registryPassword,
    "--restart-policy", "OnFailure"
)

# Add environment variables
$containerArgs += @(
    "--environment-variables",
    "PYTHONPATH=/app",
    "LOG_LEVEL=INFO",
    "ENABLE_GPU_OPTIMIZATION=true"
)

# Add GPU configuration if needed
if ($VMType -ne "cpu") {
    Write-Host "⚡ Configuring GPU resources..." -ForegroundColor Yellow
    $containerArgs += @("--gpu-count", "1", "--gpu-sku", "V100")
    $containerArgs += @("CUDA_VISIBLE_DEVICES=0", "NVIDIA_VISIBLE_DEVICES=all")
}

# Deploy container
Write-Host "`n🚀 Deploying container to Azure..." -ForegroundColor Blue
& az @containerArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to deploy container"
    exit 1
}

# Get container details
Write-Host "`n📊 Getting deployment details..." -ForegroundColor Blue
$containerInfo = az container show --resource-group $ResourceGroup --name $ContainerName --output json | ConvertFrom-Json

$fqdn = $containerInfo.ipAddress.fqdn
$ip = $containerInfo.ipAddress.ip

Write-Host "`n✅ Deployment completed successfully!" -ForegroundColor Green
Write-Host "📍 Deployment Details:" -ForegroundColor Yellow
Write-Host "  Resource Group: $ResourceGroup"
Write-Host "  Container Name: $ContainerName"
Write-Host "  VM Type: $VMType ($($config.sku))"
Write-Host "  Public IP: $ip"
Write-Host "  FQDN: $fqdn"

Write-Host "`n🌐 API Endpoints:" -ForegroundColor Cyan
Write-Host "  Health Check: http://$fqdn:8000/health"
Write-Host "  API Documentation: http://$fqdn:8000/docs"
Write-Host "  Metrics List: http://$fqdn:8000/metrics"
Write-Host "  Evaluation: http://$fqdn:8000/evaluate"

Write-Host "`n💰 Cost Information:" -ForegroundColor Magenta
Write-Host "  Estimated Cost: $($config.cost_per_hour)/hour"
Write-Host "  Monthly Estimate: $([math]::Round([double]($config.cost_per_hour.Replace('$','')) * 24 * 30, 2))"

Write-Host "`n🔧 Quick Test Commands:" -ForegroundColor White
Write-Host "  # Test health endpoint"
Write-Host "  curl http://$fqdn:8000/health"
Write-Host ""
Write-Host "  # Test evaluation endpoint"
Write-Host "  curl -X POST http://$fqdn:8000/evaluate \\"
Write-Host "    -H 'Content-Type: application/json' \\"
Write-Host "    -d '{""gt_reports"":[""No acute findings.""], ""pred_reports"":[""Heart and lungs are normal.""], ""metrics"":[""bertscore""]}'"

Write-Host "`n📚 Management Commands:" -ForegroundColor White
Write-Host "  # View logs"
Write-Host "  az container logs --resource-group $ResourceGroup --name $ContainerName"
Write-Host ""
Write-Host "  # Restart container"
Write-Host "  az container restart --resource-group $ResourceGroup --name $ContainerName"
Write-Host ""
Write-Host "  # Delete deployment"
Write-Host "  az group delete --name $ResourceGroup --yes --no-wait"

Write-Host "`n⚠️  Important Notes:" -ForegroundColor Red
Write-Host "  - GPU instances may take 2-3 minutes to start"
Write-Host "  - First API call may be slow due to model loading"
Write-Host "  - Monitor costs in Azure Portal"
Write-Host "  - Stop/delete resources when not in use to avoid charges"

# Optional: Wait for container to be ready
if ($VMType -ne "cpu") {
    Write-Host "`n⏳ Waiting for GPU container to be ready (this may take a few minutes)..." -ForegroundColor Yellow
    $retries = 0
    $maxRetries = 30
    
    do {
        Start-Sleep -Seconds 10
        $status = az container show --resource-group $ResourceGroup --name $ContainerName --query instanceView.state --output tsv
        Write-Host "  Container status: $status" -ForegroundColor Gray
        $retries++
    } while ($status -ne "Running" -and $retries -lt $maxRetries)
    
    if ($status -eq "Running") {
        Write-Host "✅ Container is running! Testing API..." -ForegroundColor Green
        
        # Test the health endpoint
        try {
            $healthResponse = Invoke-RestMethod -Uri "http://$fqdn:8000/health" -Method Get -TimeoutSec 30
            Write-Host "🎉 API is responding: $($healthResponse.status)" -ForegroundColor Green
        }
        catch {
            Write-Warning "API not responding yet, may need more time to initialize"
        }
    }
    else {
        Write-Warning "Container may still be starting. Check logs with: az container logs --resource-group $ResourceGroup --name $ContainerName"
    }
}
