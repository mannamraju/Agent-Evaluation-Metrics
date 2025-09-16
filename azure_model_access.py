#!/usr/bin/env python3
"""
Guide to Access Azure AI Models Locally
"""

def print_azure_access_guide():
    print("🌐 How to Access Azure AI Models (Including MedImageInsight)")
    print("=" * 60)
    
    print("\n1. 📋 Check Azure Access Requirements:")
    print("   • Azure subscription (Free tier available)")
    print("   • Azure AI Foundry access")
    print("   • Model-specific permissions")
    
    print("\n2. 💻 Local Deployment Options:")
    print("   • Azure Container Instances (ACI)")
    print("   • Docker containers with Azure AI SDK")
    print("   • Model export (if available)")
    
    print("\n3. 🔄 Alternative: Use Azure API Locally")
    print("   • Call model via REST API")
    print("   • Use Azure AI Inference SDK")
    print("   • Process images locally, send to Azure")
    
    print("\n4. 📊 Cost Considerations:")
    print("   • Free tier: Limited API calls")
    print("   • Pay-per-use: $0.001-$0.10 per image")
    print("   • Container deployment: Compute costs")
    
    print("\n5. 🔧 Setup Steps:")
    print("   a) Create Azure account (free): https://azure.microsoft.com/free/")
    print("   b) Go to Azure AI Foundry: https://ai.azure.com/")
    print("   c) Search for MedImageInsight model")
    print("   d) Check deployment options")
    
    print("\n6. 🐍 Python Example (if API available):")
    python_code = '''
# Example Azure AI SDK usage
from azure.ai.inference import InferenceClient
from azure.core.credentials import AzureKeyCredential

# Setup client (requires API key)
client = InferenceClient(
    endpoint="https://your-endpoint.inference.ai.azure.com",
    credential=AzureKeyCredential("your-api-key")
)

# Process medical image
with open("chest_xray.jpg", "rb") as image:
    result = client.analyze_image(image)
    print(result.predictions)
'''
    print(python_code)

if __name__ == "__main__":
    print_azure_access_guide()
