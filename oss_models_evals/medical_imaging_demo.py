"""
Medical Imaging Models Demo for RTX 4050 Laptop
Demonstrates running various open source medical AI models
"""

import torch
import requests
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt

def check_gpu_memory():
    """Check available GPU memory"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        free = gpu_memory - cached
        
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 Total VRAM: {gpu_memory:.1f}GB")
        print(f"💾 Available: {free:.1f}GB")
        print(f"🏃 CUDA Version: {torch.version.cuda}")
        return free > 2.0  # Need at least 2GB free
    else:
        print("❌ No GPU available")
        return False

def demo_biomedclip():
    """Demo BiomedCLIP - Medical Vision-Language Model"""
    print("\n🏥 Loading BiomedCLIP (Medical Vision-Language Model)...")
    
    try:
        # Use a smaller medical CLIP model that fits in 6GB
        model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        
        # Load model and processor
        model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.cuda()
            
        print(f"✅ BiomedCLIP loaded successfully!")
        print(f"🎯 Memory usage: ~2-3GB VRAM")
        print(f"🔬 Capabilities:")
        print(f"   - Medical image classification")
        print(f"   - Zero-shot pathology detection") 
        print(f"   - Medical image-text matching")
        
        # Memory cleanup
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading BiomedCLIP: {str(e)}")
        return False

def demo_chexnet_style():
    """Demo ChexNet-style model for chest X-ray pathology detection"""
    print("\n📋 ChexNet-Style Models for Chest X-ray Analysis...")
    
    # These are typically DenseNet or ResNet based models
    models_available = [
        {
            "name": "Stanford CheXNet (DenseNet-121)",
            "memory": "~1.5GB",
            "pathologies": ["Pneumonia", "Pneumothorax", "Edema", "Atelectasis", "etc."],
            "accuracy": "High (AUC > 0.8 for most pathologies)"
        },
        {
            "name": "NIH ChestX-ray14 Models", 
            "memory": "~2GB",
            "pathologies": "14 different chest pathologies",
            "accuracy": "Good baseline performance"
        }
    ]
    
    for model in models_available:
        print(f"🏥 {model['name']}")
        print(f"   💾 Memory: {model['memory']}")
        print(f"   🔍 Detects: {model['pathologies']}")
        print(f"   📊 Performance: {model['accuracy']}")
        print()

def demo_medical_sam():
    """Demo Medical Segment Anything Models"""
    print("\n🎯 Medical Segmentation Models (SAM-based)...")
    
    models = [
        {
            "name": "MedSAM",
            "memory": "~4-5GB", 
            "purpose": "General medical image segmentation",
            "organs": "Any anatomical structure with prompts"
        },
        {
            "name": "SAM-Med2D",
            "memory": "~3-4GB",
            "purpose": "2D medical image segmentation", 
            "organs": "Organs, lesions, anatomical regions"
        }
    ]
    
    for model in models:
        print(f"✂️  {model['name']}")
        print(f"   💾 Memory: {model['memory']}")
        print(f"   🎯 Purpose: {model['purpose']}")
        print(f"   🫁 Segments: {model['organs']}")
        print()

def demo_lightweight_models():
    """Demo lightweight models perfect for 6GB VRAM"""
    print("\n🪶 Lightweight Models Perfect for RTX 4050:")
    
    models = [
        {
            "name": "MobileNet-based Medical Classifiers",
            "memory": "<1GB",
            "speed": "Very Fast",
            "use_case": "Real-time medical image classification"
        },
        {
            "name": "EfficientNet Medical Models",
            "memory": "~1-2GB", 
            "speed": "Fast",
            "use_case": "High accuracy with efficiency"
        },
        {
            "name": "Medical ViT (Vision Transformer) Small",
            "memory": "~2-3GB",
            "speed": "Medium", 
            "use_case": "State-of-the-art accuracy"
        }
    ]
    
    for model in models:
        print(f"⚡ {model['name']}")
        print(f"   💾 Memory: {model['memory']}")
        print(f"   🏃 Speed: {model['speed']}")
        print(f"   🎯 Use Case: {model['use_case']}")
        print()

def recommend_models_for_rtx4050():
    """Recommend best models for RTX 4050 setup"""
    print("\n🎯 RECOMMENDED MODELS FOR YOUR RTX 4050:")
    print("="*50)
    
    recommendations = [
        {
            "priority": "🥇 HIGH PRIORITY",
            "models": [
                "BiomedCLIP - Medical vision-language understanding",
                "CheXNet DenseNet-121 - Chest X-ray pathology detection", 
                "EfficientNet-B4 Medical - Balanced performance/memory"
            ]
        },
        {
            "priority": "🥈 MEDIUM PRIORITY", 
            "models": [
                "Medical ViT-Small - Modern transformer approach",
                "RadImageNet pretrained models - General medical imaging",
                "Medical CLIP variants - Zero-shot classification"
            ]
        },
        {
            "priority": "🥉 EXPERIMENTAL",
            "models": [
                "MedSAM (if you have 5-6GB free) - Advanced segmentation",
                "Medical LLaVA - Vision-language conversations",
                "Lightweight medical diffusion models"
            ]
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['priority']}")
        for model in rec['models']:
            print(f"   • {model}")
    
    print(f"\n💡 GETTING STARTED TIPS:")
    print(f"   1. Start with BiomedCLIP - easiest to implement")
    print(f"   2. Use torch.float16 to save memory")
    print(f"   3. Process images in batches of 8-16")
    print(f"   4. Monitor GPU memory with nvidia-smi")

def main():
    print("🏥 MEDICAL IMAGING MODELS FOR RTX 4050 LAPTOP")
    print("=" * 50)
    
    # Check GPU capability
    has_gpu = check_gpu_memory()
    
    if not has_gpu:
        print("⚠️  Limited GPU memory detected. Consider CPU-only lightweight models.")
    
    # Demo various model categories
    demo_chexnet_style()
    demo_medical_sam() 
    demo_lightweight_models()
    
    # Try to load a real model
    demo_biomedclip()
    
    # Give recommendations
    recommend_models_for_rtx4050()
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Choose a model category that interests you")
    print(f"   2. Install specific model requirements")
    print(f"   3. Download sample medical images for testing")
    print(f"   4. Start with inference, then fine-tuning if needed")

if __name__ == "__main__":
    main()
