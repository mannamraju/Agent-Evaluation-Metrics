#!/usr/bin/env python3
"""
Alternative Medical Imaging Models for Local Deployment
"""

def print_medical_model_alternatives():
    print("🏥 Local Medical Imaging Model Alternatives")
    print("=" * 50)
    
    alternatives = [
        {
            "name": "MedSAM",
            "description": "Medical image segmentation model",
            "size": "~2.5GB",
            "vram": "4-8GB",
            "github": "https://github.com/MedSAM/MedSAM",
            "compatible": "✅ Your laptop can handle this"
        },
        {
            "name": "MedCLIP",
            "description": "Medical image-text understanding",
            "size": "~1.5GB", 
            "vram": "4-6GB",
            "huggingface": "Available on HuggingFace",
            "compatible": "✅ Your laptop can handle this"
        },
        {
            "name": "ChestX-ray14",
            "description": "Chest X-ray classification models",
            "size": "~500MB-2GB",
            "vram": "2-4GB",
            "source": "Multiple implementations available",
            "compatible": "✅ Your laptop can handle this"
        },
        {
            "name": "RadImageNet Models",
            "description": "Pretrained on radiological images",
            "size": "~100MB-1GB",
            "vram": "2-4GB", 
            "source": "ResNet/DenseNet variants",
            "compatible": "✅ Your laptop can handle this"
        },
        {
            "name": "MONAI Models",
            "description": "Medical imaging deep learning framework",
            "size": "Various (100MB-5GB)",
            "vram": "2-8GB",
            "website": "https://monai.io/",
            "compatible": "✅ Most models compatible"
        }
    ]
    
    for i, model in enumerate(alternatives, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Description: {model['description']}")
        print(f"   Model Size: {model['size']}")
        print(f"   VRAM Needed: {model['vram']}")
        print(f"   Compatibility: {model['compatible']}")
        if 'github' in model:
            print(f"   GitHub: {model['github']}")
        if 'huggingface' in model:
            print(f"   Source: {model['huggingface']}")
        if 'website' in model:
            print(f"   Website: {model['website']}")
    
    print(f"\n💡 Recommendation:")
    print(f"   Start with MedCLIP or MONAI models for medical imaging tasks")
    print(f"   Your RTX 4050 (6.4GB VRAM) can handle most of these models")

if __name__ == "__main__":
    print_medical_model_alternatives()
