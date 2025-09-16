#!/usr/bin/env python3
"""
Quick GPU performance test for perplexity metrics.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def test_gpu_performance():
    print("🚀 GPU Performance Test for Perplexity Metrics")
    print("=" * 50)
    
    # Check GPU availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model and test
    print("\n📥 Loading DistilGPT-2...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device)
    
    load_time = time.time() - start_time
    print(f"⏱️ Model loaded in {load_time:.2f}s on {device}")
    print(f"🎯 Model device: {next(model.parameters()).device}")
    
    # Test with medical text
    medical_texts = [
        "The chest X-ray shows clear lungs with no acute cardiopulmonary abnormalities.",
        "There is consolidation in the right lower lobe consistent with pneumonia.",
        "The heart size appears enlarged suggestive of cardiomegaly."
    ]
    
    print("\n🏥 Testing Medical Text Perplexity:")
    print("-" * 40)
    
    total_start = time.time()
    for i, text in enumerate(medical_texts, 1):
        start_time = time.time()
        
        # Tokenize and compute perplexity
        inputs = tokenizer.encode(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
            perplexity = torch.exp(outputs.loss).item()
        
        compute_time = time.time() - start_time
        
        print(f"{i}. Text: {text[:50]}...")
        print(f"   Perplexity: {perplexity:.2f}")
        print(f"   Time: {compute_time:.3f}s")
        print()
    
    total_time = time.time() - total_start
    print(f"📊 Total computation time: {total_time:.3f}s")
    print(f"⚡ Average time per text: {total_time/len(medical_texts):.3f}s")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e6
        memory_cached = torch.cuda.memory_reserved() / 1e6
        print(f"🧠 GPU Memory - Used: {memory_used:.1f}MB, Cached: {memory_cached:.1f}MB")
    
    print("\n✅ GPU Performance Test Complete!")

if __name__ == "__main__":
    test_gpu_performance()
