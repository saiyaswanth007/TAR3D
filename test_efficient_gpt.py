import torch
import sys
import os

# Add current directory to path so we can import tar3d
sys.path.append(os.getcwd())

from tar3d.autoregressive.gpt import GPT_models, ModelArgs

def test_efficient_gpt():
    print("Initializing GPT-B (Smallest model for testing)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    # block_size must be compatible with 3D RoPE (3 * grid_size^2)
    # 3072 = 3 * 32^2
    model = GPT_models['GPT-B'](vocab_size=1000, num_classes=10, block_size=3072).to(device)
    print("Model initialized successfully.")
    
    # Check layers
    print("\nChecking Layer Types:")
    linear_count = 0
    full_count = 0
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        print(f"Layer {i}: {layer_type}")
        if "Linear" in layer_type:
            linear_count += 1
        else:
            full_count += 1
            
    print(f"\nTotal Layers: {len(model.layers)}")
    print(f"Linear Layers: {linear_count}")
    print(f"Full Layers: {full_count}")
    
    assert linear_count > full_count, "Should have more Linear layers than Full layers"
    
    # Dummy Input
    batch_size = 2
    seq_len = 128
    idx = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    # cond_idx should be features (B, token_num, caption_dim)
    # CaptionEmbedder defaults token_num=197, caption_dim=2048 for GPT-B (default ModelArgs)
    cond_idx = torch.randn(batch_size, 197, 2048).to(device)
    
    print("\nRunning Forward Pass...")
    try:
        logits, loss = model(idx=idx, cond_idx=cond_idx)
        print("Forward pass successful!")
        print(f"Logits shape: {logits.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

if __name__ == "__main__":
    test_efficient_gpt()
