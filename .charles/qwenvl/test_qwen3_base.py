import os;
from transformers import AutoModelForCausalLM, AutoTokenizer

# tests/test_qwen3_base.py
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B-Base" 
#MODEL_NAME = "Qwen/Qwen3-0.6B"
#MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
#MODEL_NAME = "Qwen/Qwen3-4B-Base"

MODEL_DIR = "../.data/hf/models/"



def test_qwen3_base_model():
    """Test that Qwen3 base model can generate text."""
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    # Skip test if model not available
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Try to load without flash attention first
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
#            attn_implementation="eager",
            trust_remote_code=True  # May be needed for Qwen models
        )
        
        # Inspect model dimensions
        print("\n=== Qwen Model Architecture Analysis ===")
        print(f"Model config:")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Vocab size: {model.config.vocab_size}")
        print(f"  Number of layers: {model.config.num_hidden_layers}")
        print(f"  Number of attention heads: {model.config.num_attention_heads}")
        
        # Check embedding dimensions
        embed_tokens = model.get_input_embeddings()
        print(f"\nEmbedding layer:")
        print(f"  Embedding dim: {embed_tokens.embedding_dim}")
        print(f"  Num embeddings: {embed_tokens.num_embeddings}")
        print(f"  Weight shape: {embed_tokens.weight.shape}")
        
        # Test token embedding vs hidden size
        test_tokens = torch.tensor([[1, 2, 3]]).to(model.device)
        token_embeds = embed_tokens(test_tokens)
        print(f"\nToken embedding test:")
        print(f"  Input tokens shape: {test_tokens.shape}")
        print(f"  Output embeddings shape: {token_embeds.shape}")
        print(f"  Embedding dimension matches hidden size: {token_embeds.shape[-1] == model.config.hidden_size}")
        
        prompt = "Why the sky is blue?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=2048)
            result = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            
        print(f"\nGeneration result:\n{result}")
    except ImportError as e:
        pytest.skip(f"Flash attention dependency issue: {e}")


if __name__ == "__main__":
   # pytest.main(["-v", __file__])
   result = test_qwen3_base_model()
   print(result)