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
        
        # model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_NAME,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        #     attn_implementation="eager",
        #     trust_remote_code=True  # May be needed for Qwen models
        # )

        prompt = "Why the sky is blue?" # "In one sentence, what is the capital of France?" # hello, who are you?" #
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=2048)
            result = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            
        #assert len(result) > len(prompt)
        #assert "France" in result or "Paris" in result
        
        print(result)
    except ImportError as e:
        pytest.skip(f"Flash attention dependency issue: {e}")


if __name__ == "__main__":
   # pytest.main(["-v", __file__])
   result = test_qwen3_base_model()
   print(result)