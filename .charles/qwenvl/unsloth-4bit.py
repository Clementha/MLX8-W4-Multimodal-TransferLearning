from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

# import os
# MODEL_NAME = "unsloth/wen2.5-7B-Instruct-unsloth-bnb-4bit"
# MODEL_DIR = "../.data/hf/models/"
# model_path = os.path.join(MODEL_DIR, MODEL_NAME)

# model, tokenizer = FastVisionModel.from_pretrained(
#     model_path, #"unsloth/wen2.5-7B-Instruct-unsloth-bnb-4bit", 
#     load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
#     local_files_only=True,
# )

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/wen2.5-7B-Instruct-unsloth-bnb-4bit", 
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)
""" not working :( """