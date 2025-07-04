#!/usr/bin/env python3
"""
Generate captions for reef images (first N for testing),
and preview the resulting metadata DataFrame.
"""

import sys
from pathlib import Path

import torch
import pandas as pd
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# —— USER CONFIG — adjust paths if needed ——
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-VL-3B-Instruct"
REEF_IMG_DIR = Path(__file__).resolve().parent.parent / "data" / "reef_data" / "images"
OUTPUT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "reef_data" / "reef_metadata.parquet"

# How many to do in this test run:
TEST_ONLY = True
N_TEST = 10
# ————————————————————————————————


def load_model_and_processor(model_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, device_map={"": device}
    ).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor, device


def generate_caption(model, processor, device, image_path: Path) -> str:
    # same as your scratch script
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text",  "text": "Describe this image."},
        ]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    vision_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=vision_inputs,
        padding=True, return_tensors="pt"
    ).to(device)
    generated_ids = model.generate(
        **inputs, max_new_tokens=64, num_beams=2, do_sample=False
    )
    prompt_len = inputs.input_ids.shape[1]
    trimmed = [ids[prompt_len:] for ids in generated_ids]
    cap = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return cap


def main():
    # sanity checks
    if not MODEL_DIR.is_dir():
        print(f"Model dir not found: {MODEL_DIR}", file=sys.stderr); sys.exit(1)
    if not REEF_IMG_DIR.is_dir():
        print(f"Reef images dir not found: {REEF_IMG_DIR}", file=sys.stderr); sys.exit(1)

    model, processor, device = load_model_and_processor(MODEL_DIR)

    rows = []
    img_paths = sorted(REEF_IMG_DIR.glob("*.jpg"))
    if TEST_ONLY:
        img_paths = img_paths[:N_TEST]

    for img_path in img_paths:
        cap = generate_caption(model, processor, device, img_path)
        rows.append({
            "filename": img_path.name,
            "caption":  cap,
            "split":    "test",
        })
        print(f"{img_path.name} → {cap}")

    # build DataFrame & preview
    df = pd.DataFrame(rows, columns=["filename", "caption", "split"])
    print("\nPreview of metadata:")
    print(df)

    # (in script #2 we'll save df.to_parquet(OUTPUT_PARQUET))
    

if __name__ == "__main__":
    main()
