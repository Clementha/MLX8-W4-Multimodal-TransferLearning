#!/usr/bin/env python3
"""
qwen_3Bn_test.py

Caption a single reef test image with Qwen2.5-VL-3B-Instruct on GPU,
using the proper chat template + vision utils without invalid ** unpacking.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


# — your exact absolute paths —
MODEL_DIR = Path(
    "/root/MLX8-W4-Multimodal-TransferLearning/.ben/models/"
    "Qwen2.5-VL-3B-Instruct"
)
IMAGE_PATH = Path(
    "/root/MLX8-W4-Multimodal-TransferLearning/.ben/"
    "scratch/card_game.jpeg"
)
# ————————————————————————


def load_model_and_processor(model_dir: Path):
    """
    Load the Qwen2.5-VL model into fp16 on GPU (or CPU if no CUDA),
    and its paired processor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Load the vision-language conditional generation model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()

    # Load the single Processor that handles both text & vision
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor, device


def generate_caption(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    image_path: Path,
) -> str:
    """
    Build the “messages” list, apply the chat template, extract vision data,
    and invoke model.generate()—passing vision inputs correctly as `images=`,
    not via ** unpacking.
    """
    # 1) Build chat‐style messages, telling it “here’s one image + text ask”
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": "Describe this image."},
            ],
        }
    ]

    # 2) Turn messages → single prompt with <image> marker & generation cue
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 3) Extract exactly the right pixel_values list and image counts
    vision_inputs, video_inputs = process_vision_info(messages)
    #   vision_inputs  is a List[PIL.Image] or List[str] (URLs/base64)
    #   video_inputs   is a List (empty here, since no video in messages)

    # 4) Call the processor *with* named args, not ** unpack:
    inputs = processor(
        text=[text],              # list of one prompt
        images=vision_inputs,     # list of one image
        videos=video_inputs,      # likely empty list
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 5) Generate tokens; tweak beams / max_new_tokens if needed
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        num_beams=4,
        do_sample=False,
    )

    # 6) Strip out the prompt portion and decode only the new tokens
    prompt_len = inputs.input_ids.shape[1]
    trimmed = [ids[prompt_len:] for ids in generated_ids]
    caption = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    return caption


def main():
    # sanity
    if not MODEL_DIR.is_dir():
        print(f"[Error] Model dir not found: {MODEL_DIR}", file=sys.stderr)
        sys.exit(1)
    if not IMAGE_PATH.is_file():
        print(f"[Error] Image not found: {IMAGE_PATH}", file=sys.stderr)
        sys.exit(1)

    model, processor, device = load_model_and_processor(MODEL_DIR)
    print("Generating caption…")
    caption = generate_caption(model, processor, device, IMAGE_PATH)

    print("\n=== Caption ===")
    print(caption)


if __name__ == "__main__":
    main()
