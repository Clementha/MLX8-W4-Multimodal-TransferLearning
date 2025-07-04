#!/usr/bin/env python3
"""
gen_reef_metadata.py

Generate captions for coral reef images, either in test mode (first N images)
or full mode, and save metadata parquet matching the Flickr30k format.
"""
from pathlib import Path
import sys

import torch
import pandas as pd
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Toggle test mode: if True, only process N_TEST images
TEST_ONLY: bool = True
N_TEST: int = 10
# When TEST_ONLY is False, you can still limit via NUM_IMAGES (set to None for all)
# for now use 1000 as a default limit
NUM_IMAGES: int | None = 1000
# ————————————————————————————————
# —— USER CONFIG — adjust these prompts ——
SYSTEM_PROMPT = (
    "You are an expert marine biologist assessing coral reef health.  "
    "For each image, produce a JSON object with exactly two keys:\n"
    "  • \"description\": a very short paragraph describing substrate (live coral, rubble, rock),\n"
    "    coral growth forms with approximate percent cover, and any notable features (sponges, algae, bleaching),\n"
    "  • \"health_status\": either \"HEALTHY\" or \"DEGRADED\".\n"
    "Ignore any visible diving lines—each image is a single frame.\n"
)
USER_PROMPT = "Analyse the image and output the JSON as specified."
NUM_OUTPUT_TOKENS =128
# ——————————————————————————————————————


# —— USER CONFIG — adjust paths & modes here ——
MODEL_DIR: Path = (
    Path(__file__).resolve().parent.parent
    / "models" / "Qwen2.5-VL-3B-Instruct"
)
REEF_IMG_DIR: Path = (
    Path(__file__).resolve().parent.parent
    / "data" / "reef_data" / "images"
)
OUTPUT_PARQUET: Path = (
    Path(__file__).resolve().parent.parent
    / "data" / "reef_data" / "metadata.parquet"
)



def load_model_and_processor(model_dir: Path) -> tuple[torch.nn.Module, AutoProcessor, torch.device]:
    """
    Load the Qwen vision-language model (fp16) and its processor.

    Args:
        model_dir: Directory containing the fine-tuned model.
    Returns:
        model, processor, device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load in fp16 for memory savings
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, device_map={"": device}
    ).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor, device


def generate_caption(
    model: torch.nn.Module,
    processor: AutoProcessor,
    device: torch.device,
    image_path: Path,
) -> str:
    """
    Generate a single caption for one image.

    Args:
        model: Loaded Qwen VL model.
        processor: Paired processor for text+vision.
        device: torch device.
        image_path: Path to input JPEG.
    Returns:
        Generated caption string.
    """
    # Use a system message + user message
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text",  "text": USER_PROMPT},
        ]},
    ]
    # Apply chat template + generation prompt
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Extract vision inputs only
    vision_inputs, _ = process_vision_info(messages)
    # Prepare inputs for model.generate
    inputs = processor(
        text=[text], images=vision_inputs,
        padding=True, return_tensors="pt"
    ).to(device)
    # Generate greedily with small beam
    generated_ids = model.generate(
        **inputs, max_new_tokens=NUM_OUTPUT_TOKENS, num_beams=2, do_sample=False
    )
    # Remove prompt tokens and decode
    prompt_len = inputs.input_ids.shape[1]
    trimmed = [ids[prompt_len:] for ids in generated_ids]
    caption = processor.batch_decode(
        trimmed, skip_special_tokens=True
    )[0].strip()
    return caption


def main() -> None:
    """
    Main entry: generate captions, construct DataFrame, and save to parquet.
    """
    # sanity checks
    if not MODEL_DIR.is_dir():
        print(f"Model dir not found: {MODEL_DIR}", file=sys.stderr)
        sys.exit(1)
    if not REEF_IMG_DIR.is_dir():
        print(f"Reef images dir not found: {REEF_IMG_DIR}", file=sys.stderr)
        sys.exit(1)

    model, processor, device = load_model_and_processor(MODEL_DIR)

    # collect file paths
    img_paths = sorted(REEF_IMG_DIR.glob("*.jpg"))
    # apply test-only or general limit
    if TEST_ONLY:
        img_paths = img_paths[:N_TEST]
    elif NUM_IMAGES is not None:
        img_paths = img_paths[:NUM_IMAGES]

    rows: list[dict[str, str]] = []
    # iterate with progress bar
    for img_path in tqdm(img_paths, desc="Captioning reefs", unit="img"):
        cap = generate_caption(model, processor, device, img_path)

        if TEST_ONLY:                             # <-- only in test mode
            print(f"{img_path.name} → {cap}")     # <-- print caption

        rows.append({
            "filename": img_path.name,
            "caption":  cap,
            "split":    "test",
        })


    # build DataFrame and save
    df = pd.DataFrame(rows, columns=["filename", "caption", "split"])
    print(f"\nGenerated {len(df)} captions, saving to {OUTPUT_PARQUET}")
    df.to_parquet(OUTPUT_PARQUET, index=False)


if __name__ == "__main__":
    main()
