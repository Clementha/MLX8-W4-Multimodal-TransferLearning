#!/usr/bin/env python3
"""
gen_reef_metadata_streaming_batched.py

Same as gen_reef_metadata_streaming_batched, but now resilient to corrupted images:
 - Loads model in pure FP16 on GPU
 - Processes images in batches of BATCH_SIZE
 - Wraps generate() in torch.autocast for speed
 - Streams results to Parquet safely
 - Skips unreadable/corrupt images without crashing
"""
from pathlib import Path
import sys

import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from PIL import UnidentifiedImageError
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# —— USER CONFIG — adjust as needed ——
MODEL_DIR       = Path("models/Qwen2.5-VL-3B-Instruct")
REEF_IMG_DIR    = Path("data/reef_data/images/images")
OUTPUT_PARQUET  = Path("data/reef_data/metadata.parquet")

TEST_ONLY  = False    # True = print each, process only N_TEST
N_TEST     = 10       # number if TEST_ONLY
NUM_IMAGES = None     # limit for non-test, None=all

BATCH_SIZE     = 8    # images per inference batch
MAX_NEW_TOKENS = 128
NUM_BEAMS      = 2

SYSTEM_PROMPT = (
    "You are an expert marine biologist assessing coral reef health.  "
    "For each image, produce a JSON object with exactly two keys:\n"
    "  • \"description\": a very short paragraph describing substrate (live coral, rubble, rock),\n"
    "    coral growth forms with approximate percent cover, and any notable features (sponges, algae, bleaching),\n"
    "  • \"health_status\": either \"HEALTHY\" or \"DEGRADED\".\n"
    "Ignore any visible diving lines—each image is a single frame.\n"
)
USER_PROMPT = "Analyse the image and output the JSON as specified."
# ——————————————————————————————————————

def load_model_and_processor(model_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, device_map={"": device}
    ).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor, device


def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def generate_caption(model, processor, device, text, vision_input) -> str:
    """
    Run model.generate given preprocessed text prompt and vision tensor.
    """
    inputs = processor(
        text=[text], images=[vision_input], padding=True, return_tensors="pt"
    ).to(device)
    with torch.autocast(device.type, dtype=torch.float16):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=False,
        )
    prompt_len = inputs.input_ids.shape[1]
    trimmed = [ids[prompt_len:] for ids in generated_ids]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    # sanity
    if not MODEL_DIR.is_dir():
        print(f"Model dir not found: {MODEL_DIR}", file=sys.stderr)
        sys.exit(1)
    if not REEF_IMG_DIR.is_dir():
        print(f"Image dir not found: {REEF_IMG_DIR}", file=sys.stderr)
        sys.exit(1)

    model, processor, device = load_model_and_processor(MODEL_DIR)
    img_paths = sorted(REEF_IMG_DIR.glob("*.jpg"))
    if TEST_ONLY:
        img_paths = img_paths[:N_TEST]
    elif NUM_IMAGES is not None:
        img_paths = img_paths[:NUM_IMAGES]

    writer = None

    try:
        # process in batches
        for batch in tqdm(chunked(img_paths, BATCH_SIZE), desc="Batched captioning", unit="batch"):
            valid_imgs = []
            texts      = []
            visions    = []

            # 1) Pre-filter & prepare all valid images in this batch
            for img_path in batch:
                try:
                    msgs = [
                        {"role":"system","content":SYSTEM_PROMPT},
                        {"role":"user",  "content":[
                            {"type":"image", "image":str(img_path)},
                            {"type":"text",  "text":USER_PROMPT},
                        ]},
                    ]
                    text = processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    vision, _ = process_vision_info(msgs)
                    valid_imgs.append(img_path.name)
                    texts.append(text)
                    visions.append(vision[0])
                except (UnidentifiedImageError, OSError):
                    print(f"Skipping bad image {img_path.name}", file=sys.stderr)
                    continue

            # skip empty batches
            if not texts:
                continue

            # 2) Batch-generate
            inputs = processor(
                text=texts,
                images=visions,
                padding=True,
                return_tensors="pt"
            ).to(device)

            with torch.autocast(device.type, dtype=torch.float16):
                batch_outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                )

            # 3) Decode & collect rows
            prompt_len = inputs.input_ids.shape[1]
            trimmed    = [ids[prompt_len:] for ids in batch_outputs]
            caps       = processor.batch_decode(trimmed, skip_special_tokens=True)

            rows = []
            for fname, cap in zip(valid_imgs, caps):
                if TEST_ONLY:
                    print(f"{fname} → {cap}")
                rows.append({
                    "filename": fname,
                    "caption":  cap.strip(),
                    "split":    "test",
                })

            # 4) Stream to Parquet
            df    = pd.DataFrame(rows, columns=["filename","caption","split"])
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema)
            writer.write_table(table)

    finally:
        if writer:
            writer.close()
            print(f"Saved output to {OUTPUT_PARQUET}")
        else:
            print("No data written.")


if __name__ == "__main__":
    main()
