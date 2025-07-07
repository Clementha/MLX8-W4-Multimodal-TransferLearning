"""Gradio Demo for Multimodal Transfer Learning

Interactive web interface for testing the trained multimodal model with
random Flickr30k test images and real-time caption prediction.

Author: Charles Cai (github.com/charles-cai)
"""
import gradio as gr
import torch
import random
import os
from pathlib import Path
from PIL import Image
import time
import logging
import glob

# Load .env file for environment variables
from dotenv import load_dotenv
load_dotenv()

# Set a random seed for this session using OS entropy
import struct
_seed = struct.unpack("I", os.urandom(4))[0]
random.seed(_seed)
torch.manual_seed(_seed)

# Import the main class and configurations
from multimodal_transfer_learing import (
    MultimodalTransferLearning, 
    OUTPUT_DIR_CACHE, 
    VISION_ENCODER, 
    TRAINING_DATASET,
    SPLIT_CAPTIONS,
    INFERENCE_PROMPT,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_P,
    INFERENCE_TOP_K,
    log
)

class GradioDemo:
    def __init__(self):
        """Initialize the demo with model and test dataset"""
        self.model = None
        self.test_dataset = None
        self.raw_dataset = None  # Store raw dataset for image access
        self.current_image = None
        self.current_embedding = None
        self.selected_local_image = None  # Track selected local image
        self.image_source = None  # "flickr" or "local"
        
        # Setup logging for demo
        log.info("🎮 Initializing Gradio Demo...")
        
        try:
            # Load the trained model
            log.info("📁 Loading trained model...")
            self.model = MultimodalTransferLearning(mode="run")
            log.info("✅ Model loaded successfully!")
            
            # Load test dataset and raw images
            self._load_test_dataset()
            self._load_raw_dataset()
            
        except Exception as e:
            log.error(f"❌ Failed to initialize demo: {e}")
            raise
    
    def _load_test_dataset(self):
        """Load the cached test dataset"""
        try:
            # Load the same cached dataset used for training/testing
            split_suffix = "_split" if SPLIT_CAPTIONS else "_no_split"
            cache_file = OUTPUT_DIR_CACHE / f"{VISION_ENCODER}_{TRAINING_DATASET}_complete{split_suffix}.pt"
            
            if not cache_file.exists():
                raise FileNotFoundError(f"Test dataset cache not found: {cache_file}")
            
            log.info(f"📁 Loading test dataset from {cache_file}")
            cache_data = torch.load(cache_file, map_location='cpu', weights_only=False)
            
            # Extract test dataset
            self.test_dataset = cache_data['dataset']['test']
            
            log.info(f"✅ Loaded test dataset with {len(self.test_dataset)} samples")
            
        except Exception as e:
            log.error(f"❌ Failed to load test dataset: {e}")
            raise
    
    def _load_raw_dataset(self):
        """Load the raw Flickr30k dataset for image access"""
        try:
            log.info("📁 Loading raw Flickr30k dataset for images...")
            from datasets import load_dataset
            
            # Load raw dataset - same as used for embedding generation
            self.raw_dataset = load_dataset("lmms-lab/flickr30k", split="test", token=os.getenv("HF_TOKEN"))
            
            log.info(f"✅ Loaded raw dataset with {len(self.raw_dataset)} images")
            
        except Exception as e:
            log.error(f"❌ Failed to load raw dataset: {e}")
            # Create dummy dataset if loading fails
            self.raw_dataset = None
    
    def list_local_images(self, images_dir="./images"):
        """Return a list of image filenames in the given directory."""
        image_files = []
        try:
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"):
                image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            # Only return basenames for dropdown
            return [""] + [os.path.basename(f) for f in sorted(image_files)]
        except Exception as e:
            log.warning(f"Failed to list local images: {e}")
            return [""]

    def get_random_sample(self):
        """Get a random sample from test dataset with actual image and matching embedding"""
        try:
            if not self.test_dataset:
                return None, "❌ Test dataset not loaded", ""
            
            # Try to load actual image from raw dataset first
            image_to_display = None
            caption_to_display = ""
            
            if self.raw_dataset and len(self.raw_dataset) > 0:
                try:
                    # Get random image from raw dataset
                    raw_idx = random.randint(0, len(self.raw_dataset) - 1)
                    raw_sample = self.raw_dataset[raw_idx]
                    
                    # Get the PIL image
                    if 'image' in raw_sample and raw_sample['image'] is not None:
                        image_to_display = raw_sample['image']
                        
                        # Get ALL captions from raw sample and format them properly
                        if 'caption' in raw_sample and raw_sample['caption']:
                            if isinstance(raw_sample['caption'], list):
                                # Format multiple captions, each on a new line with numbering
                                formatted_captions = []
                                for i, caption in enumerate(raw_sample['caption'], 1):
                                    formatted_captions.append(f"{i}. {caption.strip()}")
                                caption_to_display = "\n".join(formatted_captions)
                            else:
                                caption_to_display = f"1. {str(raw_sample['caption']).strip()}"
                        else:
                            caption_to_display = "No captions available"
                        
                        # Generate embedding for this specific image using the vision encoder
                        try:
                            log.info(f"🔄 Generating embedding for displayed image...")
                            
                            # Process the image through vision encoder
                            if VISION_ENCODER == "vit":
                                pixels = self.model.processor(images=image_to_display, return_tensors="pt").pixel_values
                            else:
                                pixels = self.model.processor(images=image_to_display, return_tensors="pt").pixel_values
                            
                            pixels = pixels.to(self.model.device)
                            
                            with torch.no_grad():
                                if VISION_ENCODER == "vit":
                                    embedding = self.model.vision_encoder(pixels)[0][:, 0]  # CLS token
                                else:
                                    embedding = self.model.vision_encoder.get_image_features(pixel_values=pixels)[0]
                            
                            # Store the embedding that matches the displayed image
                            self.current_embedding = embedding.cpu()
                            self.current_caption = caption_to_display
                            
                            log.info(f"🎲 Loaded actual image {raw_idx} with matching embedding")
                            log.info(f"📝 Found {len(raw_sample['caption']) if isinstance(raw_sample['caption'], list) else 1} captions for this image")
                            
                        except Exception as e:
                            log.warning(f"⚠️ Failed to generate embedding for displayed image: {e}")
                            # Fallback to random test dataset sample
                            random_idx = random.randint(0, len(self.test_dataset) - 1)
                            sample = self.test_dataset[random_idx]
                            self.current_embedding = sample['embedding']
                            if not isinstance(self.current_embedding, torch.Tensor):
                                self.current_embedding = torch.tensor(self.current_embedding)
                            
                    else:
                        log.warning("⚠️ Raw sample has no image, using test dataset sample")
                        # Fallback to test dataset
                        random_idx = random.randint(0, len(self.test_dataset) - 1)
                        sample = self.test_dataset[random_idx]
                        self.current_embedding = sample['embedding']
                        if not isinstance(self.current_embedding, torch.Tensor):
                            self.current_embedding = torch.tensor(self.current_embedding)
                        caption_to_display = sample['original_caption'] if 'original_caption' in sample else "No caption available"
                        image_to_display = Image.new('RGB', (224, 224), color='lightgray')
                        
                except Exception as e:
                    log.warning(f"⚠️ Failed to load image from raw dataset: {e}")
                    # Fallback to test dataset
                    random_idx = random.randint(0, len(self.test_dataset) - 1)
                    sample = self.test_dataset[random_idx]
                    self.current_embedding = sample['embedding']
                    if not isinstance(self.current_embedding, torch.Tensor):
                        self.current_embedding = torch.tensor(self.current_embedding)
                    caption_to_display = sample['original_caption'] if 'original_caption' in sample else "No caption available"
                    image_to_display = Image.new('RGB', (224, 224), color='lightblue')
            else:
                log.warning("⚠️ Raw dataset not available, using test dataset sample")
                # Fallback to test dataset only
                random_idx = random.randint(0, len(self.test_dataset) - 1)
                sample = self.test_dataset[random_idx]
                self.current_embedding = sample['embedding']
                if not isinstance(self.current_embedding, torch.Tensor):
                    self.current_embedding = torch.tensor(self.current_embedding)
                caption_to_display = sample['original_caption']
                image_to_display = Image.new('RGB', (224, 224), color='lightcoral')
            
            # Log first caption only for brevity
            first_caption = caption_to_display.split('\n')[0] if caption_to_display else "No caption"
            log.info(f"🎲 Random sample selected with first caption: {first_caption[:50]}...")
            
            self.selected_local_image = None  # Reset local image selection on random
            self.image_source = "flickr"
            return image_to_display, caption_to_display, ""  # Reset prediction
            
        except Exception as e:
            log.error(f"❌ Failed to get random sample: {e}")
            placeholder_image = Image.new('RGB', (300, 300), color='red')
            return placeholder_image, f"❌ Error: {str(e)}", ""
    
    def load_local_image(self, filename, images_dir="./images"):
        """Load a local image by filename and set as current image for prediction."""
        if not filename:
            self.selected_local_image = None
            self.image_source = None
            return None, "", ""
        path = os.path.join(images_dir, filename)
        if not os.path.exists(path):
            self.selected_local_image = None
            self.image_source = None
            return None, f"❌ File not found: {filename}", ""
        try:
            image = Image.open(path).convert("RGB")
            self.selected_local_image = image
            self.image_source = "local"
            self.current_embedding = None  # Will be set on predict
            return image, "Local image selected. No reference captions.", ""
        except Exception as e:
            self.selected_local_image = None
            self.image_source = None
            log.error(f"Failed to load local image: {e}")
            return None, f"❌ Error loading image: {e}", ""

    def predict_caption(self, image_display, reference_caption, prev_predicted_caption,
                       temperature, top_p, top_k):
        """Generate caption prediction for current sample and append to previous predictions"""
        try:
            # Use the correct image source for prediction
            if self.image_source == "local" and self.selected_local_image is not None:
                image_to_use = self.selected_local_image
                # Generate embedding for local image
                if VISION_ENCODER == "vit":
                    pixels = self.model.processor(images=image_to_use, return_tensors="pt").pixel_values
                else:
                    pixels = self.model.processor(images=image_to_use, return_tensors="pt").pixel_values
                pixels = pixels.to(self.model.device)
                with torch.no_grad():
                    if VISION_ENCODER == "vit":
                        embedding = self.model.vision_encoder(pixels)[0][:, 0]
                    else:
                        embedding = self.model.vision_encoder.get_image_features(pixel_values=pixels)[0]
                self.current_embedding = embedding.cpu()
            elif self.image_source == "flickr":
                # Use the embedding from the random sample
                pass
            else:
                return prev_predicted_caption or "⚠️ Please load a random sample or select a local image first"

            if self.current_embedding is None:
                return prev_predicted_caption or "⚠️ Please load a random sample or select a local image first"
            if self.model is None:
                return prev_predicted_caption or "❌ Model not loaded"
            log.info("🔮 Generating prediction...")
            
            # Ensure embedding is a proper tensor
            if not isinstance(self.current_embedding, torch.Tensor):
                self.current_embedding = torch.tensor(self.current_embedding)
            
            # Prepare embedding tensor - add batch dimension if needed
            if self.current_embedding.dim() == 1:
                img_embedding = self.current_embedding.unsqueeze(0).to(self.model.device)
            else:
                img_embedding = self.current_embedding.to(self.model.device)
            
            # Generate caption using model's method
            predictions = self.model._generate_captions_with_prompt(
                img_embedding, 
                prompt=INFERENCE_PROMPT, 
                max_new_tokens=100,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            
            if predictions and len(predictions) > 0:
                prediction = predictions[0]
                log.info(f"✅ Generated prediction: {prediction[:50]}...")
                # Prefix with temperature, top-p, top-k
                prefix = f"[t{temperature:.2f} p{top_p:.2f} k{top_k}]: "
                prediction_line = prefix + prediction
                # Append new prediction to previous predictions
                if prev_predicted_caption and prev_predicted_caption.strip():
                    return prev_predicted_caption.rstrip() + "\n" + prediction_line
                else:
                    return prediction_line
            else:
                prefix = f"[T{temperature:.2f} P{top_p:.2f} K{top_k}]: "
                fail_msg = prefix + "❌ Failed to generate prediction"
                return prev_predicted_caption.rstrip() + "\n" + fail_msg if prev_predicted_caption else fail_msg
                
        except Exception as e:
            log.error(f"❌ Prediction failed: {e}")
            if prev_predicted_caption and prev_predicted_caption.strip():
                return prev_predicted_caption.rstrip() + f"\n❌ Error: {str(e)}"
            else:
                return f"❌ Error: {str(e)}"

    def get_model_files_info(self):
        """Return a Markdown-formatted list of all files used by the model."""
        files_info = []
        try:
            # Embedding cache file
            split_suffix = "_split" if SPLIT_CAPTIONS else "_no_split"
            cache_file = OUTPUT_DIR_CACHE / f"{VISION_ENCODER}_{TRAINING_DATASET}_complete{split_suffix}.pt"
            if cache_file.exists():
                files_info.append(f"- **Embedding Cache:** `{cache_file}`")
            else:
                files_info.append(f"- **Embedding Cache:** `{cache_file}` (not found)")

            # Optionally, add any other files used by the model
            # For example, config files, tokenizer, etc.
            if hasattr(self.model, "config_path"):
                files_info.append(f"- **Config:** `{self.model.config_path}`")
            if hasattr(self.model, "tokenizer_path"):
                files_info.append(f"- **Tokenizer:** `{self.model.tokenizer_path}`")

            # Add images directory if used
            images_dir = Path("./images")
            if images_dir.exists():
                files_info.append(f"- **Local Images Directory:** `{images_dir.resolve()}`")

        except Exception as e:
            files_info.append(f"- ⚠️ Error collecting file info: {e}")
        return "\n".join(files_info) if files_info else "_No files found_"

    def create_interface(self):
        """Create the Gradio interface"""

        def get_env_vars():
            import os
            envs = os.environ
            # Format as markdown table
            lines = ["| Variable | Value |", "|---|---|"]
            for k, v in sorted(envs.items()):
                lines.append(f"| `{k}` | `{v}` |")
            return "\n".join(lines)

        with gr.Blocks(title="Multimodal Transfer Learning Demo", theme=gr.themes.Soft()) as demo:
            # Inject custom CSS for larger font sizes
            gr.HTML("""
            <style>
            .caption-large-font textarea, .caption-large-font .prose { font-size: 1.25rem !important; }
            </style>
            """)
            with gr.Tab("Demo Description"):
                gr.Markdown(f"""
                # 🎨 Multimodal Transfer Learning Demo

                Interactive web interface for testing the trained multimodal model with
                random Flickr30k test images and real-time caption prediction.

                **Instructions:**
                1. Click "Random Flickr Image and Caption" to load a test sample
                2. Review the reference captions (Flickr30k has 5 captions per image)
                3. Click "Predict" to generate model's caption
                4. Compare reference vs prediction!
                """)
            with gr.Tab("Model Card"):
                gr.Markdown(f"""
                ## 🧾 Model Card

                **Model Configuration:**
                - Vision Encoder: `{VISION_ENCODER.upper()}`
                - Training Dataset: `{TRAINING_DATASET}`
                - Caption Splitting: `{'Enabled' if SPLIT_CAPTIONS else 'Disabled'}`

                ## 📊 Model Information

                **Dataset Stats:**
                - Test samples: `{len(self.test_dataset) if self.test_dataset else 'Loading...'}`
                - Raw images: `{len(self.raw_dataset) if self.raw_dataset else 'Loading...'}`
                - Captions per image: `5 (Flickr30k standard)`
                - Inference prompt: `"{INFERENCE_PROMPT[:60]}..."`

                ## 📁 Files Used by Model
                {self.get_model_files_info()}
                """)
            with gr.Tab("Environment Variables"):
                env_md = gr.Markdown(get_env_vars())

            with gr.Tab("Demo"):
                with gr.Group():
                    gr.Markdown("## 🎲 Random Sample")
                    
                    with gr.Row():
                        # Image column: 40%
                        with gr.Column(scale=4):
                            image_display = gr.Image(
                                label="Test Image", 
                                type="pil",
                                height=400,
                                interactive=False
                            )

                            random_button = gr.Button(
                                "🎲 Random Flickr Image and Caption", 
                                variant="primary",
                                size="lg"
                            )
                            
                            # Dropdown for local images
                            local_image_dropdown = gr.Dropdown(
                                choices=self.list_local_images(),
                                label="Or select a local image from ./images",
                                value="",
                                interactive=True
                            )
                        
                        # Captions column: 60%
                        with gr.Column(scale=6):
                            reference_caption = gr.Textbox(
                                label="📝 Reference Captions (Ground Truth) - 5 captions per image",
                                lines=6,
                                max_lines=6,
                                interactive=False,
                                elem_classes=["caption-large-font"],
                                placeholder="Click the random button to load reference captions...\n\nEach Flickr30k image has 5 different captions,\nall will be displayed here numbered 1-5."
                            )
                            
                            predicted_caption = gr.Textbox(
                                label="🤖 Predicted Caption (Model Output)",
                                lines=12,
                                max_lines=12,
                                interactive=False,
                                elem_classes=["caption-large-font"],
                                placeholder="Click 'Predict' to generate caption..."
                            )

                            # Add inference parameter sliders
                            temperature_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=INFERENCE_TEMPERATURE,
                                step=0.01, label="Temperature"
                            )
                            top_p_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=INFERENCE_TOP_P,
                                step=0.01, label="Top-p"
                            )
                            top_k_slider = gr.Slider(
                                minimum=0, maximum=100, value=INFERENCE_TOP_K,
                                step=1, label="Top-k"
                            )
                            
                            predict_button = gr.Button(
                                "🔮 Predict", 
                                variant="secondary",
                                size="lg"
                            )
                
                # Event handlers
                random_button.click(
                    fn=self.get_random_sample,
                    inputs=[],
                    outputs=[image_display, reference_caption, predicted_caption]
                )

                local_image_dropdown.change(
                    fn=lambda filename: self.load_local_image(filename),
                    inputs=[local_image_dropdown],
                    outputs=[image_display, reference_caption, predicted_caption]
                )
                
                predict_button.click(
                    fn=self.predict_caption,
                    inputs=[image_display, reference_caption, predicted_caption,
                            temperature_slider, top_p_slider, top_k_slider],
                    outputs=[predicted_caption]
                )
                
                # Load initial sample
                demo.load(
                    fn=self.get_random_sample,
                    inputs=[],
                    outputs=[image_display, reference_caption, predicted_caption]
                )
        return demo

def launch_demo(share=False, server_port=8081):
    """Launch the Gradio demo"""
    try:
        log.info("🚀 Starting Gradio Demo...")
        
        # Initialize demo
        demo_app = GradioDemo()
        
        # Create interface
        interface = demo_app.create_interface()
        
        # Launch
        log.info(f"🌐 Launching on port {server_port}...")
        interface.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0" if share else "127.0.0.1",
            show_error=True,
            inbrowser=True
        )
        
    except Exception as e:
        log.error(f"❌ Failed to launch demo: {e}")
        raise

if __name__ == "__main__":
    import argparse

    # Read port from environment variable, default to 8081 if not set
    env_port = int(os.environ.get("GRADIO_PORT", 8081))
    
    parser = argparse.ArgumentParser(description="Launch Gradio Demo")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=env_port, help="Server port (default: from .env PORT or 8081)")
    
    args = parser.parse_args()
    
    launch_demo(share=args.share, server_port=args.port)
