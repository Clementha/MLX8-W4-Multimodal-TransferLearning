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

# Import the main class and configurations
from multimodal_transfer_learing import (
    MultimodalTransferLearning, 
    OUTPUT_DIR_CACHE, 
    VISION_ENCODER, 
    TRAINING_DATASET,
    SPLIT_CAPTIONS,
    INFERENCE_PROMPT,
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
                caption_to_display = sample['original_caption'] if 'original_caption' in sample else "No caption available"
                image_to_display = Image.new('RGB', (224, 224), color='lightcoral')
            
            # Log first caption only for brevity
            first_caption = caption_to_display.split('\n')[0] if caption_to_display else "No caption"
            log.info(f"🎲 Random sample selected with first caption: {first_caption[:50]}...")
            
            return image_to_display, caption_to_display, ""  # Reset prediction
            
        except Exception as e:
            log.error(f"❌ Failed to get random sample: {e}")
            placeholder_image = Image.new('RGB', (300, 300), color='red')
            return placeholder_image, f"❌ Error: {str(e)}", ""
    
    def predict_caption(self, image_display, reference_caption):
        """Generate caption prediction for current sample"""
        try:
            if self.current_embedding is None:
                return "⚠️ Please load a random sample first"
            
            if self.model is None:
                return "❌ Model not loaded"
            
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
                max_new_tokens=100
            )
            
            if predictions and len(predictions) > 0:
                prediction = predictions[0]
                log.info(f"✅ Generated prediction: {prediction[:50]}...")
                return prediction
            else:
                return "❌ Failed to generate prediction"
                
        except Exception as e:
            log.error(f"❌ Prediction failed: {e}")
            return f"❌ Error: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(title="Multimodal Transfer Learning Demo", theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"""
            # 🎨 Multimodal Transfer Learning Demo
            
            **Model Configuration:**
            - Vision Encoder: `{VISION_ENCODER.upper()}`
            - Training Dataset: `{TRAINING_DATASET}`
            - Caption Splitting: `{'Enabled' if SPLIT_CAPTIONS else 'Disabled'}`
            
            **Instructions:**
            1. Click "Random Flickr Image and Caption" to load a test sample
            2. Review the reference captions (Flickr30k has 5 captions per image)
            3. Click "Predict" to generate model's caption
            4. Compare reference vs prediction!
            """)
            
            with gr.Group():
                gr.Markdown("## 🎲 Random Sample")
                
                with gr.Row():
                    with gr.Column(scale=1):
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
                    
                    with gr.Column(scale=1):
                        reference_caption = gr.Textbox(
                            label="📝 Reference Captions (Ground Truth) - 5 captions per image",
                            lines=8,
                            max_lines=12,
                            interactive=False,
                            placeholder="Click the random button to load reference captions...\n\nEach Flickr30k image has 5 different captions,\nall will be displayed here numbered 1-5."
                        )
                        
                        predicted_caption = gr.Textbox(
                            label="🤖 Predicted Caption (Model Output)",
                            lines=4,
                            max_lines=6,
                            interactive=False,
                            placeholder="Click 'Predict' to generate caption..."
                        )
                        
                        predict_button = gr.Button(
                            "🔮 Predict", 
                            variant="secondary",
                            size="lg"
                        )
            
            # Statistics and info
            with gr.Group():
                gr.Markdown("## 📊 Model Information")
                
                with gr.Row():
                    gr.Markdown(f"""
                    **Dataset Stats:**
                    - Test samples: `{len(self.test_dataset) if self.test_dataset else 'Loading...'}`
                    - Raw images: `{len(self.raw_dataset) if self.raw_dataset else 'Loading...'}`
                    - Captions per image: `5 (Flickr30k standard)`
                    - Inference prompt: `"{INFERENCE_PROMPT[:60]}..."`
                    """)
            
            # Event handlers
            random_button.click(
                fn=self.get_random_sample,
                inputs=[],
                outputs=[image_display, reference_caption, predicted_caption]
            )
            
            predict_button.click(
                fn=self.predict_caption,
                inputs=[image_display, reference_caption],
                outputs=[predicted_caption]
            )
            
            # Load initial sample
            demo.load(
                fn=self.get_random_sample,
                inputs=[],
                outputs=[image_display, reference_caption, predicted_caption]
            )
        
        return demo

def launch_demo(share=False, server_port=7860):
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
    
    parser = argparse.ArgumentParser(description="Launch Gradio Demo")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    
    args = parser.parse_args()
    
    launch_demo(share=args.share, server_port=args.port)
