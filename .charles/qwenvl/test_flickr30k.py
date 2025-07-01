import os
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

def test_flickr30k():
    # Load dataset from local folder
    dataset_path = "../.data/hf/datasets/lmms-lab/flickr30k"
    
    print("Loading Flickr30k dataset...")
    try:
        # Load all splits info
        if os.path.exists(dataset_path):
            ds_all = load_dataset(dataset_path)
            print(f"Loaded dataset from local path: {dataset_path}")
        else:
            ds_all = load_dataset("lmms-lab/flickr30k")
            print("Loaded dataset from HuggingFace Hub")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Print available splits and total images
    print("\nAvailable splits:", list(ds_all.keys()))
    total_images = sum(len(ds_all[split]) for split in ds_all.keys())
    print(f"Total images across all splits: {total_images}")

    # For backward compatibility, keep loading the 'test' split for the rest of the code
    ds = ds_all["test"]
    
    # Dataset statistics
    print(f"\nDataset size: {len(ds)} samples")
    print(f"Dataset features: {ds.features}")
    
    # Sample a few examples
    print("\n--- Sample Examples ---")
    for i in range(min(3, len(ds))):
        sample = ds[i]
        print(f"\nExample {i+1}:")
        print(f"Image shape: {sample['image'].size if hasattr(sample['image'], 'size') else 'N/A'}")
        print(f"Captions ({len(sample['caption'])} total):")
        for j, caption in enumerate(sample['caption'][:2]):  # Show first 2 captions
            print(f"  {j+1}: {caption}")
        if len(sample['caption']) > 2:
            print(f"  ... and {len(sample['caption']) - 2} more captions")
    
    # Display first image with captions
    try:
        first_sample = ds[0]
        plt.figure(figsize=(10, 6))
        plt.imshow(first_sample['image'])
        plt.axis('off')
        plt.title("Sample Image from Flickr30k")
        
        # Show captions below image
        caption_text = "\n".join([f"{i+1}. {cap}" for i, cap in enumerate(first_sample['caption'])])
        plt.figtext(0.1, 0.02, f"Captions:\n{caption_text}", fontsize=10, wrap=True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error displaying image: {e}")

if __name__ == "__main__":
    test_flickr30k()



"""
Loading Flickr30k dataset...
Loaded dataset from local path: ../.data/hf/datasets/flickr30k

Available splits: ['test']
Total images across all splits: 31783

Dataset size: 31783 samples
Dataset features: {'image': Image(mode=None, decode=True, id=None), 'caption': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), 'sentids': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), 'img_id': Value(dtype='string', id=None), 'filename': Value(dtype='string', id=None)}

--- Sample Examples ---

Example 1:
Image shape: (333, 500)
Captions (5 total):
  1: Two young guys with shaggy hair look at their hands while hanging out in the yard .
  2: Two young  White males are outside near many bushes .
  ... and 3 more captions

Example 2:
Image shape: (500, 374)
Captions (5 total):
  1: Several men in hard hats are operating a giant pulley system .
  2: Workers look down from up above on a piece of equipment .
  ... and 3 more captions

Example 3:
Image shape: (375, 500)
Captions (5 total):
  1: A child in a pink dress is climbing up a set of stairs in an entry way .
  2: A little girl in a pink dress going into a wooden cabin .
  ... and 3 more captions
"""