## Data / Models offline preparation


```bash
git lfs install

# datasets: test_flickr30k.py
git clone https://huggingface.co/datasets/lmms-lab/flickr30k            ../.data/hf/datasets/lmms-labs/flickr30k

# models:  

git clone https://huggingface.co/google/vit-base-patch16-224            ../.data/hf/datasets/vit-base-patch16-224
git clone https://huggingface.co/openai/clip-vit-base-patch32           ../.data/hf/models/openai/clip-vit-base-patch32


# QWen3-0.6B-Base
git clone https://huggingface.co/Qwen/Qwen3-0.6B                   ../.data/hf/models/Qwen/Qwen3-0.6B
git clone https://huggingface.co/Qwen/Qwen3-0.6B-Base                   ../.data/hf/models/Qwen/Qwen3-0.6B-Base
git clone https://huggingface.co/Qwen/Qwen3-1.7B-Base                   ../.data/hf/models/Qwen/Qwen3-1.7B-Base
git clone https://huggingface.co/Qwen/Qwen3-4B-Base                   ../.data/hf/models/Qwen/Qwen3-4B-Base

uv run test_qwen3_base.py


# Image-Text custom training dataset generation

git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct            ../.data/hf/models/Qwen/Qwen2.5-VL-7B-Instruct
git clone https://hugggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit  ../.data/hf/models//unsloth/Qwen2.5-7B-Instruct-bnb-4bit

uv run test_qwen25_vl.py # OOM on 3090
uv run test_unsloth_4bit.py # flash_attn uv messed up, need to use 3.10 and old cuda

# Ollama quantized qwen25_vl
ollama pull qwen2.5vl
ollama serve
uv run test_ollama_vision.py
uv flaticon_dataset_gen.py
```

Also on Hugging Face, but we downloaded the **CIFAR-10** dataset here: https://www.cs.toronto.edu/~kriz/cifar.html
And **FashionMnist** dataset from: https://github.com/zalandoresearch/fashion-mnist.git, images are in its `data` folder. `uv run test_fashion_mnist.py` to show a few sample labelled images.