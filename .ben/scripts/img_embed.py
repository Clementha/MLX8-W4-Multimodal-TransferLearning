import transformers

# To count the number of parameters in the model
params = lambda m: sum(p.numel() for p in m.parameters())
# To get model size in MB
model_size_mb = lambda m: params(m) * 4 / (1024 ** 2)

# Load ViT
print("Loading pretrained ViT model...")
pretrained_vis = transformers.AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
print("ViT:", params(pretrained_vis))
print("ViT size (MB):", model_size_mb(pretrained_vis))