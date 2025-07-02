### 🗒️ Project one-pager – **Multimodal Transfer-Learning prototype**

| Item                           | Description                                                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                       |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Goal**                       | Turn the **Qwen 3-0.6 B-Base** language model into a light vision–language system that can caption or answer questions about images, using only the **Flickr30k** captions (\~30 k pairs) for fine-tuning.                                                                                                                                                                                                 |                                                                                                                                                       |
| **High-level idea**            | 1) Keep a strong **vision encoder** *frozen* (choose **ViT-B/16** or **CLIP ViT-B/32**).<br>2) Map its single 768-d / 512-d embedding into Qwen’s 4 096-d token space with a tiny 2-layer **MLP adapter** (trainable).<br>3) Optionally unfreeze the **top *K*** decoder blocks of Qwen (default *K = 3*) to let the LM specialise.<br>4) Optimise only adapter + those K blocks on caption cross-entropy. |                                                                                                                                                       |
| **Repository layout**          | *Single* Python file `multimodal_transfer_learning.py` (canvas). Contains: env-driven config, colourful logger, data split (80/10/10), W\&B tracking, training/val/test loops, checkpoint saver (\`../.data/models/{epoch}\_{vit                                                                                                                                                                           | clip}\_topK.pt\`).                                                                                                                                    |
| **Config via `.env`**          | \`VISION\_ENCODER (vit                                                                                                                                                                                                                                                                                                                                                                                     | clip)`, `TOP\_K`, `BATCH\_SIZE`, `EPOCHS`, `LR\_ADAPTER`, `LR\_QWEN`, `SEED`, `OUTPUT\_DIR\_MODELS`, `WANDB\_ENTITY`, `WANDB\_PROJECT`, `HF\_TOKEN\`. |
| **Metrics logged (per epoch)** | BLEU-4, CIDEr-D, Recall\@1/5, Precision\@1/5 – reported to **Weights & Biases** project `mlx8-w4-multimodal-transferlearning`.                                                                                                                                                                                                                                                                             |                                                                                                                                                       |
| **Hardware targets**           | Fits a single 24 GB GPU with BF16 weights, batch = 2, and up to `TOP_K = 3` unfrozen Qwen blocks.                                                                                                                                                                                                                                                                                                          |                                                                                                                                                       |
| **Extra utilities**            | *Tiny* standalone `tiny_decoder.py` shows how to build a 100-line GPT-style decoder if you want to load pretrained weights manually later.                                                                                                                                                                                                                                                                 |                                                                                                                                                       |
| **CLI**                        | `python multimodal_transfer_learning.py --train` → train + eval + save checkpoints.<br>`python multimodal_transfer_learning.py --test` → load last checkpoint, run final test split.                                                                                                                                                                                                                       |                                                                                                                                                       |

#### 🧠 Model Architecture

GPT-4.1
```mermaid
flowchart LR
    A[Input Image] --> B[Vision Encoder<br/>(ViT-B/16 or CLIP ViT-B/32)<br/><i>Frozen</i>]
    B -- CLS/Image Embedding --> C[2-layer MLP Adapter<br/>(Trainable)]
    C -- Visual Tokens --> D[Qwen 3-0.6B<br/>Language Model]
    E[Input Caption<br/>(Tokenized)] --> F[Qwen Embedding Layer]
    F -- Text Tokens --> D
    D -- Output --> G[Caption Prediction<br/>or VQA Answer]
    subgraph Qwen 3-0.6B
        D
    end
    style B fill:#e0e0e0,stroke:#333,stroke-width:2px
    style C fill:#d1e7dd,stroke:#333,stroke-width:2px
    style D fill:#f8d7da,stroke:#333,stroke-width:2px
```

Another version (GPT-o3)
```mermaid
flowchart LR
    %% =========  INPUT  ========= %%
    IMG[User-supplied Image] --> |224×224 RGB| PREP[Pre-processing<br>Resize · Center-crop · Norm];

    %% =========  VISION ENCODER  ========= %%
    PREP --> VE;
    subgraph Frozen Vision Encoder
        direction TB
        VE["Google **ViT-B/16**<br>(CLS 768-d)":::vit]:::box
        altVE["OpenAI **CLIP ViT-B/32**<br>(image_emb 512-d)":::clip]:::box
    end
    classDef box stroke:#555,stroke-width:1px,fill:#fdfdfd;

    %% =========  ADAPTER  ========= %%
    VE -. chooses .-> BRIDGE;
    altVE -. chooses .-> BRIDGE;
    BRIDGE[**MLP Adapter**<br>in 512/768 → out N×4096<br>(*trainable*)]:::train;
    classDef train fill:#e8fff0,stroke:#22aa55,stroke-width:2px;

    %% =========  QWEN DECODER  ========= %%
    BRIDGE -->|concat with prompt| QWEN;
    subgraph Qwen-3-0.6B Decoder
        direction TB
        QWEN["28-layer GPT-style<br>decoder (d = 4096)"]:::frozen
        TOPK{{Top-K layers<br>unfrozen (K = 0–3)}}:::train
    end
    classDef frozen fill:#f0f4ff,stroke:#4466dd;

    QWEN --> TEXT[Text output<br>(caption / VQA answer)];

    %% =========  LEGEND  ========= %%
    class IMG,PREP,VE,altVE,BRIDGE,QWEN,TEXT,TOPK default;
    subgraph Legend
        L1[🔵 Frozen weights]:::frozen
        L2[🟢 Trainable weights]:::train
    end
```

**Diagram key**

* 🔵 **Frozen modules** – Vision encoder (ViT-B/16 *or* CLIP ViT-B/32) and most of Qwen’s 28 decoder blocks.
* 🟢 **Trainable modules** – a two-layer MLP **Adapter** that maps the 768-d (ViT) or 512-d (CLIP) image vector into *N* pseudo-tokens of width 4096; plus the **top-K** Qwen blocks you choose to unfreeze (default K = 3, K = 0 means fully frozen).

**End-to-end flow**

1. **Image → Vision Encoder** (frozen) → single global embedding.
2. **Embedding → Adapter** → *N* visual tokens aligned to Qwen’s hidden size.
3. **Visual tokens + text prompt → Qwen Decoder**; only the top-K layers receive gradients during fine-tuning.
4. Decoder autoregressively produces the caption or answer.
