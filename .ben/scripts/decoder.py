""" Hand coding a decoder. Compared to the att is all you need figure, this uses only the ddocder block.
Additionally, in this dcoder block no cross-attention is used as embeddings from the encoder are not passed in 
at this later stage, they are passed in only once at the start. So instead all we need is a masked multi-head attention 
and the norm, projections and feed forward layers.
"""


import math
import torch
from torch import Tensor, nn

class DecoderModel(nn.Module):
    """
    Transformer decoder-only model for image captioning.

    Prepends a projected image token to the text embeddings,
    adds positional embeddings, applies N blocks, then
    projects to vocabulary logits.
    """

    def __init__(
        self,
        image_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            image_dim: Dimensionality of image feature vector.
            hidden_dim: Decoder hidden size (must match text embedding size).
            num_heads: Number of attention heads.
            num_layers: Number of DecoderBlock layers.
            vocab_size: Size of output vocabulary.
            max_seq_len: Maximum number of text tokens (excluding image).
            dropout: Dropout probability.
        """
        super().__init__()
        # project image → hidden_dim
        self.img_proj = nn.Linear(image_dim, hidden_dim)

        # learned positional embeddings for [IMG] + tokens
        self.pos_emb = nn.Parameter(torch.zeros(max_seq_len + 1, hidden_dim))

        # stack of decoder blocks
        ff_dim = 4 * hidden_dim # tp scale up the feed-forward inner size
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # final projection: hidden_dim → vocab_size
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        img_embed: Tensor,
        token_embeds: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            img_embed: (batch, image_dim) pre-computed image vectors
            token_embeds: (batch, seq_len, hidden_dim) text embeddings
            attention_mask: (batch, seq_len) bool mask for text tokens
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T, _ = token_embeds.size() # Here B is batch size, T is sequence length, _ is hidden dimension

        # 1) project the img_embed to the txt_embed size and prepend this project img_embed
        # First, add an extra dim to go from (B, hidden_dim) to (B, 1, hidden_dim)
        img_token = self.img_proj(img_embed).unsqueeze(1)  # Now (B,1,hidden_dim)
        # Next, concatenate the image token with the text embeddings
        x = torch.cat([img_token, token_embeds], dim=1)    # Now, X is (B, T+1, hidden_dim), where T is the og seq len

        # 2) add positional embeddings
        seq_len = T + 1 # +1 as we prepended the image token
        x = x + self.pos_emb[:seq_len].unsqueeze(0)  # adds an extra dim to go from (B, hidden_dim) to (B, 1, hidden_dim)

        # 3) Build mask for attention. The att mask is passed in from collate to the dl to here. 
        # First, build a tensor that is the img mask and make it all True, so it is always 1 (never hidden)
        img_mask = torch.ones((B, 1), dtype=torch.bool, device=attention_mask.device)
        # Then, concatenate it with the attention mask for text tokens
        combined_mask = torch.cat([img_mask, attention_mask.bool()], dim=1)

        # 4) pass through all decoder blocks
        for layer in self.layers:
            x = layer(x, combined_mask)

        # 5) project 
        logits = self.output_proj(x)       # (B, T+1, vocab_size)

        # 6) drop the [IMG] token logit head
        return logits[:, 1:, :]            # (B, T, vocab_size)
    
    
class DecoderBlock(nn.Module):
    """
    Single transformer decoder block (masked self-attention + FFN).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            hidden_dim: Model dimension.
            num_heads: Number of attention heads.
            ff_dim: Inner size of feed-forward.
            dropout: Dropout probability.
        """
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        # masked self-attention
        attn_out = self.self_attn(x, attention_mask)
        # residual connection + layer norm
        x = self.norm1(x + attn_out)

        # feed-forward, scales up then scales down
        ffn_out = self.ffn(x)
        # residual connection + layer norm
        return self.norm2(x + ffn_out)
    

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head masked (causal) self-attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        """
        Args:
            hidden_dim: Feature size of the input (and output).
            num_heads: Number of parallel attention heads.
                       Must evenly divide hidden_dim.
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Three separate linear maps for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # After concatenating heads, project back to hidden_dim
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, hidden_dim)
            attention_mask: Bool Tensor of shape (batch, seq_len)
                            True for real tokens, False for padding.
        Returns:
            Tensor of shape (batch, seq_len, hidden_dim)
        """
        B, T, _ = x.size()

        # 1) project to Q, K, V
        Q = self.q_proj(x)  # (B, T, D)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2) reshape into heads: (B, heads, T, head_dim)
        def _split_heads(y: Tensor) -> Tensor:
            # y arrives with shape (B, T, hidden_dim), view unpacks to (B, T, num_heads, head_dim)
            # permute re-arranges from (B, T, heads, head_dim) to (B, heads, T, head_dim)
            return y.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 

        # Make individual Q, K and V's for each head 
        Qh = _split_heads(Q) # (B, T, D) → (B, num_heads, T, head_dim)
        Kh = _split_heads(K)
        Vh = _split_heads(V)

        # 3) scaled dot-product between Q and K
        # scale outputs to stop gradient explosion by dividing by sqrt(head_dim)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)

        # Build causal mask matrix (no attending to future positions)
        causal = torch.tril(torch.ones((T, T), device=x.device)).bool()
        # pad mask: (B, 1, 1, T)
        pad = attention_mask.view(B, 1, 1, T).bool()
        # combine causal and pad masks to get full mask
        full_mask = causal.view(1, 1, T, T) & pad

        scores = scores.masked_fill(~full_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights) 

        # 4) weighted sum
        context = torch.matmul(weights, Vh)  # (B, heads, T, head_dim)

        # 5) Unzip the results from each head and concatenate them
        context = (
            context.permute(0, 2, 1, 3) # swap to go from(batch, heads, tokens, per_head_dim) → (batch, tokens, heads, per_head_dim)
                   .contiguous() # some memeory magic thing
                   .view(B, T, self.num_heads * self.head_dim) # flattens to (B, T, hidden_dim)
        )

        # 6) final linear
        return self.out_proj(context)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            hidden_dim: Input and output size.
            ff_dim: Inner hidden size.
            dropout: Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)



if __name__ == "__main__":
    # smoke test
    B, L, D_img, D_txt, V = 2, 5, 768, 1024, 30522
    dec = DecoderModel(D_img, D_txt, 8, 6, V, L)
    fake_img = torch.randn(B, D_img)
    fake_txt = torch.randn(B, L, D_txt)
    fake_mask = torch.ones(B, L, dtype=torch.bool)
    out = dec(fake_img, fake_txt, fake_mask)
    print("output shape:", out.shape)  # → (2,5,30522)



# Decoder class
# class Decoder(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
    # self.text embeddings # somehow need to pass this to the class
    # self.image embeddings # somehow need to pass this to the class
    # self.img_proj_layer = # for projecting image tensor to text tensor shape 
    # self.proj_out_layer = (in, dim_out = set to vocab size) # for projecting final output to text tensor shape
    # self.pos_embed = # positional embedding
    # self.norm = # layer norm
    # self.mmmha = # masked multi-head attention
    

    # def forward ():
        # image_embeds = self.img_proj_layer(image_embeds)  # project to test emebedding shape
        # x = concat(image_embeds, text embeds) # append img_embed to pos 0.
        # x = x + pos embed
        # for block in num_blocks:
        #   x = masked_mulithead_att(x)
        # x = self.proj_out_layer(x) # project to vocab size
        # logits = softmax(x) # although CEL might do this automatically
        # return logits # (1 logit vector per seq len each with 1 logit per element full vocab)
    
# class MaskedMultiHeadAttention(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
    # init
    # self.num_heads = # set num heads
    # self.linear_proj = # for projecting down to original shape
    # self.norm = # normalisation layer
    # TODO: figure mask out?

    # def forward(x):
       # x_array = []
       # for head in num_heads:
          # x = self_att(x)
          # x_array.append(x)
       # x = x_array.proj (project down back to original shape)
       # x = norm
       # return x

# class SelfAttention(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
    # init
    # self.q_w
    # self.k_w
    # self.v_w
    # self.H_W

   # def forward(x):
     # somehow treat each token one by one and do:
     # Q = x * self.q_w
     # K = x * self.k_w
     # V = x * self.v_w
     # A = Q @ K
     # H = A * V
     # H = H * self.H_W
     # H = softmax(H/sqrt(len(k))) #check this
     # return H  # this is the attention output for one token, need to do this for all tokens in the sequence

