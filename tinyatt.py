import torch.nn as nn

class TinyAttentionLM(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.attn = SingleHeadSelfAttention(emb_dim, block_size)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)
        pos = self.position_embedding(pos)[None]
        h = tok + pos
        h = self.attn(h)
        logits = self.lm_head(h)
        return logits

model = TinyAttentionLM(vocab_size, block_size)
logits = model(xb)
print("logits.shape:", logits.shape)
