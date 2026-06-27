import torch
import torch.nn as nn

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, block_size):
        super().__init__()
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v
        return out
