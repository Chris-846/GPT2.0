import torch
import torch.nn as nn

class TinySequenceLM(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)            # (B, T, C)
        pos = self.position_embedding(pos)[None] # (1, T, C)
        h = tok + pos
        logits = self.lm_head(h)                 # (B, T, V)
        return logits

model = TinySequenceLM(vocab_size, block_size)
print("logits.shape:", logits.shape)


def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)

print("initial loss:", sequence_cross_entropy(logits, yb).item())
