import torch
import torch.nn.functional as F
from dataset import vocab_size, block_size
from multi import TinyGPT

chars = sorted(list(set("abcdefghijklmnopqrstuvwxyz ")))
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyGPT(vocab_size, block_size)

def sample_sequence_model(model, block_size, stoi, itos, device, start_text="ROMEO:", max_new_tokens=300):
    model.eval()
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)
    out = list(start_text)
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)
    return "".join(out)

print(sample_sequence_model(model, block_size, stoi, itos, device, start_text="ROMEO:", max_new_tokens=400))
