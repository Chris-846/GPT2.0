import torch
from dataset import vocab_size, block_size
from tiny import sequence_cross_entropy
from tiny import TinySequenceLM
from torch.utils.data import Dataset, DataLoader
from multi import MyDataset

def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()
    total_loss, total_count = 0.0, 0
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return total_loss / total_count

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinySequenceLM(vocab_size, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(100):
    train_loss = train_one_epoch(model, loader, optimizer, device, max_steps=300)
    print(f"epoch {epoch:2d} | train loss {train_loss:.4f}")
