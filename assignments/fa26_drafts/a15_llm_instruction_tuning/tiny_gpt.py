"""The tiny GPT from Assignment 14, packaged for reuse in Assignments 15 and 16.

The model classes are the same ones you build in Assignment 14.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Tokenizer: one token per character. The vocabulary is fixed (a newline plus
# the 95 printable ASCII characters), so every model in Assignments 14-16
# uses the same token IDs. Characters outside the vocabulary are dropped.
# ---------------------------------------------------------------------------
chars = ["\n"] + [chr(i) for i in range(32, 127)]
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def clean(text):
    return "".join(c for c in text if c in stoi)


def encode(text):
    return [stoi[c] for c in text if c in stoi]


def decode(ids):
    return "".join(itos[i] for i in ids)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, embed_size, head_size, context_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        return F.softmax(scores, dim=-1) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, context_size):
        super().__init__()
        head_size = embed_size // num_heads
        self.heads = nn.ModuleList([Head(embed_size, head_size, context_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        return self.proj(torch.cat([head(x) for head in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_size, 4 * embed_size), nn.ReLU(), nn.Linear(4 * embed_size, embed_size))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_size, num_heads, context_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, num_heads, context_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.feed_forward(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size=vocab_size, embed_size=96, num_heads=4, num_layers=4, context_size=128):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads,
                           num_layers=num_layers, context_size=context_size)
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(context_size, embed_size)
        self.blocks = nn.Sequential(*[Block(embed_size, num_heads, context_size) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(embed_size)
        self.output = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        logits = self.output(self.ln_final(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


# ---------------------------------------------------------------------------
# Training and generation
# ---------------------------------------------------------------------------
def get_batch(data, batch_size, context_size):
    starts = torch.randint(len(data) - context_size - 1, (batch_size,))
    x = torch.stack([data[i:i + context_size] for i in starts])
    y = torch.stack([data[i + 1:i + context_size + 1] for i in starts])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, batch_size=32, eval_batches=50):
    model.eval()
    losses = [model(*get_batch(data, batch_size, model.context_size))[1].item() for _ in range(eval_batches)]
    model.train()
    return sum(losses) / len(losses)


def train(model, train_data, val_data, steps, learning_rate=1e-3, batch_size=32, report_every=500):
    """Next-token training on one long tensor of token IDs. Returns [(step, train loss, val loss), ...]."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for step in range(steps):
        if step % report_every == 0 or step == steps - 1:
            tr, va = estimate_loss(model, train_data), estimate_loss(model, val_data)
            history.append((step, tr, va))
            print(f"step {step:5d}:  train loss {tr:.3f}   val loss {va:.3f}")
        _, loss = model(*get_batch(train_data, batch_size, model.context_size))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history


@torch.no_grad()
def generate(model, prompt, max_new_chars=200, temperature=1.0, stop="\n\n"):
    """Continue `prompt` one character at a time. Returns only the new text, cut at `stop`."""
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long)
    new = ""
    for _ in range(max_new_chars):
        logits, _ = model(idx[:, -model.context_size:])
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        new += itos[next_id.item()]
        if stop and new.endswith(stop):
            new = new[:-len(stop)]
            break
    model.train()
    return new


def save_model(model, path):
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = TinyGPT(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model
