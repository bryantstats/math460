"""Precompute the real results shown in slides/cnn_fa26.qmd, so the deck renders in seconds.

Run once:  python make_results.py <folder with the four MNIST .gz files> [resnet18 checkpoint]
Writes:    results.npz, next to this script

Part 1. Train a fully connected network and a small CNN on MNIST under different conditions.
Part 2. Read the first-layer filters of a ResNet-18 trained on ImageNet, and trace one photo through it.
"""
import gzip
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.datasets import load_sample_image

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
UPDATES = 5000          # the same number of gradient descent steps in every run
BATCH = 64


def read_idx(path):
    raw = gzip.open(path).read()
    dims = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * dims).reshape(shape)


class FC(nn.Module):
    """Layer sizes 784, 128, 10: the 'Digit images' row in the neural network deck."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """28x28x1 -> conv 16 -> pool -> conv 32 -> pool -> 7x7x32 -> 10 outputs."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def maps(self, x):
        a = torch.relu(self.conv1(x))
        b = torch.relu(self.conv2(self.pool(a)))
        return a, b

    def forward(self, x):
        _, b = self.maps(x)
        return self.fc(self.pool(b).flatten(1))


def train(make, X, y, seed):
    torch.manual_seed(seed)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    perm, at = torch.randperm(n, generator=g), 0
    for _ in range(UPDATES):
        if at + BATCH > n:
            perm, at = torch.randperm(n, generator=g), 0
        b = perm[at:at + min(BATCH, n)]
        at += BATCH
        opt.zero_grad()
        nn.functional.cross_entropy(model(X[b]), y[b]).backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def accuracy(model, X, y):
    hits = sum((model(X[i:i + 2000]).argmax(1) == y[i:i + 2000]).sum().item() for i in range(0, len(X), 2000))
    return hits / len(X)


def shifted(X, dx, dy):
    """Move every image; pixels moved in from outside are black (0), not wrapped around."""
    out = torch.zeros_like(X)
    h, w = X.shape[-2:]
    out[..., max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        X[..., max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def part1(folder, results):
    train_x = torch.tensor(read_idx(f"{folder}/train-images-idx3-ubyte.gz") / 255.0, dtype=torch.float32)[:, None]
    train_y = torch.tensor(read_idx(f"{folder}/train-labels-idx1-ubyte.gz").copy(), dtype=torch.long)
    test_x = torch.tensor(read_idx(f"{folder}/t10k-images-idx3-ubyte.gz") / 255.0, dtype=torch.float32)[:, None]
    test_y = torch.tensor(read_idx(f"{folder}/t10k-labels-idx1-ubyte.gz").copy(), dtype=torch.long)
    assert train_x.shape == (60000, 1, 28, 28) and test_x.shape == (10000, 1, 28, 28)

    order = np.random.default_rng(0).permutation(784)
    shuf = lambda A: A.flatten(1)[:, order].reshape(A.shape)
    subset = torch.tensor(np.random.default_rng(1).permutation(60000))

    distances = [0, 1, 2, 3, 4, 5, 6]
    sizes = [100, 1000, 10000, 60000]
    start = time.time()
    for name, make in [("fc", FC), ("cnn", CNN)]:
        by_size, by_shift, by_shuffle = [], [], []
        for seed in SEEDS:
            full = None
            row = []
            for size in sizes:
                idx = subset[:size]
                model = train(make, train_x[idx], train_y[idx], seed)
                row.append(accuracy(model, test_x, test_y))
                print(f"{name} seed={seed} n={size} acc={row[-1]:.4f} [{time.time() - start:.0f}s]", flush=True)
                if size == 60000:
                    full = model
            by_size.append(row)
            by_shift.append([np.mean([accuracy(full, shifted(test_x, d * a, d * b), test_y)
                                      for a, b in [(1, 0), (-1, 0), (0, 1), (0, -1)]]) for d in distances])
            model = train(make, shuf(train_x), train_y, seed)
            by_shuffle.append(accuracy(model, shuf(test_x), test_y))
            print(f"{name} seed={seed} shuffled acc={by_shuffle[-1]:.4f} [{time.time() - start:.0f}s]", flush=True)
            if name == "cnn" and seed == 0:
                results["cnn_filters_trained"] = full.conv1.weight.detach().numpy().copy()
                torch.manual_seed(0)
                results["cnn_filters_random"] = CNN().conv1.weight.detach().numpy().copy()
                digit = test_x[(test_y == 3).nonzero()[0].item()][None]
                a, b = full.maps(digit)
                results["sample_digit"] = digit[0, 0].numpy()
                results["sample_maps1"] = a[0].detach().numpy()
                results["sample_maps2"] = b[0].detach().numpy()
        results[f"{name}_by_size"] = np.array(by_size)
        results[f"{name}_by_shift"] = np.array(by_shift)
        results[f"{name}_shuffled"] = np.array(by_shuffle)
        results[f"{name}_params"] = sum(p.numel() for p in make().parameters())
    results["sizes"] = np.array(sizes)
    results["distances"] = np.array(distances)
    results["mnist_examples"] = test_x[:40, 0].numpy()
    results["mnist_labels"] = test_y[:40].numpy()
    results["shuffle_order"] = order


# ---------------------------------------------------------------------------
# ResNet-18, written out so that this script does not depend on torchvision
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, a, b, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(a, b, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(b)
        self.conv2 = nn.Conv2d(b, b, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(b)
        self.downsample = None
        if stride != 1 or a != b:
            self.downsample = nn.Sequential(nn.Conv2d(a, b, 1, stride, bias=False), nn.BatchNorm2d(b))

    def forward(self, x):
        skip = x if self.downsample is None else self.downsample(x)
        return torch.relu(self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x))))) + skip)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(Block(64, 64), Block(64, 64))
        self.layer2 = nn.Sequential(Block(64, 128, 2), Block(128, 128))
        self.layer3 = nn.Sequential(Block(128, 256, 2), Block(256, 256))
        self.layer4 = nn.Sequential(Block(256, 512, 2), Block(512, 512))
        self.fc = nn.Linear(512, 1000)

    def stages(self, x):
        out = {}
        x = torch.relu(self.bn1(self.conv1(x)))
        out["conv1"] = x
        x = nn.functional.max_pool2d(x, 3, 2, 1)
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            x = getattr(self, name)(x)
            out[name] = x
        return out


def part2(checkpoint, results):
    model = ResNet18()
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    results["resnet_conv1"] = model.conv1.weight.detach().numpy().copy()
    results["resnet_params"] = sum(p.numel() for p in model.parameters())

    photo = Image.fromarray(load_sample_image("flower.jpg"))
    short = min(photo.size)
    left, top = (photo.width - short) // 2, (photo.height - short) // 2
    photo = photo.crop((left, top, left + short, top + short)).resize((224, 224), Image.BICUBIC)
    rgb = np.asarray(photo, dtype=np.float32) / 255.0
    results["resnet_photo"] = rgb
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    x = torch.tensor(((rgb - mean) / std).astype(np.float32)).permute(2, 0, 1)[None]
    with torch.no_grad():
        stages = model.stages(x)
    for name, value in stages.items():
        value = value[0].numpy()
        strongest = np.argsort(value.mean(axis=(1, 2)))[::-1][:4]   # the 4 most active feature maps
        results[f"stage_{name}_shape"] = np.array(value.shape)
        results[f"stage_{name}_maps"] = value[strongest]


if __name__ == "__main__":
    mnist = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    out = {}
    part2(ckpt, out)
    part1(mnist, out)
    np.savez_compressed(os.path.join(HERE, "results.npz"), **out)
    print("saved", sorted(out))
