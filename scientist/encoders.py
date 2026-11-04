"""Small per-modality encoders (plain PyTorch). Each maps raw data -> a d-dim vector.

  MolEncoder      SMILES -> Morgan fingerprint (2048 bits) -> MLP
  SpectrumEncoder 1D curve (256 points: XRD, HPLC, IR, growth curves) -> 1D CNN
  ImageEncoder    32x32 microscopy patch -> small 2D CNN
  TextEncoder     hashed word/bigram bag -> MLP (no tokenizer needed, KISS)
"""
import re
import zlib

import numpy as np
import torch
from torch import nn


def morgan(smiles, n_bits=2048):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles or "")
    if m is None:
        return np.zeros(n_bits, np.float32)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    return gen.GetFingerprintAsNumPy(m).astype(np.float32)


def norm_curve(y, n=256):
    y = np.asarray(y, np.float32)
    if len(y) != n:
        y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y).astype(np.float32)
    return (y - y.mean()) / (y.std() + 1e-6)


def hash_text(text, dim=4096):
    v = np.zeros(dim, np.float32)
    w = re.findall(r"[a-z0-9]+", (text or "").lower())
    for tok in w + [a + "_" + b for a, b in zip(w, w[1:])]:
        v[zlib.crc32(tok.encode()) % dim] += 1.0
    return np.log1p(v)


class MolEncoder(nn.Module):
    def __init__(self, d=256, n_bits=2048):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_bits, 512), nn.ReLU(), nn.Linear(512, d))

    def forward(self, x):
        return self.net(x)


class SpectrumEncoder(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.ReLU(), nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
                                 nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(1024, d))

    def forward(self, x):  # x: (B, 256)
        return self.net(x[:, None, :])


class ImageEncoder(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 8 * 8, d))

    def forward(self, x):  # (B, 32, 32)
        return self.net(x[:, None])


class TextEncoder(nn.Module):
    def __init__(self, d=256, dim=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, d))

    def forward(self, x):
        return self.net(x)
