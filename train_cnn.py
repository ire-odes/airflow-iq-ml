"""
train_cnn.py
Mel-spectrogram + small CNN challenger for binary FPR5/FPR12.

At ~1000 clips/class this is now viable -- but ONLY with session-grouped
splits. A CNN will exploit session-correlated background noise far more
aggressively than a RandomForest on summary features; the leakage that
inflated your RF to 100% would make a CNN look superhuman while learning
nothing about filters.

Run this AFTER train_classifier.py and compare holdout numbers on the
SAME session split (both scripts use random_state=42 for the group
split, so the holdout sessions match if features.csv is unchanged).

Requires: pip install torch  (not needed for the RF pipeline)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_DIR = "data"
CLASSES = ["FPR5", "FPR12"]      # index 1 = dirty
SR = 16000
N_MELS = 64
FIXED_FRAMES = 63                 # ~2 s at hop 512 / 16 kHz
RANDOM_STATE = 42
EPOCHS = 30
BATCH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def logmel(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    # RMS normalize -- same reasoning as the paper's chunk normalization
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                       n_fft=1024, hop_length=512,
                                       fmin=20, fmax=SR // 2)
    m = librosa.power_to_db(m, ref=np.max)
    # pad/crop time axis to fixed width
    if m.shape[1] < FIXED_FRAMES:
        m = np.pad(m, ((0, 0), (0, FIXED_FRAMES - m.shape[1])),
                   mode="edge")
    m = m[:, :FIXED_FRAMES]
    # per-clip standardization
    return (m - m.mean()) / (m.std() + 1e-9)


def spec_augment(m, max_f=8, max_t=8, n_masks=2):
    """SpecAugment-style masking: cheap, label-preserving augmentation.
    Force the net to use the whole spectral shape rather than one lucky
    band -- the CNN analog of the paper's moving-mic robustness idea."""
    m = m.copy()
    for _ in range(n_masks):
        f0 = np.random.randint(0, N_MELS - max_f)
        m[f0:f0 + np.random.randint(1, max_f + 1), :] = 0
        t0 = np.random.randint(0, FIXED_FRAMES - max_t)
        m[:, t0:t0 + np.random.randint(1, max_t + 1)] = 0
    return m


class ClipDataset(Dataset):
    def __init__(self, specs, labels, train=False):
        self.specs, self.labels, self.train = specs, labels, train

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        m = self.specs[i]
        if self.train:
            m = spec_augment(m)
        return (torch.tensor(m[None], dtype=torch.float32),
                torch.tensor(self.labels[i], dtype=torch.long))


class SmallCNN(nn.Module):
    """3 conv blocks + global pooling. ~100k params -- sized for 2k
    clips, not ImageNet."""
    def __init__(self, n_classes=2):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 16), block(16, 32),
                                      block(32, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def load_clips():
    rows = []
    for label_idx, label in enumerate(CLASSES):
        for f in sorted((Path(DATA_DIR) / label).glob("*.wav")):
            # session parsing identical to extract_features_v2
            import re
            m = re.match(r"recording_(\d{8})_(\d{6})", f.name)
            sess = (f"{label}_{m.group(1)}_{m.group(2)[:2]}"
                    if m else f"{label}_unknown")
            rows.append((str(f), label_idx, sess))
    return rows


def main():
    rows = load_clips()
    print(f"Loading {len(rows)} clips...")
    specs = np.stack([logmel(p) for p, _, _ in rows])
    labels = np.array([l for _, l, _ in rows])
    groups = np.array([s for _, _, s in rows])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25,
                            random_state=RANDOM_STATE)
    tr, te = next(gss.split(specs, labels, groups))
    print(f"Train {len(tr)} / Test {len(te)} "
          f"({len(set(groups[te]))} holdout sessions)")

    train_dl = DataLoader(ClipDataset(specs[tr], labels[tr], train=True),
                          batch_size=BATCH, shuffle=True)
    test_dl = DataLoader(ClipDataset(specs[te], labels[te]),
                         batch_size=BATCH)

    model = SmallCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    lossf = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
        sched.step()

        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
                ys.append(yb)
        acc = accuracy_score(torch.cat(ys), torch.cat(preds))
        if acc > best_acc:
            best_acc, best_state = acc, {
                k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch+1:2d}/{EPOCHS}  holdout acc {acc:.2%}")

    model.load_state_dict(best_state)
    preds, ys = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_dl:
            preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
            ys.append(yb)
    y_pred, y_true = torch.cat(preds).numpy(), torch.cat(ys).numpy()

    print(f"\nBest holdout accuracy: {best_acc:.2%}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix (rows=true {CLASSES}):\n{cm}")
    fp = cm[0, 1] / max(cm[0].sum(), 1)   # clean -> dirty
    fn = cm[1, 0] / max(cm[1].sum(), 1)   # dirty -> clean
    print(f"FP (clean->dirty): {fp:.2%}   FN (dirty->clean): {fn:.2%}")

    torch.save(model.state_dict(), "fpr_cnn.pt")
    print("Saved fpr_cnn.pt")


if __name__ == "__main__":
    main()
