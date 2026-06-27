"""
Simple TSS LSTM — No Latent Vector (Colab version)
---------------------------------------------------
This is a deliberately simpler model than the definitive one.
No athlete latent vector, no personalization — just a plain LSTM
that takes [power, atl, ctl, tsb] and predicts HR.

Purpose: test whether the simpler model learns the correct fatigue
direction. The hypothesis is that the latent vector in the definitive
model absorbed the fitness signal so well that TSB lost its influence.
In this model TSB has to do more work — if the direction is still wrong,
the problem is in the data, not model complexity.

Architecture:
  - Single LSTM layer, hidden=32
  - No athlete embedding
  - Dropout 0.2, linear output layer

Split: 70% train / 15% val / 15% test (same seed as definitive model
so test athletes are identical — allows direct comparison).
"""

import csv as csv_module
import io
import random
import zipfile
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH     = "/content/drive/MyDrive/Research_Project/Dataset_Reduced.zip"
ATL_CTL_TSB_PATH = "/content/drive/MyDrive/Research_Project/tss_atl_ctl_tsb.csv"
MODEL_SAVE       = "/content/drive/MyDrive/Research_Project/simple_model.pt"
HISTORY_SAVE     = "/content/drive/MyDrive/Research_Project/simple_model_history.csv"

SEED         = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
DOWNSAMPLE   = 10
MIN_STEPS    = 60
MAX_STEPS    = 300

HIDDEN_SIZE  = 32
DROPOUT      = 0.2
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 0.001
CLIP_GRAD    = 1.0
INPUT_SIZE   = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Fatigue lookup
# ---------------------------------------------------------------------------

def load_fatigue_lookup(path):
    df = pd.read_csv(path, dtype={"date": str})
    lookup = defaultdict(list)
    for _, row in df.iterrows():
        key = (row["athlete_id"], row["date"])
        lookup[key].append((float(row["atl_pre"]), float(row["ctl_pre"]), float(row["tsb_pre"])))
    return lookup


def get_fatigue(lookup, athlete_id, csv_name, date_counter):
    try:
        date_str = csv_name[:10].replace("_", "-")
        d = date.fromisoformat(date_str)
    except (ValueError, IndexError):
        return None
    for delta in [0, -1, 1, -2, 2]:
        candidate = (d + timedelta(days=delta)).isoformat()
        key = (athlete_id, candidate)
        entries = lookup.get(key)
        if entries:
            idx = date_counter.get(key, 0)
            if idx < len(entries):
                date_counter[key] = idx + 1
                return entries[idx]
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_rides(dataset_path, fatigue_lookup):
    athlete_rides = {}
    with zipfile.ZipFile(dataset_path, "r") as outer:
        athlete_zips = sorted(n for n in outer.namelist() if n.endswith(".zip"))
        n = len(athlete_zips)
        for i, az_name in enumerate(athlete_zips, 1):
            if i % 20 == 0:
                print(f"  Loading athlete {i}/{n}...")
            athlete_id   = az_name.replace(".zip", "")
            rides        = []
            date_counter = {}
            with zipfile.ZipFile(io.BytesIO(outer.read(az_name))) as inner:
                csv_files = sorted(f for f in inner.namelist() if f.endswith(".csv"))
                for csv_name in csv_files:
                    csv_basename = csv_name.split("/")[-1].split("\\")[-1]
                    fatigue = get_fatigue(fatigue_lookup, athlete_id, csv_basename, date_counter)
                    if fatigue is None:
                        continue
                    atl, ctl, tsb = fatigue
                    try:
                        with inner.open(csv_name) as f:
                            df = pd.read_csv(f)
                        if "power" not in df.columns or "hr" not in df.columns:
                            continue
                        df = df[["power", "hr"]].dropna()
                        df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)
                        if len(df) < MIN_STEPS:
                            continue
                        if df["power"].min() < 0 or df["hr"].min() <= 0:
                            continue
                        df = df.iloc[:MAX_STEPS]
                        power = df["power"].values.astype(np.float32)
                        hr    = df["hr"].values.astype(np.float32)
                        T     = len(power)
                        features = np.stack([
                            power,
                            np.full(T, atl, dtype=np.float32),
                            np.full(T, ctl, dtype=np.float32),
                            np.full(T, tsb, dtype=np.float32),
                        ], axis=1)
                        rides.append((features, hr))
                    except Exception:
                        continue
            if rides:
                athlete_rides[athlete_id] = rides
    return athlete_rides


# ---------------------------------------------------------------------------
# Split / normalization
# ---------------------------------------------------------------------------

def split_athletes_three_way(athlete_ids, train_ratio, val_ratio, seed):
    ids = sorted(athlete_ids)
    random.seed(seed)
    random.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    n_val   = int(len(ids) * val_ratio)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def compute_stats(rides):
    all_features = np.concatenate([r[0] for r in rides], axis=0)
    all_hr       = np.concatenate([r[1] for r in rides])
    return {
        "feat_mean": all_features.mean(axis=0).astype(np.float32),
        "feat_std":  all_features.std(axis=0).astype(np.float32),
        "hr_mean":   float(all_hr.mean()),
        "hr_std":    float(all_hr.std()),
    }


def normalize(rides, stats):
    fm, fs = stats["feat_mean"], stats["feat_std"]
    hm, hs = stats["hr_mean"],   stats["hr_std"]
    return [((f - fm) / (fs + 1e-8), (h - hm) / (hs + 1e-8)) for f, h in rides]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RideDataset(Dataset):
    def __init__(self, rides):
        self.rides = rides

    def __len__(self):
        return len(self.rides)

    def __getitem__(self, idx):
        f, h = self.rides[idx]
        return torch.from_numpy(f), torch.from_numpy(h)


def collate_fn(batch):
    features, hrs = zip(*batch)
    lengths         = torch.tensor([len(f) for f in features], dtype=torch.long)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    hrs_padded      = pad_sequence(hrs,      batch_first=True, padding_value=0.0)
    return features_padded, hrs_padded, lengths


# ---------------------------------------------------------------------------
# Model — no latent vector
# ---------------------------------------------------------------------------

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed        = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _        = pad_packed_sequence(out_packed, batch_first=True)
        out           = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_mask(pred, lengths):
    return torch.arange(pred.size(1), device=pred.device).unsqueeze(0) < lengths.to(pred.device).unsqueeze(1)


def masked_rmse(pred, target, lengths, hr_std):
    mask = make_mask(pred, lengths)
    mse  = ((pred - target) ** 2 * mask).sum() / mask.sum()
    return (mse.item() ** 0.5) * hr_std


def evaluate(model, loader, hr_std):
    model.eval()
    total_sq, total_n = 0.0, 0
    with torch.no_grad():
        for features, hr, lengths in loader:
            features, hr = features.to(DEVICE), hr.to(DEVICE)
            pred         = model(features, lengths)
            mask         = make_mask(pred, lengths)
            total_sq    += ((pred - hr) ** 2 * mask).sum().item()
            total_n     += mask.sum().item()
    return (total_sq / total_n) ** 0.5 * hr_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device: {DEVICE}")

    print("Loading fatigue features...")
    fatigue_lookup = load_fatigue_lookup(ATL_CTL_TSB_PATH)

    print("Loading rides...")
    athlete_rides = load_all_rides(DATASET_PATH, fatigue_lookup)
    print(f"  {len(athlete_rides)} athletes loaded\n")

    train_ids, val_ids, test_ids = split_athletes_three_way(
        list(athlete_rides.keys()), TRAIN_RATIO, VAL_RATIO, SEED
    )
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test athletes")

    train_rides = [r for aid in train_ids for r in athlete_rides[aid]]
    val_rides   = [r for aid in val_ids   for r in athlete_rides[aid]]
    test_rides  = [r for aid in test_ids  for r in athlete_rides[aid]]

    stats = compute_stats(train_rides)
    print(f"  HR mean={stats['hr_mean']:.1f} bpm  std={stats['hr_std']:.1f} bpm\n")

    train_norm = normalize(train_rides, stats)
    val_norm   = normalize(val_rides,   stats)
    test_norm  = normalize(test_rides,  stats)

    train_loader = DataLoader(RideDataset(train_norm), batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(RideDataset(val_norm),   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(RideDataset(test_norm),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model     = SimpleLSTM(INPUT_SIZE, HIDDEN_SIZE, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training for {EPOCHS} epochs...\n")

    best_val  = float("inf")
    best_epoch = 0
    history   = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_sq, total_n = 0.0, 0
        for features, hr, lengths in train_loader:
            features, hr = features.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad()
            pred = model(features, lengths)
            mask = make_mask(pred, lengths)
            loss = ((pred - hr) ** 2 * mask).sum() / mask.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            total_sq += ((pred.detach() - hr) ** 2 * mask).sum().item()
            total_n  += mask.sum().item()

        train_rmse = (total_sq / total_n) ** 0.5 * stats["hr_std"]
        val_rmse   = evaluate(model, val_loader,  stats["hr_std"])

        history.append({"epoch": epoch, "train_rmse": round(train_rmse, 3), "val_rmse": round(val_rmse, 3)})
        print(f"Epoch {epoch:2d}/{EPOCHS}  Train {train_rmse:.2f} bpm  Val {val_rmse:.2f} bpm")

        if val_rmse < best_val:
            best_val   = val_rmse
            best_epoch = epoch
            torch.save({"model_state": model.state_dict(), "stats": stats}, MODEL_SAVE)

    print(f"\nBest val RMSE: {best_val:.2f} bpm (epoch {best_epoch})")

    # Load best model and evaluate on test set once
    checkpoint = torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_rmse = evaluate(model, test_loader, stats["hr_std"])
    print(f"Test RMSE:     {test_rmse:.2f} bpm")

    with open(HISTORY_SAVE, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=["epoch", "train_rmse", "val_rmse"])
        writer.writeheader()
        writer.writerows(history)
    print(f"History saved: {HISTORY_SAVE}")


if __name__ == "__main__":
    main()
