"""
Definitive TSS LSTM — Proper Train / Validation / Test Split
-------------------------------------------------------------
Identical architecture to improved_tss_lstm.py with one key change:

Athletes are split three ways (70% / 15% / 15%):
  - Train      : model weights + embeddings learned here
  - Validation : best epoch selected here (early stopping) — never touches test
  - Test       : evaluated ONCE at the end on the best saved model

This addresses the supervisor's feedback that picking the best epoch based on
the test set makes the reported RMSE optimistic (test set was acting as a
validation set). With this split the test RMSE is a clean, unbiased estimate.

Adaptation protocol (val and test athletes):
  ADAPT_RATIO (30%) of rides used to adapt the latent vector (LSTM frozen).
  Remaining 70% used for evaluation.
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
MODEL_SAVE       = "/content/drive/MyDrive/Research_Project/definitive_model.pt"
HISTORY_SAVE     = "/content/drive/MyDrive/Research_Project/definitive_history.csv"

SEED         = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# remaining 0.15 is test
DOWNSAMPLE   = 10
MIN_STEPS    = 60
MAX_STEPS    = 300

LATENT_DIM   = 8
HIDDEN_SIZE  = 64
DROPOUT      = 0.2
BATCH_SIZE   = 64
EPOCHS       = 30
LR           = 0.001
CLIP_GRAD    = 1.0

ADAPT_RATIO  = 0.30
ADAPT_EPOCHS = 5
ADAPT_LR     = 0.01

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
    """Returns dict: athlete_id -> list of (features (T,4), hr (T,))"""
    athlete_rides = {}
    skipped_no_fatigue = 0

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
                        skipped_no_fatigue += 1
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

                        atl_arr = np.full(T, atl,  dtype=np.float32)
                        ctl_arr = np.full(T, ctl,  dtype=np.float32)
                        tsb_arr = np.full(T, tsb,  dtype=np.float32)

                        features = np.stack(
                            [power, atl_arr, ctl_arr, tsb_arr],
                            axis=1,
                        )
                        rides.append((features, hr))

                    except Exception:
                        continue

            if rides:
                athlete_rides[athlete_id] = rides

    print(f"  Skipped {skipped_no_fatigue} rides with no fatigue match")
    return athlete_rides


# ---------------------------------------------------------------------------
# Three-way split
# ---------------------------------------------------------------------------

def split_athletes_three_way(athlete_ids, train_ratio, val_ratio, seed):
    ids = sorted(athlete_ids)
    random.seed(seed)
    random.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    n_val   = int(len(ids) * val_ratio)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_stats(rides):
    all_features = np.concatenate([r[0] for r in rides], axis=0)
    all_hr       = np.concatenate([r[1] for r in rides])
    return {
        "feat_mean": all_features.mean(axis=0).astype(np.float32),
        "feat_std":  all_features.std(axis=0).astype(np.float32),
        "hr_mean":   float(all_hr.mean()),
        "hr_std":    float(all_hr.std()),
    }


def normalize_rides(rides, stats):
    fm, fs = stats["feat_mean"], stats["feat_std"]
    hm, hs = stats["hr_mean"],   stats["hr_std"]
    return [
        (
            (f - fm) / (fs + 1e-8),
            (h - hm) / (hs + 1e-8),
        )
        for f, h in rides
    ]


# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------

class TrainRideDataset(Dataset):
    def __init__(self, rides_with_idx):
        self.rides = rides_with_idx

    def __len__(self):
        return len(self.rides)

    def __getitem__(self, idx):
        features, hr, ath_idx = self.rides[idx]
        return (
            torch.from_numpy(features),
            torch.from_numpy(hr),
            torch.tensor(ath_idx, dtype=torch.long),
        )


def collate_fn(batch):
    features, hrs, ath_idxs = zip(*batch)
    lengths         = torch.tensor([len(f) for f in features], dtype=torch.long)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    hrs_padded      = pad_sequence(hrs,      batch_first=True, padding_value=0.0)
    ath_idxs_t      = torch.stack(ath_idxs)
    return features_padded, hrs_padded, lengths, ath_idxs_t


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, dropout):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.h_proj  = nn.Linear(latent_dim, hidden_size)
        self.c_proj  = nn.Linear(latent_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths, latent):
        h0 = self.h_proj(latent).unsqueeze(0)
        c0 = self.c_proj(latent).unsqueeze(0)

        packed        = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                             enforce_sorted=False)
        out_packed, _ = self.lstm(packed, (h0, c0))
        out, _        = pad_packed_sequence(out_packed, batch_first=True)
        out           = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_mask(pred, lengths):
    T = pred.size(1)
    return torch.arange(T, device=pred.device).unsqueeze(0) < lengths.to(pred.device).unsqueeze(1)


def masked_mse(pred, target, lengths):
    mask = make_mask(pred, lengths)
    return ((pred - target) ** 2 * mask).sum() / mask.sum()


def train_epoch(model, embedding, loader, optimizer):
    model.train()
    embedding.train()
    total_loss, total_n = 0.0, 0

    for features, hr, lengths, ath_idxs in loader:
        features = features.to(DEVICE)
        hr       = hr.to(DEVICE)
        ath_idxs = ath_idxs.to(DEVICE)

        latent = embedding(ath_idxs)

        optimizer.zero_grad()
        pred = model(features, lengths, latent)
        loss = masked_mse(pred, hr, lengths)
        loss.backward()

        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(embedding.parameters()), CLIP_GRAD
        )
        optimizer.step()

        mask        = make_mask(pred, lengths)
        total_loss += ((pred.detach() - hr) ** 2 * mask).sum().item()
        total_n    += mask.sum().item()

    return total_loss / total_n


def adapt_and_evaluate(model, embedding, athlete_rides_dict):
    """
    For each athlete in athlete_rides_dict:
      1. Adapt latent vector on first ADAPT_RATIO rides (LSTM frozen).
      2. Evaluate on remaining rides.
    Returns MSE (normalized scale).
    """
    with torch.no_grad():
        mean_latent = embedding.weight.mean(dim=0).detach()

    total_loss = 0.0
    total_n    = 0

    for athlete_id, rides_norm in athlete_rides_dict.items():
        n_adapt   = max(1, int(len(rides_norm) * ADAPT_RATIO))
        adapt_set = rides_norm[:n_adapt]
        eval_set  = rides_norm[n_adapt:] if len(rides_norm) > n_adapt else rides_norm

        latent    = nn.Parameter(mean_latent.clone().to(DEVICE))
        adapt_opt = torch.optim.Adam([latent], lr=ADAPT_LR)

        model.train()
        for _ in range(ADAPT_EPOCHS):
            for features, hr in adapt_set:
                feat_t = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
                hr_t   = torch.from_numpy(hr).unsqueeze(0).to(DEVICE)
                length = torch.tensor([len(hr)], dtype=torch.long)

                adapt_opt.zero_grad()
                pred = model(feat_t, length, latent.unsqueeze(0))
                loss = masked_mse(pred, hr_t, length)
                loss.backward()
                adapt_opt.step()

        model.eval()
        with torch.no_grad():
            for features, hr in eval_set:
                feat_t = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
                hr_t   = torch.from_numpy(hr).unsqueeze(0).to(DEVICE)
                length = torch.tensor([len(hr)], dtype=torch.long)

                pred = model(feat_t, length, latent.unsqueeze(0))
                mask = make_mask(pred, length)
                total_loss += ((pred - hr_t) ** 2 * mask).sum().item()
                total_n    += mask.sum().item()

    return total_loss / total_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device     : {DEVICE}")
    print(f"Loading fatigue features...")
    fatigue_lookup = load_fatigue_lookup(ATL_CTL_TSB_PATH)
    print(f"  {len(fatigue_lookup)} (athlete, date) entries loaded")

    print("Loading rides...")
    athlete_rides = load_all_rides(DATASET_PATH, fatigue_lookup)
    total_rides   = sum(len(v) for v in athlete_rides.values())
    print(f"  {len(athlete_rides)} athletes, {total_rides} rides loaded\n")

    train_ids, val_ids, test_ids = split_athletes_three_way(
        list(athlete_rides.keys()), TRAIN_RATIO, VAL_RATIO, SEED
    )
    val_ids  = [aid for aid in val_ids  if aid in athlete_rides]
    test_ids = [aid for aid in test_ids if aid in athlete_rides]

    train_rides_flat = [(f, h) for aid in train_ids for (f, h) in athlete_rides[aid]]
    val_rides_dict   = {aid: athlete_rides[aid] for aid in val_ids}
    test_rides_dict  = {aid: athlete_rides[aid] for aid in test_ids}

    print(f"Split      : {len(train_ids)} train ({len(train_rides_flat)} rides) "
          f"| {len(val_ids)} val ({sum(len(v) for v in val_rides_dict.values())} rides) "
          f"| {len(test_ids)} test ({sum(len(v) for v in test_rides_dict.values())} rides)")

    # Stats from train only
    stats = compute_stats(train_rides_flat)
    feat_names = ["power", "atl_pre", "ctl_pre", "tsb_pre"]
    for i, name in enumerate(feat_names):
        print(f"  {name:<8} mean={stats['feat_mean'][i]:.2f}  std={stats['feat_std'][i]:.2f}")
    print(f"  {'hr':<8} mean={stats['hr_mean']:.1f} bpm  std={stats['hr_std']:.1f} bpm\n")

    # Build train dataset with athlete index
    train_athlete_index = {aid: i for i, aid in enumerate(sorted(train_ids))}

    train_with_idx = []
    for aid in sorted(train_ids):
        for f, h in normalize_rides(athlete_rides[aid], stats):
            train_with_idx.append((f, h, train_athlete_index[aid]))

    val_norm_dict  = {aid: normalize_rides(rides, stats) for aid, rides in val_rides_dict.items()}
    test_norm_dict = {aid: normalize_rides(rides, stats) for aid, rides in test_rides_dict.items()}

    train_loader = DataLoader(
        TrainRideDataset(train_with_idx),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
    )

    model     = ImprovedLSTM(INPUT_SIZE, HIDDEN_SIZE, LATENT_DIM, DROPOUT).to(DEVICE)
    embedding = nn.Embedding(len(train_ids), LATENT_DIM).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()) + embedding.weight.numel()
    print(f"Parameters : {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embedding.parameters()), lr=LR
    )
    print(f"Training for {EPOCHS} epochs...\n")

    history      = []
    best_val     = float("inf")
    best_epoch   = -1

    for epoch in range(1, EPOCHS + 1):
        train_mse = train_epoch(model, embedding, train_loader, optimizer)
        val_mse   = adapt_and_evaluate(model, embedding, val_norm_dict)

        train_rmse_bpm = (train_mse ** 0.5) * stats["hr_std"]
        val_rmse_bpm   = (val_mse   ** 0.5) * stats["hr_std"]

        history.append({
            "epoch":          epoch,
            "train_mse":      round(train_mse,     6),
            "val_mse":        round(val_mse,        6),
            "train_rmse_bpm": round(train_rmse_bpm, 3),
            "val_rmse_bpm":   round(val_rmse_bpm,   3),
        })

        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"Train {train_mse:.4f} ({train_rmse_bpm:.2f} bpm)  "
              f"Val {val_mse:.4f} ({val_rmse_bpm:.2f} bpm)")

        if val_mse < best_val:
            best_val   = val_mse
            best_epoch = epoch
            torch.save({
                "model_state":         model.state_dict(),
                "embedding_state":     embedding.state_dict(),
                "stats":               stats,
                "train_athlete_index": train_athlete_index,
            }, MODEL_SAVE)

    # Load best model and evaluate on test set — only once
    print(f"\nBest val RMSE : {(best_val ** 0.5) * stats['hr_std']:.2f} bpm (epoch {best_epoch})")
    print("Loading best model and evaluating on test set...")

    checkpoint = torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    embedding.load_state_dict(checkpoint["embedding_state"])

    test_mse      = adapt_and_evaluate(model, embedding, test_norm_dict)
    test_rmse_bpm = (test_mse ** 0.5) * stats["hr_std"]

    print(f"Final test RMSE : {test_rmse_bpm:.2f} bpm  (evaluated once on held-out test set)")
    print(f"Model saved     : {MODEL_SAVE}")

    with open(HISTORY_SAVE, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"History saved   : {HISTORY_SAVE}")


if __name__ == "__main__":
    main()
