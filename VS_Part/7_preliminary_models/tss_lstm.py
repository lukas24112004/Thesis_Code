"""
TSS LSTM
--------
Same architecture as baseline_lstm.py but with TSS-based ATL, CTL, TSB added as inputs.
Input at each timestep: [power, atl_pre, ctl_pre, tsb_pre] (4 features).

Training load features come from tss_atl_ctl_tsb.csv, joined to each ride by
athlete_id + date. CSV filenames are in local time; JSON dates in atl_ctl_tsb
are UTC. For the rare timezone edge case (±1 day mismatch), adjacent dates
are tried as a fallback.

All settings are identical to baseline so results are directly comparable.
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
# Config  — identical to baseline
# ---------------------------------------------------------------------------

DATASET_PATH    = r"C:\Users\Gebruiker\Desktop\The_Project\Dataset_Reduced.zip"
ATL_CTL_TSB_PATH = r"C:\Users\Gebruiker\Desktop\The_Project\tss_atl_ctl_tsb.csv"
MODEL_SAVE      = r"C:\Users\Gebruiker\Desktop\The_Project\VS_Part\7_model\tss_lstm.pt"
HISTORY_SAVE    = r"C:\Users\Gebruiker\Desktop\The_Project\VS_Part\7_model\tss_history.csv"

SEED         = 42
TRAIN_RATIO  = 0.70
DOWNSAMPLE   = 10
MIN_STEPS    = 60
MAX_STEPS    = 300

HIDDEN_SIZE  = 32
DROPOUT      = 0.2
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 0.001
CLIP_GRAD    = 1.0

INPUT_SIZE   = 4   # power + atl_pre + ctl_pre + tsb_pre

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Load fatigue lookup
# ---------------------------------------------------------------------------

def load_fatigue_lookup(path):
    """
    Returns dict: (athlete_id, date_str) -> list of (atl_pre, ctl_pre, tsb_pre)
    ordered by appearance in the CSV (which matches JSON ride order).
    """
    df = pd.read_csv(path, dtype={"date": str})
    lookup = defaultdict(list)
    for _, row in df.iterrows():
        key = (row["athlete_id"], row["date"])
        lookup[key].append((
            float(row["atl_pre"]),
            float(row["ctl_pre"]),
            float(row["tsb_pre"]),
        ))
    return lookup


def get_fatigue(lookup, athlete_id, csv_name, date_counter):
    """
    Parse date from CSV filename (local time), look up fatigue features.
    Tries exact date first, then ±1 day for timezone edge cases.
    Returns (atl, ctl, tsb) or None if no match.
    """
    try:
        date_str = csv_name[:10].replace("_", "-")   # "2018_10_27_..." -> "2018-10-27"
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
    """Returns dict: athlete_id -> list of (feature_array (T,4), hr_array (T,))"""
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
            date_counter = {}   # per-athlete counter for multiple rides same day

            with zipfile.ZipFile(io.BytesIO(outer.read(az_name))) as inner:
                csv_files = sorted(f for f in inner.namelist() if f.endswith(".csv"))

                for csv_name in csv_files:
                    # Strip path prefix if present
                    csv_basename = csv_name.split("/")[-1].split("\\")[-1]

                    fatigue = get_fatigue(fatigue_lookup, athlete_id,
                                         csv_basename, date_counter)
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

                        # Fatigue features repeated for every timestep
                        atl_arr = np.full(T, atl, dtype=np.float32)
                        ctl_arr = np.full(T, ctl, dtype=np.float32)
                        tsb_arr = np.full(T, tsb, dtype=np.float32)

                        features = np.stack([power, atl_arr, ctl_arr, tsb_arr], axis=1)  # (T, 4)
                        rides.append((features, hr))

                    except Exception:
                        continue

            if rides:
                athlete_rides[athlete_id] = rides

    print(f"  Skipped {skipped_no_fatigue} rides with no fatigue match")
    return athlete_rides


# ---------------------------------------------------------------------------
# Splitting and normalization
# ---------------------------------------------------------------------------

def split_athletes(athlete_ids, train_ratio, seed):
    ids = sorted(athlete_ids)
    random.seed(seed)
    random.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    return ids[:n_train], ids[n_train:]


def compute_stats(rides):
    """Compute mean and std for each of the 4 input features and for HR."""
    all_features = np.concatenate([r[0] for r in rides], axis=0)  # (N, 4)
    all_hr       = np.concatenate([r[1] for r in rides])
    means = all_features.mean(axis=0)  # (4,)
    stds  = all_features.std(axis=0)   # (4,)
    return {
        "feat_mean": means,
        "feat_std":  stds,
        "hr_mean":   float(all_hr.mean()),
        "hr_std":    float(all_hr.std()),
    }


def normalize(rides, stats):
    norm_feats = (stats["feat_mean"], stats["feat_std"])
    return [
        (
            (f - norm_feats[0]) / (norm_feats[1] + 1e-8),
            (h - stats["hr_mean"]) / (stats["hr_std"] + 1e-8),
        )
        for f, h in rides
    ]


# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------

class RideDataset(Dataset):
    def __init__(self, rides):
        self.rides = rides

    def __len__(self):
        return len(self.rides)

    def __getitem__(self, idx):
        features, hr = self.rides[idx]
        return (
            torch.from_numpy(features),   # (T, 4)
            torch.from_numpy(hr),         # (T,)
        )


def collate_fn(batch):
    features, hrs = zip(*batch)
    lengths          = torch.tensor([len(f) for f in features], dtype=torch.long)
    features_padded  = pad_sequence(features, batch_first=True, padding_value=0.0)
    hrs_padded       = pad_sequence(hrs,      batch_first=True, padding_value=0.0)
    return features_padded, hrs_padded, lengths


# ---------------------------------------------------------------------------
# Model  — identical to baseline except input_size=4
# ---------------------------------------------------------------------------

class FatigueLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, dropout=0.2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed        = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                             enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
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


def evaluate(model, loader):
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for features, hr, lengths in loader:
            features, hr = features.to(DEVICE), hr.to(DEVICE)
            pred         = model(features, lengths)
            mask         = make_mask(pred, lengths)
            total_loss  += ((pred - hr) ** 2 * mask).sum().item()
            total_n     += mask.sum().item()
    return total_loss / total_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device : {DEVICE}")
    print(f"Loading fatigue features from {ATL_CTL_TSB_PATH}...")
    fatigue_lookup = load_fatigue_lookup(ATL_CTL_TSB_PATH)
    print(f"  {len(fatigue_lookup)} (athlete, date) entries loaded")

    print(f"Loading rides...")
    athlete_rides = load_all_rides(DATASET_PATH, fatigue_lookup)
    total_rides   = sum(len(v) for v in athlete_rides.values())
    print(f"  {len(athlete_rides)} athletes, {total_rides} rides loaded\n")

    train_ids, test_ids = split_athletes(list(athlete_rides.keys()), TRAIN_RATIO, SEED)
    train_rides = [r for aid in train_ids for r in athlete_rides[aid]]
    test_rides  = [r for aid in test_ids  for r in athlete_rides[aid]]
    print(f"Split   : {len(train_ids)} train athletes ({len(train_rides)} rides) "
          f"| {len(test_ids)} test athletes ({len(test_rides)} rides)")

    stats = compute_stats(train_rides)
    feat_names = ["power", "atl_pre", "ctl_pre", "tsb_pre"]
    for i, name in enumerate(feat_names):
        print(f"  {name:<10} mean={stats['feat_mean'][i]:.2f}  std={stats['feat_std'][i]:.2f}")
    print(f"  {'hr':<10} mean={stats['hr_mean']:.1f} bpm  std={stats['hr_std']:.1f} bpm\n")

    train_norm = normalize(train_rides, stats)
    test_norm  = normalize(test_rides,  stats)

    train_loader = DataLoader(RideDataset(train_norm), batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(RideDataset(test_norm),  batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    model     = FatigueLSTM(INPUT_SIZE, HIDDEN_SIZE, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Model   : {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training for {EPOCHS} epochs...\n")

    history   = []
    best_test = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_n = 0.0, 0

        for features, hr, lengths in train_loader:
            features, hr = features.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad()
            pred = model(features, lengths)
            loss = masked_mse(pred, hr, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            mask        = make_mask(pred, lengths)
            train_loss += ((pred.detach() - hr) ** 2 * mask).sum().item()
            train_n    += mask.sum().item()

        train_mse      = train_loss / train_n
        test_mse       = evaluate(model, test_loader)
        train_rmse_bpm = (train_mse ** 0.5) * stats["hr_std"]
        test_rmse_bpm  = (test_mse  ** 0.5) * stats["hr_std"]

        history.append({
            "epoch":          epoch,
            "train_mse":      round(train_mse,       6),
            "test_mse":       round(test_mse,         6),
            "train_rmse_bpm": round(train_rmse_bpm,  3),
            "test_rmse_bpm":  round(test_rmse_bpm,   3),
        })

        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"Train {train_mse:.4f} ({train_rmse_bpm:.2f} bpm)  "
              f"Test {test_mse:.4f} ({test_rmse_bpm:.2f} bpm)")

        if test_mse < best_test:
            best_test = test_mse
            torch.save(model.state_dict(), MODEL_SAVE)

    with open(HISTORY_SAVE, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    best_rmse_bpm = (best_test ** 0.5) * stats["hr_std"]
    print(f"\nBest test MSE  : {best_test:.4f}  ({best_rmse_bpm:.2f} bpm)")
    print(f"Model saved    : {MODEL_SAVE}")
    print(f"History saved  : {HISTORY_SAVE}")


if __name__ == "__main__":
    main()
