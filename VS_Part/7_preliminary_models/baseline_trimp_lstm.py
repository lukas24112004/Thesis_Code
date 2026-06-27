"""
Baseline LSTM — TRIMP Dataset
------------------------------
Identical to baseline_lstm.py but trained on TRIMP_Bounds_Filtered_Dataset.zip
(118 athletes) instead of Dataset_Reduced.zip (94 athletes).

Purpose: establish a fair baseline for the TRIMP population so that the
TRIMP fatigue model result (16.26 bpm) can be properly interpreted.
All hyperparameters are identical to baseline_lstm.py.

Input  : power (1 feature, downsampled every 10 seconds)
Output : heart rate
Reads  : TRIMP_Bounds_Filtered_Dataset.zip
Writes : baseline_trimp_lstm.pt  +  baseline_trimp_history.csv
"""

import csv as csv_module
import io
import random
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_PATH = r"C:\Users\Gebruiker\Desktop\The_Project\TRIMP_Bounds_Filtered_Dataset.zip"
MODEL_SAVE   = r"C:\Users\Gebruiker\Desktop\The_Project\VS_Part\7_model\baseline_trimp_lstm.pt"
HISTORY_SAVE = r"C:\Users\Gebruiker\Desktop\The_Project\VS_Part\7_model\baseline_trimp_history.csv"

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_rides(dataset_path):
    athlete_rides = {}

    with zipfile.ZipFile(dataset_path, "r") as outer:
        athlete_zips = sorted(n for n in outer.namelist() if n.endswith(".zip"))
        n = len(athlete_zips)

        for i, az_name in enumerate(athlete_zips, 1):
            if i % 20 == 0:
                print(f"  Loading athlete {i}/{n}...")

            athlete_id = az_name.replace(".zip", "")
            rides = []

            with zipfile.ZipFile(io.BytesIO(outer.read(az_name))) as inner:
                csv_files = [f for f in inner.namelist() if f.endswith(".csv")]

                for csv_name in csv_files:
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

                        rides.append((
                            df["power"].values.astype(np.float32),
                            df["hr"].values.astype(np.float32),
                        ))
                    except Exception:
                        continue

            if rides:
                athlete_rides[athlete_id] = rides

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
    all_power = np.concatenate([r[0] for r in rides])
    all_hr    = np.concatenate([r[1] for r in rides])
    return {
        "power_mean": float(all_power.mean()),
        "power_std":  float(all_power.std()),
        "hr_mean":    float(all_hr.mean()),
        "hr_std":     float(all_hr.std()),
    }


def normalize(rides, stats):
    return [
        (
            (p - stats["power_mean"]) / (stats["power_std"] + 1e-8),
            (h - stats["hr_mean"])    / (stats["hr_std"]    + 1e-8),
        )
        for p, h in rides
    ]


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

class RideDataset(Dataset):
    def __init__(self, rides):
        self.rides = rides

    def __len__(self):
        return len(self.rides)

    def __getitem__(self, idx):
        power, hr = self.rides[idx]
        return torch.from_numpy(power).unsqueeze(-1), torch.from_numpy(hr)


def collate_fn(batch):
    powers, hrs   = zip(*batch)
    lengths       = torch.tensor([len(p) for p in powers], dtype=torch.long)
    powers_padded = pad_sequence(powers, batch_first=True, padding_value=0.0)
    hrs_padded    = pad_sequence(hrs,    batch_first=True, padding_value=0.0)
    return powers_padded, hrs_padded, lengths


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BaselineLSTM(nn.Module):
    def __init__(self, hidden_size=32, dropout=0.2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=1, hidden_size=hidden_size,
                               num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed        = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                             enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _        = pad_packed_sequence(out_packed, batch_first=True)
        return self.fc(self.dropout(out)).squeeze(-1)


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
        for power, hr, lengths in loader:
            power, hr = power.to(DEVICE), hr.to(DEVICE)
            pred      = model(power, lengths)
            mask      = make_mask(pred, lengths)
            total_loss += ((pred - hr) ** 2 * mask).sum().item()
            total_n    += mask.sum().item()
    return total_loss / total_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device : {DEVICE}")
    print(f"Loading rides...")
    athlete_rides = load_all_rides(DATASET_PATH)
    total_rides   = sum(len(v) for v in athlete_rides.values())
    print(f"  {len(athlete_rides)} athletes, {total_rides} rides loaded\n")

    train_ids, test_ids = split_athletes(list(athlete_rides.keys()), TRAIN_RATIO, SEED)
    train_rides = [r for aid in train_ids for r in athlete_rides[aid]]
    test_rides  = [r for aid in test_ids  for r in athlete_rides[aid]]
    print(f"Split   : {len(train_ids)} train athletes ({len(train_rides)} rides) "
          f"| {len(test_ids)} test athletes ({len(test_rides)} rides)")

    stats = compute_stats(train_rides)
    print(f"Stats   : power mean={stats['power_mean']:.1f}W  std={stats['power_std']:.1f}W")
    print(f"          hr    mean={stats['hr_mean']:.1f} bpm  std={stats['hr_std']:.1f} bpm\n")

    train_norm = normalize(train_rides, stats)
    test_norm  = normalize(test_rides,  stats)

    train_loader = DataLoader(RideDataset(train_norm), batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(RideDataset(test_norm),  batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    model     = BaselineLSTM(HIDDEN_SIZE, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Model   : {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training for {EPOCHS} epochs...\n")

    history   = []
    best_test = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_n = 0.0, 0

        for power, hr, lengths in train_loader:
            power, hr = power.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad()
            pred = model(power, lengths)
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
            "train_mse":      round(train_mse,      6),
            "test_mse":       round(test_mse,        6),
            "train_rmse_bpm": round(train_rmse_bpm, 3),
            "test_rmse_bpm":  round(test_rmse_bpm,  3),
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
    print(f"\nBest test RMSE : {best_rmse_bpm:.2f} bpm")
    print(f"Model saved    : {MODEL_SAVE}")
    print(f"History saved  : {HISTORY_SAVE}")


if __name__ == "__main__":
    main()
