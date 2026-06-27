"""
Fatigue Effect on Predicted HR
--------------------------------
Loads the trained definitive model and estimates how TSB affects predicted
heart rate at a fixed effort level, per test athlete.

Method:
  - Reconstruct the same train/test split (same seed) so we know who the test athletes are
  - Re-adapt each test athlete's latent vector on their first 30% of rides
  - For each test athlete, take one real ride from their evaluation set as the power template
  - Replace that ride's ATL/CTL/TSB with swept values (TSB from -40 to +20, CTL fixed at
    the athlete's median CTL, ATL = CTL - TSB)
  - Run the model and record the mean predicted HR over the ride
  - Plot predicted HR vs TSB: one line per athlete + bold mean line

Why a real ride as power template:
  The model was trained on real varying-power sequences. Using a flat constant-power
  sequence would be out-of-distribution. Keeping the real power and only changing the
  fatigue context isolates the pure effect of TSB on predicted HR.

Run on Google Colab after training completes. Saves plot to Drive.
"""

import io
import random
import zipfile
from collections import defaultdict
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ---------------------------------------------------------------------------
# Config  — must match training script exactly
# ---------------------------------------------------------------------------

DATASET_PATH     = "/content/drive/MyDrive/Research_Project/Dataset_Reduced.zip"
ATL_CTL_TSB_PATH = "/content/drive/MyDrive/Research_Project/tss_atl_ctl_tsb.csv"
MODEL_PATH       = "/content/drive/MyDrive/Research_Project/definitive_model.pt"
PLOT_SAVE        = "/content/drive/MyDrive/Research_Project/fatigue_effect.png"
RESULTS_SAVE     = "/content/drive/MyDrive/Research_Project/fatigue_effect.csv"

SEED         = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
DOWNSAMPLE   = 10
MIN_STEPS    = 60
MAX_STEPS    = 300

LATENT_DIM   = 8
HIDDEN_SIZE  = 64
DROPOUT      = 0.2
INPUT_SIZE   = 4

ADAPT_RATIO  = 0.30
ADAPT_EPOCHS = 5
ADAPT_LR     = 0.01

# TSB sweep range
TSB_VALUES = list(range(-40, 25, 5))   # -40, -35, ..., +20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model definition (identical to training script)
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
                            np.full(T, atl,  dtype=np.float32),
                            np.full(T, ctl,  dtype=np.float32),
                            np.full(T, tsb,  dtype=np.float32),
                        ], axis=1)
                        rides.append((features, hr))
                    except Exception:
                        continue
            if rides:
                athlete_rides[athlete_id] = rides
    return athlete_rides


# ---------------------------------------------------------------------------
# Split (must match training script)
# ---------------------------------------------------------------------------

def split_athletes_three_way(athlete_ids, train_ratio, val_ratio, seed):
    ids = sorted(athlete_ids)
    random.seed(seed)
    random.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    n_val   = int(len(ids) * val_ratio)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_rides(rides, stats):
    fm, fs = stats["feat_mean"], stats["feat_std"]
    hm, hs = stats["hr_mean"],   stats["hr_std"]
    return [((f - fm) / (fs + 1e-8), (h - hm) / (hs + 1e-8)) for f, h in rides]


def normalize_features(features, stats):
    return (features - stats["feat_mean"]) / (stats["feat_std"] + 1e-8)


def denormalize_hr(hr_norm, stats):
    return hr_norm * stats["hr_std"] + stats["hr_mean"]


# ---------------------------------------------------------------------------
# Latent adaptation
# ---------------------------------------------------------------------------

def adapt_latent(model, mean_latent, rides_norm):
    """Adapt a latent vector for one athlete on their adaptation rides."""
    latent    = nn.Parameter(mean_latent.clone().to(DEVICE))
    adapt_opt = torch.optim.Adam([latent], lr=ADAPT_LR)

    model.train()
    for _ in range(ADAPT_EPOCHS):
        for features, hr in rides_norm:
            feat_t = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
            hr_t   = torch.from_numpy(hr).unsqueeze(0).to(DEVICE)
            length = torch.tensor([len(hr)], dtype=torch.long)

            adapt_opt.zero_grad()
            pred = model(feat_t, length, latent.unsqueeze(0))
            mask = torch.arange(pred.size(1), device=DEVICE) < length.to(DEVICE)
            loss = ((pred - hr_t) ** 2 * mask).sum() / mask.sum()
            loss.backward()
            adapt_opt.step()

    return latent.detach()


# ---------------------------------------------------------------------------
# TSB sweep for one athlete
# ---------------------------------------------------------------------------

def sweep_tsb(model, latent, power_template, median_ctl, stats):
    """
    Takes one real power sequence, replaces fatigue features with swept TSB values.
    Returns list of mean predicted HR (in bpm) for each TSB value.
    """
    model.eval()
    T = len(power_template)

    predicted_hrs = []

    with torch.no_grad():
        for tsb in TSB_VALUES:
            atl = median_ctl - tsb   # CTL fixed, ATL varies with TSB

            features = np.stack([
                power_template,
                np.full(T, atl,        dtype=np.float32),
                np.full(T, median_ctl, dtype=np.float32),
                np.full(T, tsb,        dtype=np.float32),
            ], axis=1)

            feat_norm = normalize_features(features, stats)
            feat_t    = torch.from_numpy(feat_norm).unsqueeze(0).to(DEVICE)
            length    = torch.tensor([T], dtype=torch.long)

            pred_norm = model(feat_t, length, latent.unsqueeze(0))
            pred_bpm  = denormalize_hr(pred_norm.squeeze(0).cpu().numpy(), stats)

            # Use mean of the second half of the ride as steady-state HR
            steady_state = pred_bpm[T // 2:].mean()
            predicted_hrs.append(float(steady_state))

    return predicted_hrs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device : {DEVICE}")

    # Load model
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    stats      = checkpoint["stats"]

    model = ImprovedLSTM(INPUT_SIZE, HIDDEN_SIZE, LATENT_DIM, DROPOUT).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    embedding = nn.Embedding(len(checkpoint["train_athlete_index"]), LATENT_DIM).to(DEVICE)
    embedding.load_state_dict(checkpoint["embedding_state"])

    with torch.no_grad():
        mean_latent = embedding.weight.mean(dim=0).detach()

    # Load data
    print("Loading fatigue features...")
    fatigue_lookup = load_fatigue_lookup(ATL_CTL_TSB_PATH)

    print("Loading rides...")
    athlete_rides = load_all_rides(DATASET_PATH, fatigue_lookup)

    # Reconstruct same split
    train_ids, val_ids, test_ids = split_athletes_three_way(
        list(athlete_rides.keys()), TRAIN_RATIO, VAL_RATIO, SEED
    )
    test_ids = [aid for aid in test_ids if aid in athlete_rides]
    print(f"  {len(test_ids)} test athletes\n")

    # Load fatigue CSV for median CTL per athlete
    fatigue_df = pd.read_csv(ATL_CTL_TSB_PATH, dtype={"date": str})

    # Run sweep for each test athlete
    all_results = {}   # athlete_id -> list of predicted HR per TSB

    for i, athlete_id in enumerate(test_ids, 1):
        print(f"  Athlete {i}/{len(test_ids)}: {athlete_id}")

        rides      = athlete_rides[athlete_id]
        rides_norm = normalize_rides(rides, stats)

        # Adapt latent
        n_adapt    = max(1, int(len(rides_norm) * ADAPT_RATIO))
        adapt_set  = rides_norm[:n_adapt]
        eval_set   = rides_norm[n_adapt:] if len(rides_norm) > n_adapt else rides_norm

        latent = adapt_latent(model, mean_latent, adapt_set)

        # Median CTL for this athlete
        ath_df     = fatigue_df[fatigue_df["athlete_id"] == athlete_id]
        median_ctl = float(ath_df["ctl_pre"].median()) if len(ath_df) > 0 else 60.0

        # Use median power ride from eval set as power template
        # Pick the ride whose mean power is closest to the athlete's median power
        all_powers    = np.concatenate([f[:, 0] for f, _ in eval_set])
        median_power  = np.median(all_powers)
        best_ride     = min(eval_set, key=lambda r: abs(r[0][:, 0].mean() - median_power))

        # Denormalize power back to watts for the template
        power_norm    = best_ride[0][:, 0]
        power_watts   = power_norm * stats["feat_std"][0] + stats["feat_mean"][0]

        predicted_hrs = sweep_tsb(model, latent, power_watts, median_ctl, stats)
        all_results[athlete_id] = predicted_hrs

        print(f"    Median CTL={median_ctl:.1f}  HR range: "
              f"{min(predicted_hrs):.1f}–{max(predicted_hrs):.1f} bpm  "
              f"Δ={max(predicted_hrs)-min(predicted_hrs):.1f} bpm")

    # Save results CSV
    rows = []
    for athlete_id, hrs in all_results.items():
        for tsb, hr in zip(TSB_VALUES, hrs):
            rows.append({"athlete_id": athlete_id, "tsb": tsb, "predicted_hr_bpm": round(hr, 2)})
    pd.DataFrame(rows).to_csv(RESULTS_SAVE, index=False)
    print(f"\nResults saved: {RESULTS_SAVE}")

    # Plot
    hrs_matrix = np.array(list(all_results.values()))   # (n_athletes, n_tsb)
    mean_hrs   = hrs_matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    for athlete_id, hrs in all_results.items():
        ax.plot(TSB_VALUES, hrs, color="steelblue", alpha=0.3, linewidth=1)

    ax.plot(TSB_VALUES, mean_hrs, color="crimson", linewidth=2.5, label="Mean across athletes")

    ax.set_xlabel("TSB (Training Stress Balance)", fontsize=12)
    ax.set_ylabel("Predicted Heart Rate (bpm)", fontsize=12)
    ax.set_title("Effect of Fatigue State (TSB) on Predicted HR at Fixed Power", fontsize=13)
    ax.legend(fontsize=11)
    ax.invert_xaxis()   # most fatigued (negative TSB) on the left, most rested on the right
    ax.grid(True, alpha=0.3)

    # Annotate direction
    ax.text(0.02, 0.97, "← More fatigued", transform=ax.transAxes,
            fontsize=9, va="top", color="gray")
    ax.text(0.98, 0.97, "More rested →", transform=ax.transAxes,
            fontsize=9, va="top", ha="right", color="gray")

    plt.tight_layout()
    plt.savefig(PLOT_SAVE, dpi=150)
    plt.show()
    print(f"Plot saved: {PLOT_SAVE}")

    # Summary
    print(f"\nMean predicted HR across athletes:")
    for tsb, hr in zip(TSB_VALUES, mean_hrs):
        print(f"  TSB={tsb:+4d}   HR={hr:.1f} bpm")

    total_delta = mean_hrs[0] - mean_hrs[-1]
    print(f"\nTotal HR shift from TSB=-40 to TSB=+20: {total_delta:+.1f} bpm")


if __name__ == "__main__":
    main()
