"""
Optimal Fatigue State for Maximum Performance
----------------------------------------------
For each test athlete: sweep TSB from -40 to +20 (keeping CTL fixed at their
median) and run the power ramp at each TSB value. Record the estimated max
sustainable power at each fatigue level.

The TSB where estimated max power is highest = the model's "optimal fatigue state"
for that athlete.

Plots:
  - Line plot: estimated max power vs TSB for each athlete (gray) + mean (bold)
  - Bar chart: optimal TSB per athlete

Run on Google Colab. Outputs saved to Drive.
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
# Paths
# ---------------------------------------------------------------------------

DATASET_PATH     = "/content/drive/MyDrive/Research_Project/Dataset_Reduced.zip"
ATL_CTL_TSB_PATH = "/content/drive/MyDrive/Research_Project/tss_atl_ctl_tsb.csv"
MODEL_PATH       = "/content/drive/MyDrive/Research_Project/definitive_model.pt"
SAVE_LINE        = "/content/drive/MyDrive/Research_Project/optimal_fatigue_line.png"
SAVE_BAR         = "/content/drive/MyDrive/Research_Project/optimal_fatigue_bar.png"
SAVE_CSV         = "/content/drive/MyDrive/Research_Project/optimal_fatigue_results.csv"

# ---------------------------------------------------------------------------
# Config — must match training script exactly
# ---------------------------------------------------------------------------

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

RAMP_START_W = 50
RAMP_END_W   = 700
RAMP_STEP_W  = 10
STEP_SAMPLES = 60
HR_THRESHOLD = 0.90
HR_PERCENTILE= 99

# TSB sweep range
TSB_VALUES   = list(range(-40, 25, 5))   # -40, -35, ..., +20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        packed        = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed, (h0, c0))
        out, _        = pad_packed_sequence(out_packed, batch_first=True)
        out           = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Data helpers (identical to other scripts)
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


def load_all_rides(dataset_path, fatigue_lookup):
    athlete_rides = {}
    with zipfile.ZipFile(dataset_path, "r") as outer:
        athlete_zips = sorted(n for n in outer.namelist() if n.endswith(".zip"))
        for i, az_name in enumerate(athlete_zips, 1):
            if i % 20 == 0:
                print(f"  Loading athlete {i}/{len(athlete_zips)}...")
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


def split_athletes_three_way(athlete_ids, train_ratio, val_ratio, seed):
    ids = sorted(athlete_ids)
    random.seed(seed)
    random.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    n_val   = int(len(ids) * val_ratio)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def normalize_rides(rides, stats):
    fm, fs = stats["feat_mean"], stats["feat_std"]
    hm, hs = stats["hr_mean"],   stats["hr_std"]
    return [((f - fm) / (fs + 1e-8), (h - hm) / (hs + 1e-8)) for f, h in rides]


def denormalize_hr(hr_norm, stats):
    return hr_norm * stats["hr_std"] + stats["hr_mean"]


def adapt_latent(model, mean_latent, rides_norm):
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
# Ramp at a given fatigue state
# ---------------------------------------------------------------------------

def estimate_max_power_at_tsb(model, latent, ctl, tsb, hr_threshold, stats):
    """Run the power ramp with fixed CTL and given TSB. Returns estimated max power (or None)."""
    atl          = ctl - tsb
    power_levels = list(range(RAMP_START_W, RAMP_END_W + RAMP_STEP_W, RAMP_STEP_W))
    T            = len(power_levels) * STEP_SAMPLES
    power_seq    = np.repeat(np.array(power_levels, dtype=np.float32), STEP_SAMPLES)

    features = np.stack([
        power_seq,
        np.full(T, atl,  dtype=np.float32),
        np.full(T, ctl,  dtype=np.float32),
        np.full(T, tsb,  dtype=np.float32),
    ], axis=1)

    feat_norm = (features - stats["feat_mean"]) / (stats["feat_std"] + 1e-8)
    feat_t    = torch.from_numpy(feat_norm).unsqueeze(0).to(DEVICE)
    length    = torch.tensor([len(features)], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        pred_norm = model(feat_t, length, latent.unsqueeze(0))
    pred_bpm = denormalize_hr(pred_norm.squeeze(0).cpu().numpy(), stats)

    # Mean predicted HR per step (second half = steady state)
    n_steps           = len(power_levels)
    predicted_step_hr = np.zeros(n_steps, dtype=np.float32)
    for i in range(n_steps):
        s = i * STEP_SAMPLES
        predicted_step_hr[i] = pred_bpm[s + STEP_SAMPLES // 2 : s + STEP_SAMPLES].mean()

    # Find first crossing
    for i, (p, hr) in enumerate(zip(power_levels, predicted_step_hr)):
        if hr >= hr_threshold:
            if i > 0:
                p_prev  = power_levels[i - 1]
                hr_prev = predicted_step_hr[i - 1]
                frac    = (hr_threshold - hr_prev) / (hr - hr_prev + 1e-8)
                return p_prev + frac * (p - p_prev)
            return float(p)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device: {DEVICE}")

    # Load model
    print("Loading model...")
    checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    stats       = checkpoint["stats"]

    model = ImprovedLSTM(INPUT_SIZE, HIDDEN_SIZE, LATENT_DIM, DROPOUT).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    embedding = nn.Embedding(len(checkpoint["train_athlete_index"]), LATENT_DIM).to(DEVICE)
    embedding.load_state_dict(checkpoint["embedding_state"])

    with torch.no_grad():
        mean_latent = embedding.weight.mean(dim=0).detach()

    # Load data
    print("Loading fatigue features...")
    fatigue_lookup = load_fatigue_lookup(ATL_CTL_TSB_PATH)
    fatigue_df     = pd.read_csv(ATL_CTL_TSB_PATH, dtype={"date": str})

    print("Loading rides...")
    athlete_rides = load_all_rides(DATASET_PATH, fatigue_lookup)

    _, _, test_ids = split_athletes_three_way(
        list(athlete_rides.keys()), TRAIN_RATIO, VAL_RATIO, SEED
    )
    test_ids = [aid for aid in test_ids if aid in athlete_rides]
    print(f"  {len(test_ids)} test athletes\n")

    # --- TSB sweep for each test athlete ---
    all_curves   = []   # shape (n_athletes, n_tsb_values)
    optimal_tsbs = []
    rows         = []

    for i, athlete_id in enumerate(test_ids, 1):
        print(f"  Athlete {i}/{len(test_ids)}: {athlete_id}")

        rides      = athlete_rides[athlete_id]
        rides_norm = normalize_rides(rides, stats)
        n_adapt    = max(1, int(len(rides_norm) * ADAPT_RATIO))
        latent     = adapt_latent(model, mean_latent, rides_norm[:n_adapt])

        # HR threshold from actual ride data
        all_hr     = np.concatenate([r[1] for r in rides])
        obs_max_hr = float(np.percentile(all_hr, HR_PERCENTILE))
        threshold  = HR_THRESHOLD * obs_max_hr

        # Fixed CTL at athlete median
        ath_df     = fatigue_df[fatigue_df["athlete_id"] == athlete_id]
        median_ctl = float(ath_df["ctl_pre"].median()) if len(ath_df) > 0 else 60.0

        curve = []
        for tsb in TSB_VALUES:
            max_p = estimate_max_power_at_tsb(model, latent, median_ctl, tsb, threshold, stats)
            curve.append(max_p if max_p is not None else np.nan)
            print(f"    TSB={tsb:+3d}  →  {max_p:.0f} W" if max_p else f"    TSB={tsb:+3d}  →  not reached")

        curve_arr = np.array(curve)
        all_curves.append(curve_arr)

        # Optimal TSB = TSB with highest estimated max power
        valid_mask = ~np.isnan(curve_arr)
        if valid_mask.any():
            best_idx     = int(np.nanargmax(curve_arr))
            optimal_tsb  = TSB_VALUES[best_idx]
            optimal_power = curve_arr[best_idx]
        else:
            optimal_tsb   = None
            optimal_power = None

        optimal_tsbs.append(optimal_tsb)
        rows.append({
            "athlete_id":    athlete_id,
            "median_ctl":    round(median_ctl, 1),
            "optimal_tsb":   optimal_tsb,
            "optimal_power_w": round(optimal_power, 1) if optimal_power else None,
        })
        print(f"    → Optimal TSB: {optimal_tsb}  (est. max power: {optimal_power:.0f} W)\n"
              if optimal_power else f"    → No valid result\n")

    # Save CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(SAVE_CSV, index=False)
    print(df_out.to_string(index=False))

    # --- Plot 1: Max power vs TSB curves ---
    all_curves = np.array(all_curves, dtype=float)
    mean_curve = np.nanmean(all_curves, axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))

    for curve in all_curves:
        ax.plot(TSB_VALUES, curve, color="gray", linewidth=1.0, alpha=0.35)

    ax.plot(TSB_VALUES, mean_curve, color="#1f77b4", linewidth=2.5,
            label=f"Mean across {len(test_ids)} athletes")

    # Mark the mean optimal TSB
    best_mean_idx = int(np.nanargmax(mean_curve))
    ax.axvline(TSB_VALUES[best_mean_idx], color="#d62728", linewidth=1.5, linestyle="--",
               label=f"Mean optimal TSB = {TSB_VALUES[best_mean_idx]}")

    ax.set_xlabel("TSB (Training Stress Balance)", fontsize=12)
    ax.set_ylabel("Estimated Max Sustainable Power (W)", fontsize=12)
    ax.set_title("Effect of Fatigue State on Estimated Maximum Power\n"
                 "CTL held fixed at each athlete's median — only TSB varies", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_LINE, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {SAVE_LINE}")

    # --- Plot 2: Optimal TSB per athlete ---
    valid_rows = df_out.dropna(subset=["optimal_tsb"]).sort_values("optimal_tsb")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bars = ax2.bar(range(len(valid_rows)), valid_rows["optimal_tsb"],
                   color="#1f77b4", edgecolor="none")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.axhline(valid_rows["optimal_tsb"].mean(), color="#d62728", linewidth=1.5,
                linestyle="--", label=f"Mean optimal TSB = {valid_rows['optimal_tsb'].mean():.1f}")
    ax2.set_xticks(range(len(valid_rows)))
    ax2.set_xticklabels([f"Ath {i+1}" for i in range(len(valid_rows))], fontsize=9)
    ax2.set_ylabel("Optimal TSB", fontsize=12)
    ax2.set_title("Model-Predicted Optimal Fatigue State per Test Athlete\n"
                  "(TSB at which estimated max power is highest)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(SAVE_BAR, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {SAVE_BAR}")


if __name__ == "__main__":
    main()
