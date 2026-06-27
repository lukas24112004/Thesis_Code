"""
Fatigue Direction Check — Simple Model (No Latent Vector)
----------------------------------------------------------
Same analysis as trial2_within_athlete_real_rides.py but using the
simple model (no latent vector, no personalization).

Hypothesis: the definitive model's latent vector absorbed so much of
the between-athlete fitness signal that TSB lost its influence. The
simple model has no latent vector, so TSB is forced to carry more of
the prediction weight. Does this result in the correct fatigue direction?

Method (identical to trial2):
  1. For each test athlete: run all their eval rides through the model
  2. Record mean predicted HR (second half of ride), mean power, TSB
  3. Residualize predicted HR on mean power
  4. Regress HR residuals on TSB → slope = fatigue effect

No adaptation step — the simple model has no latent vector to adapt.
The model runs directly on test athlete rides.

Run on Colab after train_simple_model.py completes.
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
from scipy import stats as scipy_stats
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_PATH     = "/content/drive/MyDrive/Research_Project/Dataset_Reduced.zip"
ATL_CTL_TSB_PATH = "/content/drive/MyDrive/Research_Project/tss_atl_ctl_tsb.csv"
MODEL_PATH       = "/content/drive/MyDrive/Research_Project/simple_model.pt"
SAVE_PLOT        = "/content/drive/MyDrive/Research_Project/simple_model_direction.png"
SAVE_CSV         = "/content/drive/MyDrive/Research_Project/simple_model_direction.csv"

# ---------------------------------------------------------------------------
# Config — must match training script
# ---------------------------------------------------------------------------

SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
DOWNSAMPLE  = 10
MIN_STEPS   = 60
MAX_STEPS   = 300
HIDDEN_SIZE = 32
DROPOUT     = 0.2
INPUT_SIZE  = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model
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
# Data helpers
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


def predict_ride(model, features_norm, stats):
    feat_t = torch.from_numpy(features_norm).unsqueeze(0).to(DEVICE)
    length = torch.tensor([len(features_norm)], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pred_norm = model(feat_t, length)
    return denormalize_hr(pred_norm.squeeze(0).cpu().numpy(), stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device: {DEVICE}")

    # Load model
    print("Loading simple model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    stats      = checkpoint["stats"]

    model = SimpleLSTM(INPUT_SIZE, HIDDEN_SIZE, DROPOUT).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load data
    print("Loading fatigue features...")
    fatigue_lookup = load_fatigue_lookup(ATL_CTL_TSB_PATH)

    print("Loading rides...")
    athlete_rides = load_all_rides(DATASET_PATH, fatigue_lookup)

    _, _, test_ids = split_athletes_three_way(
        list(athlete_rides.keys()), TRAIN_RATIO, VAL_RATIO, SEED
    )
    test_ids = [aid for aid in test_ids if aid in athlete_rides]
    print(f"  {len(test_ids)} test athletes\n")

    # --- Direction analysis ---
    results = []
    all_coefs = []

    n_cols = 3
    n_rows = (len(test_ids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes_flat = axes.flatten()

    for ax_idx, athlete_id in enumerate(test_ids):
        rides      = athlete_rides[athlete_id]
        rides_norm = normalize_rides(rides, stats)

        # No adaptation — simple model runs directly on all rides
        # Use same 70/30 split for consistency: first 30% excluded (would be adapt set)
        n_adapt  = max(1, int(len(rides_norm) * 0.30))
        eval_set = rides_norm[n_adapt:]
        raw_eval = rides[n_adapt:]

        tsb_vals, power_vals, hr_pred_vals = [], [], []

        for ride_idx, (features_norm, hr_norm) in enumerate(eval_set):
            pred_hr   = predict_ride(model, features_norm, stats)
            half      = len(pred_hr) // 2
            mean_pred = pred_hr[half:].mean()
            raw_feat  = raw_eval[ride_idx][0]
            mean_pow  = raw_feat[:, 0].mean()
            tsb       = raw_feat[0, 3]

            tsb_vals.append(tsb)
            power_vals.append(mean_pow)
            hr_pred_vals.append(mean_pred)

        tsb_arr   = np.array(tsb_vals)
        power_arr = np.array(power_vals)
        hr_arr    = np.array(hr_pred_vals)

        # Residualize HR on power
        if len(power_arr) > 2:
            coeffs   = np.polyfit(power_arr, hr_arr, 1)
            hr_resid = hr_arr - np.polyval(coeffs, power_arr)
        else:
            hr_resid = hr_arr

        # Regress TSB on HR residuals
        if len(tsb_arr) > 2:
            slope, intercept, r, p, _ = scipy_stats.linregress(tsb_arr, hr_resid)
        else:
            slope, r, p = 0.0, 0.0, 1.0

        direction = "correct ↓" if slope < 0 else "WRONG ↑"
        all_coefs.append(slope)

        results.append({
            "athlete_id": athlete_id,
            "tsb_coef":   round(slope, 4),
            "r":          round(r, 3),
            "p_value":    round(p, 3),
            "n_rides":    len(eval_set),
            "direction":  "correct" if slope < 0 else "wrong",
        })

        print(f"  Athlete {ax_idx+1:2d}: coef={slope:+.4f}  r={r:.3f}  p={p:.3f}"
              f"  n={len(eval_set)}  [{direction}]")

        # Plot
        ax = axes_flat[ax_idx]
        ax.scatter(tsb_arr, hr_resid, alpha=0.4, s=15, color="steelblue", edgecolors="none")
        if len(tsb_arr) > 2:
            x_line = np.linspace(tsb_arr.min(), tsb_arr.max(), 100)
            color  = "green" if slope < 0 else "crimson"
            ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2,
                    label=f"slope={slope:+.3f} ({direction})")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"Athlete {ax_idx+1}  (n={len(eval_set)} rides)", fontsize=10)
        ax.set_xlabel("TSB", fontsize=8)
        ax.set_ylabel("HR residual (bpm)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(len(test_ids), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    n_correct = sum(1 for r in results if r["direction"] == "correct")
    mean_coef = np.mean(all_coefs)

    fig.suptitle(f"Simple Model — Fatigue Direction Analysis\n"
                 f"Correct direction: {n_correct}/{len(test_ids)}  |  "
                 f"Mean TSB coef: {mean_coef:+.4f} bpm/unit",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(SAVE_PLOT, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved: {SAVE_PLOT}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Correct direction: {n_correct}/{len(test_ids)} athletes")
    print(f"Mean TSB coef:     {mean_coef:+.4f} bpm/unit")
    print(f"\nComparison:")
    print(f"  Definitive model (with latent vector): 3/15 correct, mean coef +0.10")
    print(f"  Simple model (no latent vector):       {n_correct}/{len(test_ids)} correct, mean coef {mean_coef:+.4f}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(SAVE_CSV, index=False)
    print(f"\nResults saved: {SAVE_CSV}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
