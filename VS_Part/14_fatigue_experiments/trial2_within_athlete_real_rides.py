"""
Fatigue Effect on Predicted HR — Real Rides Analysis
-----------------------------------------------------
Instead of sweeping synthetic TSB values, this script uses real rides from
each test athlete with their actual TSB values. This eliminates the between-
athlete fitness confound that affected the synthetic sweep.

Method per test athlete:
  1. Re-adapt their latent vector on their first 30% of rides (LSTM frozen)
  2. For each eval ride (70%), run the model and record:
       - Mean predicted HR (second half of ride, to avoid warmup)
       - Mean power (second half of ride, same window)
       - TSB value for that ride (constant within a ride)
  3. Residualize predicted HR on mean power (remove the effect of riding harder)
  4. Regress HR residuals on TSB → slope = fatigue effect (bpm per TSB unit)

Aggregated across all test athletes:
  - Distribution of TSB coefficients
  - Mean effect and % of athletes with physiologically correct direction (negative slope)
  - Expected: higher TSB (more rested) → lower predicted HR at same power

Outputs:
  - fatigue_real_rides.png : per-athlete scatter + trend lines
  - fatigue_coefficients.csv : per-athlete TSB coefficient, r, p-value, n_rides
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
# Config — must match training script exactly
# ---------------------------------------------------------------------------

DATASET_PATH     = "/content/drive/MyDrive/Research_Project/Dataset_Reduced.zip"
ATL_CTL_TSB_PATH = "/content/drive/MyDrive/Research_Project/tss_atl_ctl_tsb.csv"
MODEL_PATH       = "/content/drive/MyDrive/Research_Project/definitive_model.pt"
PLOT_SAVE        = "/content/drive/MyDrive/Research_Project/fatigue_real_rides.png"
COEF_SAVE        = "/content/drive/MyDrive/Research_Project/fatigue_coefficients.csv"

SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
DOWNSAMPLE  = 10
MIN_STEPS   = 60
MAX_STEPS   = 300

LATENT_DIM  = 8
HIDDEN_SIZE = 64
DROPOUT     = 0.2
INPUT_SIZE  = 4

ADAPT_RATIO  = 0.30
ADAPT_EPOCHS = 5
ADAPT_LR     = 0.01

MIN_EVAL_RIDES = 10   # minimum eval rides needed for a meaningful regression

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
        packed        = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                             enforce_sorted=False)
        out_packed, _ = self.lstm(packed, (h0, c0))
        out, _        = pad_packed_sequence(out_packed, batch_first=True)
        out           = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Data loading
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
# Split and normalization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Latent adaptation
# ---------------------------------------------------------------------------

def adapt_latent(model, mean_latent, adapt_set):
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
            mask = torch.arange(pred.size(1), device=DEVICE) < length.to(DEVICE)
            loss = ((pred - hr_t) ** 2 * mask).sum() / mask.sum()
            loss.backward()
            adapt_opt.step()

    return latent.detach()


# ---------------------------------------------------------------------------
# Per-ride prediction
# ---------------------------------------------------------------------------

def predict_ride(model, latent, features_norm, stats):
    """
    Run model on one normalized ride.
    Returns (mean_pred_hr_bpm, mean_power_watts, tsb_value) using second half of ride.
    """
    T      = len(features_norm)
    half   = T // 2

    feat_t = torch.from_numpy(features_norm).unsqueeze(0).to(DEVICE)
    length = torch.tensor([T], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        pred_norm = model(feat_t, length, latent.unsqueeze(0))

    pred_bpm = pred_norm.squeeze(0).cpu().numpy() * stats["hr_std"] + stats["hr_mean"]

    # Denormalize power and TSB from normalized features
    power_norm = features_norm[:, 0]
    tsb_norm   = features_norm[0, 3]   # constant within ride

    power_watts = power_norm * stats["feat_std"][0] + stats["feat_mean"][0]
    tsb_value   = tsb_norm   * stats["feat_std"][3] + stats["feat_mean"][3]

    # Use second half to avoid warmup effects
    mean_pred_hr  = float(pred_bpm[half:].mean())
    mean_power    = float(power_watts[half:].mean())

    return mean_pred_hr, mean_power, float(tsb_value)


# ---------------------------------------------------------------------------
# Per-athlete analysis
# ---------------------------------------------------------------------------

def analyse_athlete(model, latent, eval_set, stats):
    """
    For each eval ride: get predicted HR, mean power, TSB.
    Residualize HR on power, regress residuals on TSB.
    Returns dict with coefficient, stats, and raw arrays.
    """
    pred_hrs, powers, tsbs = [], [], []

    for features_norm, _ in eval_set:
        hr, power, tsb = predict_ride(model, latent, features_norm, stats)
        pred_hrs.append(hr)
        powers.append(power)
        tsbs.append(tsb)

    pred_hrs = np.array(pred_hrs)
    powers   = np.array(powers)
    tsbs     = np.array(tsbs)

    # Step 1: residualize HR on power
    slope_p, intercept_p, _, _, _ = scipy_stats.linregress(powers, pred_hrs)
    hr_resid = pred_hrs - (slope_p * powers + intercept_p)

    # Step 2: regress residuals on TSB
    slope_tsb, intercept_tsb, r, p, _ = scipy_stats.linregress(tsbs, hr_resid)

    return {
        "tsb_coef":   slope_tsb,    # bpm per TSB unit — negative = physiologically correct
        "r":          r,
        "p_value":    p,
        "n_rides":    len(tsbs),
        "tsbs":       tsbs,
        "hr_resid":   hr_resid,
    }


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

    # Analyse each test athlete
    results     = {}
    coef_rows   = []

    for i, athlete_id in enumerate(test_ids, 1):
        rides      = athlete_rides[athlete_id]
        rides_norm = normalize_rides(rides, stats)

        n_adapt   = max(1, int(len(rides_norm) * ADAPT_RATIO))
        adapt_set = rides_norm[:n_adapt]
        eval_set  = rides_norm[n_adapt:] if len(rides_norm) > n_adapt else rides_norm

        if len(eval_set) < MIN_EVAL_RIDES:
            print(f"  Athlete {i}/{len(test_ids)}: skipped (only {len(eval_set)} eval rides)")
            continue

        latent = adapt_latent(model, mean_latent, adapt_set)
        res    = analyse_athlete(model, latent, eval_set, stats)
        results[athlete_id] = res

        direction = "correct" if res["tsb_coef"] < 0 else "WRONG"
        print(f"  Athlete {i}/{len(test_ids)}: "
              f"TSB coef={res['tsb_coef']:+.4f} bpm/unit  "
              f"r={res['r']:.3f}  p={res['p_value']:.3f}  "
              f"n={res['n_rides']}  [{direction}]")

        coef_rows.append({
            "athlete_id": athlete_id,
            "tsb_coef":   round(res["tsb_coef"],  4),
            "r":          round(res["r"],           3),
            "p_value":    round(res["p_value"],     4),
            "n_rides":    res["n_rides"],
        })

    # Save coefficients
    pd.DataFrame(coef_rows).to_csv(COEF_SAVE, index=False)

    # Summary stats
    coefs = np.array([r["tsb_coef"] for r in results.values()])
    n_correct = (coefs < 0).sum()
    print(f"\nSummary across {len(coefs)} athletes:")
    print(f"  Mean TSB coefficient : {coefs.mean():+.4f} bpm per TSB unit")
    print(f"  Median TSB coefficient: {np.median(coefs):+.4f} bpm per TSB unit")
    print(f"  Correct direction (negative): {n_correct}/{len(coefs)} athletes ({100*n_correct/len(coefs):.0f}%)")
    print(f"  Effect of TSB +40 swing: {coefs.mean()*40:+.2f} bpm on average")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: per-athlete scatter + trend lines
    ax = axes[0]
    all_tsbs, all_resids = [], []

    for (athlete_id, res), color in zip(results.items(), colors):
        tsbs    = res["tsbs"]
        resids  = res["hr_resid"]
        coef    = res["tsb_coef"]

        ax.scatter(tsbs, resids, color=color, alpha=0.3, s=10)

        x_line = np.linspace(tsbs.min(), tsbs.max(), 50)
        y_line = coef * x_line + (resids.mean() - coef * tsbs.mean())
        ax.plot(x_line, y_line, color=color, linewidth=1.2, alpha=0.7)

        all_tsbs.append(tsbs)
        all_resids.append(resids)

    # Overall trend line
    all_tsbs_cat   = np.concatenate(all_tsbs)
    all_resids_cat = np.concatenate(all_resids)
    slope_all, intercept_all, _, _, _ = scipy_stats.linregress(all_tsbs_cat, all_resids_cat)
    x_all = np.linspace(all_tsbs_cat.min(), all_tsbs_cat.max(), 100)
    ax.plot(x_all, slope_all * x_all + intercept_all,
            color="black", linewidth=2.5, label=f"Overall trend ({slope_all:+.3f} bpm/TSB)")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("TSB (Training Stress Balance)", fontsize=12)
    ax.set_ylabel("Predicted HR residual (bpm, power-adjusted)", fontsize=12)
    ax.set_title("TSB vs Predicted HR\n(power-adjusted, per-athlete)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: histogram of TSB coefficients
    ax2 = axes[1]
    ax2.hist(coefs, bins=8, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(0,           color="red",   linewidth=1.5, linestyle="--", label="No effect")
    ax2.axvline(coefs.mean(), color="black", linewidth=2.0,
                label=f"Mean = {coefs.mean():+.4f} bpm/TSB unit")
    ax2.set_xlabel("TSB coefficient (bpm per TSB unit)", fontsize=12)
    ax2.set_ylabel("Number of athletes", fontsize=12)
    ax2.set_title("Distribution of Fatigue Effect\nacross test athletes", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE, dpi=150)
    plt.show()
    print(f"\nPlot saved  : {PLOT_SAVE}")
    print(f"Coefficients: {COEF_SAVE}")


if __name__ == "__main__":
    main()
