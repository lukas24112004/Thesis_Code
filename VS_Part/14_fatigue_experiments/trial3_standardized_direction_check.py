"""
Fatigue Effect — Hopefully Fixed Model
----------------------------------------
Same real-rides fatigue analysis as fatigue_real_rides.py but using the
within-athlete standardized model.

Key difference in this script:
  Before running any ride through the model, ATL/CTL/TSB are standardized
  using the athlete's own adaptation-set stats. This matches exactly how the
  model was trained.

For the regression analysis we use the RAW TSB values (not standardized),
so the coefficient is still interpretable in bpm per TSB unit and directly
comparable to the original model's results.
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
# Config — must match validation_hopefully_fixed.py exactly
# ---------------------------------------------------------------------------

DATASET_PATH     = "/content/drive/MyDrive/Research_Project/Dataset_Reduced.zip"
ATL_CTL_TSB_PATH = "/content/drive/MyDrive/Research_Project/tss_atl_ctl_tsb.csv"
MODEL_PATH       = "/content/drive/MyDrive/Research_Project/hopefully_fixed_model.pt"
PLOT_SAVE        = "/content/drive/MyDrive/Research_Project/fatigue_real_rides_fixed.png"
COEF_SAVE        = "/content/drive/MyDrive/Research_Project/fatigue_coefficients_fixed.csv"

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

MIN_EVAL_RIDES = 10

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
# Split
# ---------------------------------------------------------------------------

def split_athletes_three_way(athlete_ids, train_ratio, val_ratio, seed):
    ids = sorted(athlete_ids)
    random.seed(seed)
    random.shuffle(ids)
    n_train = int(len(ids) * train_ratio)
    n_val   = int(len(ids) * val_ratio)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


# ---------------------------------------------------------------------------
# Within-athlete standardization (must match training script)
# ---------------------------------------------------------------------------

def compute_athlete_fatigue_stats(rides):
    atl_vals = np.array([r[0][0, 1] for r in rides])
    ctl_vals = np.array([r[0][0, 2] for r in rides])
    tsb_vals = np.array([r[0][0, 3] for r in rides])
    return {
        "atl_mean": atl_vals.mean(), "atl_std": max(atl_vals.std(), 1e-4),
        "ctl_mean": ctl_vals.mean(), "ctl_std": max(ctl_vals.std(), 1e-4),
        "tsb_mean": tsb_vals.mean(), "tsb_std": max(tsb_vals.std(), 1e-4),
    }


def apply_athlete_standardization(rides, ath_stats):
    standardized = []
    for features, hr in rides:
        f = features.copy()
        f[:, 1] = (f[:, 1] - ath_stats["atl_mean"]) / ath_stats["atl_std"]
        f[:, 2] = (f[:, 2] - ath_stats["ctl_mean"]) / ath_stats["ctl_std"]
        f[:, 3] = (f[:, 3] - ath_stats["tsb_mean"]) / ath_stats["tsb_std"]
        standardized.append((f, hr))
    return standardized


def normalize_rides(rides, stats):
    fm, fs = stats["feat_mean"], stats["feat_std"]
    hm, hs = stats["hr_mean"],   stats["hr_std"]
    return [((f - fm) / (fs + 1e-8), (h - hm) / (hs + 1e-8)) for f, h in rides]


# ---------------------------------------------------------------------------
# Latent adaptation
# ---------------------------------------------------------------------------

def adapt_latent(model, mean_latent, adapt_set_norm):
    latent    = nn.Parameter(mean_latent.clone().to(DEVICE))
    adapt_opt = torch.optim.Adam([latent], lr=ADAPT_LR)

    model.train()
    for _ in range(ADAPT_EPOCHS):
        for features, hr in adapt_set_norm:
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

def predict_ride(model, latent, features_norm, raw_tsb, stats):
    """
    Run model on one normalized+standardized ride.
    Returns (mean_pred_hr_bpm, mean_power_watts, raw_tsb_value).
    raw_tsb is passed in directly since it's been transformed in the features.
    """
    T    = len(features_norm)
    half = T // 2

    feat_t = torch.from_numpy(features_norm).unsqueeze(0).to(DEVICE)
    length = torch.tensor([T], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        pred_norm = model(feat_t, length, latent.unsqueeze(0))

    pred_bpm    = pred_norm.squeeze(0).cpu().numpy() * stats["hr_std"] + stats["hr_mean"]
    power_norm  = features_norm[:, 0]
    power_watts = power_norm * stats["feat_std"][0] + stats["feat_mean"][0]

    mean_pred_hr = float(pred_bpm[half:].mean())
    mean_power   = float(power_watts[half:].mean())

    return mean_pred_hr, mean_power, raw_tsb


# ---------------------------------------------------------------------------
# Per-athlete analysis
# ---------------------------------------------------------------------------

def analyse_athlete(model, latent, eval_set_norm, eval_raw_tsbs, stats):
    pred_hrs, powers, tsbs = [], [], []

    for (features_norm, _), raw_tsb in zip(eval_set_norm, eval_raw_tsbs):
        hr, power, tsb = predict_ride(model, latent, features_norm, raw_tsb, stats)
        pred_hrs.append(hr)
        powers.append(power)
        tsbs.append(tsb)

    pred_hrs = np.array(pred_hrs)
    powers   = np.array(powers)
    tsbs     = np.array(tsbs)

    slope_p, intercept_p, _, _, _ = scipy_stats.linregress(powers, pred_hrs)
    hr_resid = pred_hrs - (slope_p * powers + intercept_p)

    slope_tsb, _, r, p, _ = scipy_stats.linregress(tsbs, hr_resid)

    return {
        "tsb_coef": slope_tsb,
        "r":        r,
        "p_value":  p,
        "n_rides":  len(tsbs),
        "tsbs":     tsbs,
        "hr_resid": hr_resid,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device : {DEVICE}")

    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    stats      = checkpoint["stats"]

    model = ImprovedLSTM(INPUT_SIZE, HIDDEN_SIZE, LATENT_DIM, DROPOUT).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    embedding = nn.Embedding(len(checkpoint["train_athlete_index"]), LATENT_DIM).to(DEVICE)
    embedding.load_state_dict(checkpoint["embedding_state"])

    with torch.no_grad():
        mean_latent = embedding.weight.mean(dim=0).detach()

    print("Loading fatigue features...")
    fatigue_lookup = load_fatigue_lookup(ATL_CTL_TSB_PATH)

    print("Loading rides...")
    athlete_rides = load_all_rides(DATASET_PATH, fatigue_lookup)

    train_ids, val_ids, test_ids = split_athletes_three_way(
        list(athlete_rides.keys()), TRAIN_RATIO, VAL_RATIO, SEED
    )
    test_ids = [aid for aid in test_ids if aid in athlete_rides]
    print(f"  {len(test_ids)} test athletes\n")

    results   = {}
    coef_rows = []

    for i, athlete_id in enumerate(test_ids, 1):
        rides_raw = athlete_rides[athlete_id]

        n_adapt   = max(1, int(len(rides_raw) * ADAPT_RATIO))
        adapt_raw = rides_raw[:n_adapt]
        eval_raw  = rides_raw[n_adapt:] if len(rides_raw) > n_adapt else rides_raw

        if len(eval_raw) < MIN_EVAL_RIDES:
            print(f"  Athlete {i}/{len(test_ids)}: skipped (only {len(eval_raw)} eval rides)")
            continue

        # Within-athlete standardization using adapt set stats
        ath_stats = compute_athlete_fatigue_stats(adapt_raw)
        adapt_std = apply_athlete_standardization(adapt_raw, ath_stats)
        eval_std  = apply_athlete_standardization(eval_raw,  ath_stats)

        # Global normalization
        adapt_norm = normalize_rides(adapt_std, stats)
        eval_norm  = normalize_rides(eval_std,  stats)

        # Keep raw TSB values for regression (col 3 of original features)
        eval_raw_tsbs = [r[0][0, 3] for r in eval_raw]

        latent = adapt_latent(model, mean_latent, adapt_norm)
        res    = analyse_athlete(model, latent, eval_norm, eval_raw_tsbs, stats)
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

    pd.DataFrame(coef_rows).to_csv(COEF_SAVE, index=False)

    coefs     = np.array([r["tsb_coef"] for r in results.values()])
    n_correct = (coefs < 0).sum()
    print(f"\nHopefully fixed model — summary across {len(coefs)} athletes:")
    print(f"  Mean TSB coefficient  : {coefs.mean():+.4f} bpm per TSB unit")
    print(f"  Median TSB coefficient: {np.median(coefs):+.4f} bpm per TSB unit")
    print(f"  Correct direction     : {n_correct}/{len(coefs)} athletes ({100*n_correct/len(coefs):.0f}%)")
    print(f"  Effect of TSB +40 swing: {coefs.mean()*40:+.2f} bpm on average")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    all_tsbs, all_resids = [], []

    for (athlete_id, res), color in zip(results.items(), colors):
        tsbs   = res["tsbs"]
        resids = res["hr_resid"]
        coef   = res["tsb_coef"]

        ax.scatter(tsbs, resids, color=color, alpha=0.3, s=10)
        x_line = np.linspace(tsbs.min(), tsbs.max(), 50)
        y_line = coef * x_line + (resids.mean() - coef * tsbs.mean())
        ax.plot(x_line, y_line, color=color, linewidth=1.2, alpha=0.7)

        all_tsbs.append(tsbs)
        all_resids.append(resids)

    all_tsbs_cat   = np.concatenate(all_tsbs)
    all_resids_cat = np.concatenate(all_resids)
    slope_all, intercept_all, _, _, _ = scipy_stats.linregress(all_tsbs_cat, all_resids_cat)
    x_all = np.linspace(all_tsbs_cat.min(), all_tsbs_cat.max(), 100)
    ax.plot(x_all, slope_all * x_all + intercept_all,
            color="black", linewidth=2.5,
            label=f"Overall trend ({slope_all:+.3f} bpm/TSB)")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("TSB (Training Stress Balance)", fontsize=12)
    ax.set_ylabel("Predicted HR residual (bpm, power-adjusted)", fontsize=12)
    ax.set_title("Fixed Model: TSB vs Predicted HR\n(power-adjusted, per-athlete)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(coefs, bins=8, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(0,            color="red",   linewidth=1.5, linestyle="--", label="No effect")
    ax2.axvline(coefs.mean(), color="black", linewidth=2.0,
                label=f"Mean = {coefs.mean():+.4f} bpm/TSB unit")
    ax2.set_xlabel("TSB coefficient (bpm per TSB unit)", fontsize=12)
    ax2.set_ylabel("Number of athletes", fontsize=12)
    ax2.set_title("Fixed Model: Distribution of Fatigue Effect\nacross test athletes", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE, dpi=150)
    plt.show()
    print(f"\nPlot saved        : {PLOT_SAVE}")
    print(f"Coefficients saved: {COEF_SAVE}")


if __name__ == "__main__":
    main()
