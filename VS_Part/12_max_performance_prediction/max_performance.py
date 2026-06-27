"""
Maximum Sustainable Power Estimation
--------------------------------------
Loads the trained definitive model and estimates the maximum sustainable
power for each test athlete — defined as the power level where predicted HR
crosses 90 % of that athlete's robust observed maximum HR.

Method:
  - Reconstruct the same train/test split (same seed) so we know who the test
    athletes are
  - Re-adapt each test athlete's latent vector on their first 30 % of rides
  - Build a step-ramp power sequence (50 W → 700 W in 10 W increments, each
    step held for 60 samples after down-sampling at 10 s resolution = 10 min/step)
  - Run the model on that ramp with the athlete's median ATL/CTL/TSB values
  - Find the first power level at which the mean predicted HR in the second
    half of that step exceeds 90 % of the athlete's robust observed max HR
  - If the model never reaches that threshold, report NaN for that athlete

Why 90 % of observed max HR:
  90 % of max HR corresponds roughly to threshold/tempo effort, which is the
  physiological definition of maximum sustainable power. Values above 95 %
  represent VO2max territory (short hard efforts, not sustainable), so 90 %
  is the appropriate target.

Why 99th percentile for observed max HR:
  Raw maximum HR is corrupted by sensor spikes (e.g. 595 bpm readings from
  a faulty chest strap). A single bad sample inflates the threshold to an
  unreachable level. The 99th percentile is robust to these artifacts while
  still capturing the athlete's genuine near-maximum HR.

Run on Google Colab after training completes. Saves plot and CSV to Drive.
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
PLOT_SAVE        = "/content/drive/MyDrive/Research_Project/max_performance.png"
RESULTS_SAVE     = "/content/drive/MyDrive/Research_Project/max_performance.csv"

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

# Ramp parameters
RAMP_START_W   = 50      # watts
RAMP_END_W     = 700     # watts — raised to avoid ceiling for fit athletes
RAMP_STEP_W    = 10      # watts per increment
STEP_SAMPLES   = 60      # samples per step (60 × 10 s = 10 min per step at 1-s resolution)
HR_THRESHOLD   = 0.90    # fraction of observed max HR (≈ threshold/tempo effort)
HR_PERCENTILE  = 99      # percentile used to estimate max HR (robust to sensor spikes)

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
# Ramp sequence builder
# ---------------------------------------------------------------------------

def build_ramp_sequence(atl, ctl, tsb):
    """
    Builds a step-ramp power sequence from RAMP_START_W to RAMP_END_W.
    Each step lasts STEP_SAMPLES time steps.
    Returns feature array of shape (T, 4): [power, atl, ctl, tsb].
    Also returns the power level for each step.
    """
    power_levels = list(range(RAMP_START_W, RAMP_END_W + RAMP_STEP_W, RAMP_STEP_W))
    T = len(power_levels) * STEP_SAMPLES

    power_seq = np.repeat(np.array(power_levels, dtype=np.float32), STEP_SAMPLES)
    features  = np.stack([
        power_seq,
        np.full(T, atl,  dtype=np.float32),
        np.full(T, ctl,  dtype=np.float32),
        np.full(T, tsb,  dtype=np.float32),
    ], axis=1)

    return features, power_levels


# ---------------------------------------------------------------------------
# Max power estimation for one athlete
# ---------------------------------------------------------------------------

def estimate_max_power(model, latent, athlete_id, rides, fatigue_df, stats):
    """
    Runs a power ramp through the model and finds the first step where
    predicted HR exceeds HR_THRESHOLD * observed_max_hr.

    Returns a dict with:
      - observed_max_hr
      - hr_threshold (= 0.9 × observed_max_hr)
      - estimated_max_power_w (None if threshold never crossed)
      - power_levels and predicted_step_hr arrays for plotting
    """
    # Robust observed max HR — 99th percentile guards against sensor spikes
    all_hr     = np.concatenate([r[1] for r in rides])
    obs_max_hr = float(np.percentile(all_hr, HR_PERCENTILE))
    threshold  = HR_THRESHOLD * obs_max_hr

    # Athlete median fatigue state
    ath_df     = fatigue_df[fatigue_df["athlete_id"] == athlete_id]
    median_ctl = float(ath_df["ctl_pre"].median()) if len(ath_df) > 0 else 60.0
    median_atl = float(ath_df["atl_pre"].median()) if len(ath_df) > 0 else 60.0
    median_tsb = float(ath_df["tsb_pre"].median()) if len(ath_df) > 0 else 0.0

    features, power_levels = build_ramp_sequence(median_atl, median_ctl, median_tsb)

    feat_norm = normalize_features(features, stats)
    feat_t    = torch.from_numpy(feat_norm).unsqueeze(0).to(DEVICE)
    length    = torch.tensor([len(features)], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        pred_norm = model(feat_t, length, latent.unsqueeze(0))

    pred_bpm = denormalize_hr(pred_norm.squeeze(0).cpu().numpy(), stats)

    # Mean predicted HR per step (use second half of each step for steady-state)
    n_steps           = len(power_levels)
    predicted_step_hr = np.zeros(n_steps, dtype=np.float32)
    for i in range(n_steps):
        step_start = i * STEP_SAMPLES
        step_mid   = step_start + STEP_SAMPLES // 2
        step_end   = step_start + STEP_SAMPLES
        predicted_step_hr[i] = pred_bpm[step_mid:step_end].mean()

    # Find first crossing
    estimated_max_power = None
    for i, (p, hr) in enumerate(zip(power_levels, predicted_step_hr)):
        if hr >= threshold:
            if i > 0:
                # Linear interpolation between previous and current step
                p_prev  = power_levels[i - 1]
                hr_prev = predicted_step_hr[i - 1]
                frac    = (threshold - hr_prev) / (hr - hr_prev + 1e-8)
                estimated_max_power = p_prev + frac * (p - p_prev)
            else:
                estimated_max_power = float(p)
            break

    return {
        "observed_max_hr":       obs_max_hr,
        "hr_threshold":          threshold,
        "estimated_max_power_w": estimated_max_power,
        "median_ctl":            median_ctl,
        "power_levels":          power_levels,
        "predicted_step_hr":     predicted_step_hr.tolist(),
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
    fatigue_df     = pd.read_csv(ATL_CTL_TSB_PATH, dtype={"date": str})

    print("Loading rides...")
    athlete_rides = load_all_rides(DATASET_PATH, fatigue_lookup)

    # Reconstruct same split
    train_ids, val_ids, test_ids = split_athletes_three_way(
        list(athlete_rides.keys()), TRAIN_RATIO, VAL_RATIO, SEED
    )
    train_athlete_index = checkpoint["train_athlete_index"]  # maps athlete_id -> embedding index

    all_athlete_ids = (
        [(aid, "train") for aid in train_ids if aid in athlete_rides] +
        [(aid, "val")   for aid in val_ids   if aid in athlete_rides] +
        [(aid, "test")  for aid in test_ids  if aid in athlete_rides]
    )
    print(f"  {len(all_athlete_ids)} total athletes "
          f"({sum(1 for _,s in all_athlete_ids if s=='train')} train, "
          f"{sum(1 for _,s in all_athlete_ids if s=='val')} val, "
          f"{sum(1 for _,s in all_athlete_ids if s=='test')} test)\n")

    # Estimate max power for all athletes
    all_results  = {}
    athlete_sets = {}

    for i, (athlete_id, split) in enumerate(all_athlete_ids, 1):
        print(f"  Athlete {i}/{len(all_athlete_ids)} [{split}]: {athlete_id}")

        rides      = athlete_rides[athlete_id]
        rides_norm = normalize_rides(rides, stats)

        if split == "train" and athlete_id in train_athlete_index:
            # Use the pre-learned embedding directly — no adaptation needed
            idx    = train_athlete_index[athlete_id]
            latent = embedding.weight[idx].detach()
        else:
            # Adapt latent on first ADAPT_RATIO of rides (val and test athletes)
            n_adapt   = max(1, int(len(rides_norm) * ADAPT_RATIO))
            adapt_set = rides_norm[:n_adapt]
            latent    = adapt_latent(model, mean_latent, adapt_set)

        result = estimate_max_power(model, latent, athlete_id, rides, fatigue_df, stats)
        all_results[athlete_id]  = result
        athlete_sets[athlete_id] = split

        max_p = result["estimated_max_power_w"]
        print(f"    Obs max HR: {result['observed_max_hr']:.0f} bpm  "
              f"Threshold: {result['hr_threshold']:.1f} bpm  "
              f"Est. max power: {max_p:.0f} W" if max_p is not None
              else f"    Obs max HR: {result['observed_max_hr']:.0f} bpm  "
                   f"Threshold: {result['hr_threshold']:.1f} bpm  "
                   f"Est. max power: not reached")

    # Save results CSV
    rows = []
    for athlete_id, res in all_results.items():
        rows.append({
            "athlete_id":              athlete_id,
            "split":                   athlete_sets[athlete_id],
            "obs_max_hr_p99_bpm":      round(res["observed_max_hr"], 1),
            "hr_threshold_bpm":        round(res["hr_threshold"], 1),
            "estimated_max_power_w":   round(res["estimated_max_power_w"], 1)
                                       if res["estimated_max_power_w"] is not None else None,
            "median_ctl":              round(res["median_ctl"], 1),
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS_SAVE, index=False)
    print(f"\nResults saved: {RESULTS_SAVE}")
    print(df_out.to_string(index=False))

    # Summary stats — overall and by split
    valid = df_out["estimated_max_power_w"].dropna()
    print(f"\nAll athletes — n={len(valid)}/{len(df_out)} reached threshold")
    if len(valid) > 0:
        print(f"  Mean:   {valid.mean():.0f} W")
        print(f"  Median: {valid.median():.0f} W")
        print(f"  Range:  {valid.min():.0f}–{valid.max():.0f} W")

    for split_name in ["train", "val", "test"]:
        sub = df_out[df_out["split"] == split_name]["estimated_max_power_w"].dropna()
        if len(sub) > 0:
            print(f"\n  [{split_name}] n={len(sub)}  mean={sub.mean():.0f} W  "
                  f"median={sub.median():.0f} W  range={sub.min():.0f}–{sub.max():.0f} W")

    # Summary bar chart — all athletes sorted by power, colored by split
    split_colors = {"train": "steelblue", "val": "mediumseagreen", "test": "darkorange"}

    ids_sorted  = sorted(all_results.keys(),
                         key=lambda a: all_results[a]["estimated_max_power_w"] or 0)
    powers_plot = [all_results[a]["estimated_max_power_w"] or 0 for a in ids_sorted]
    colors      = [split_colors.get(athlete_sets[a], "lightgray") for a in ids_sorted]

    fig2, ax2 = plt.subplots(figsize=(18, 6))
    ax2.bar(range(len(ids_sorted)), powers_plot, color=colors)
    ax2.set_xticks([])
    ax2.set_ylabel("Estimated Max Sustainable Power (W)", fontsize=11)
    ax2.set_title(f"Predicted Maximum Sustainable Power — All {len(ids_sorted)} Athletes", fontsize=12)
    ax2.grid(True, axis="y", alpha=0.3)

    if len(valid) > 0:
        ax2.axhline(valid.mean(), color="crimson", linestyle="--", linewidth=1.5,
                    label=f"Mean = {valid.mean():.0f} W")

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=s) for s, c in split_colors.items()]
    legend_handles.append(plt.Line2D([0], [0], color="crimson", linestyle="--", label=f"Mean = {valid.mean():.0f} W"))
    ax2.legend(handles=legend_handles, fontsize=10)

    plt.tight_layout()
    summary_plot_save = PLOT_SAVE.replace(".png", "_summary.png")
    plt.savefig(summary_plot_save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Summary plot saved: {summary_plot_save}")


if __name__ == "__main__":
    main()
