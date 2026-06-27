"""
Ride Type Analysis — Does TSB Predict Ride Variability?
---------------------------------------------------------
This script tests the hypothesis used to explain the wrong fatigue direction:
"On high-TSB (fresh) days, athletes choose more variable/intense rides
(intervals, hard efforts), while on low-TSB (fatigued) days they choose
easier, steadier rides."

If true, we expect:
  - Higher TSB → higher normalized power relative to average power
  - Higher TSB → higher power variability (coefficient of variation)
  - Higher TSB → higher average power overall

For each of the 15 test athletes:
  1. Load all their rides with TSB values
  2. For each ride compute:
     - Mean power (active timesteps only, power > 50W)
     - Normalized power (NP = rolling 30s 4th-root mean power^4)
     - Variability index = NP / mean power (1.0 = perfectly steady, higher = more variable)
     - Coefficient of variation of power = std(power) / mean(power)
  3. Bin rides into TSB groups: < -20, -20 to 0, 0 to +20, > +20
  4. Plot mean variability index and mean power per TSB bin
  5. Run linear regression: TSB vs variability index (within each athlete)

If the hypothesis is correct: positive slope (higher TSB → more variable rides).
If the hypothesis is wrong: no relationship or negative slope.

Run on Colab. Uses the same dataset and split as all other scripts.
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
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_PATH     = "/content/drive/MyDrive/Research_Project/Dataset_Reduced.zip"
ATL_CTL_TSB_PATH = "/content/drive/MyDrive/Research_Project/tss_atl_ctl_tsb.csv"
SAVE_SCATTER     = "/content/drive/MyDrive/Research_Project/ride_type_scatter.png"
SAVE_BINS        = "/content/drive/MyDrive/Research_Project/ride_type_bins.png"
SAVE_CSV         = "/content/drive/MyDrive/Research_Project/ride_type_results.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED         = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
DOWNSAMPLE   = 10
MIN_STEPS    = 60
MAX_STEPS    = 300
NP_WINDOW    = 3
MIN_ACTIVE_W = 50    # minimum power to count as "riding"

TSB_BINS     = [(-999, -20, "< -20\n(very fatigued)"),
                (-20,    0, "-20 to 0\n(fatigued)"),
                (  0,   20, "0 to +20\n(rested)"),
                ( 20,  999, "> +20\n(very rested)")]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_np_rolling(power, window=3):
    T      = len(power)
    p      = np.maximum(power, 0.0)
    np_arr = np.empty(T, dtype=np.float32)
    for t in range(T):
        start     = max(0, t - window + 1)
        chunk     = p[start : t + 1]
        np_arr[t] = float(np.mean(chunk ** 4) ** 0.25)
    return np_arr


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
                        df    = df.iloc[:MAX_STEPS]
                        power = df["power"].values.astype(np.float32)
                        rides.append((power, tsb))
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


def ride_metrics(power, min_active=MIN_ACTIVE_W):
    active = power[power > min_active]
    if len(active) < 10:
        return None
    mean_p = float(active.mean())
    np_seq = compute_np_rolling(power, NP_WINDOW)
    np_val = float(np_seq[power > min_active].mean())
    vi     = np_val / mean_p if mean_p > 0 else None   # variability index
    cv     = float(active.std() / mean_p) if mean_p > 0 else None
    return {"mean_power": mean_p, "norm_power": np_val,
            "variability_index": vi, "coef_variation": cv}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print("Loading fatigue features...")
    fatigue_lookup = load_fatigue_lookup(ATL_CTL_TSB_PATH)

    print("Loading rides...")
    athlete_rides = load_all_rides(DATASET_PATH, fatigue_lookup)

    _, _, test_ids = split_athletes_three_way(
        list(athlete_rides.keys()), TRAIN_RATIO, VAL_RATIO, SEED
    )
    test_ids = [aid for aid in test_ids if aid in athlete_rides]
    print(f"  {len(test_ids)} test athletes\n")

    all_rows  = []
    reg_results = []

    for ath_idx, athlete_id in enumerate(test_ids):
        rides = athlete_rides[athlete_id]
        for power, tsb in rides:
            m = ride_metrics(power)
            if m is None:
                continue
            m["athlete_id"] = athlete_id
            m["tsb"]        = tsb
            all_rows.append(m)

    df = pd.DataFrame(all_rows)
    print(f"Total rides analysed: {len(df)}\n")

    # --- Per-athlete regression: TSB vs variability index ---
    print("Per-athlete regression (TSB → variability index):")
    print(f"  {'Athlete':<5} {'slope':>8} {'r':>6} {'p':>6} {'n':>5} {'direction'}")
    print(f"  {'-'*50}")

    for ath_idx, athlete_id in enumerate(test_ids):
        sub = df[df["athlete_id"] == athlete_id].dropna(subset=["variability_index"])
        if len(sub) < 5:
            continue
        slope, intercept, r, p, _ = scipy_stats.linregress(sub["tsb"], sub["variability_index"])
        direction = "positive ✓" if slope > 0 else "negative ✗"
        print(f"  {ath_idx+1:<5} {slope:>+8.5f} {r:>6.3f} {p:>6.3f} {len(sub):>5}  [{direction}]")
        reg_results.append({"athlete": ath_idx+1, "slope": round(slope,5),
                            "r": round(r,3), "p": round(p,3), "n": len(sub),
                            "direction": "positive" if slope > 0 else "negative"})

    n_pos  = sum(1 for r in reg_results if r["direction"] == "positive")
    slopes = [r["slope"] for r in reg_results]
    print(f"\n  Positive slope (hypothesis supported): {n_pos}/{len(reg_results)}")
    print(f"  Mean slope: {np.mean(slopes):+.5f}")

    # --- Bin plot: mean variability index and mean power by TSB bin ---
    bin_labels, bin_vi, bin_pow = [], [], []
    for lo, hi, label in TSB_BINS:
        subset = df[(df["tsb"] > lo) & (df["tsb"] <= hi)]
        if len(subset) > 0:
            bin_labels.append(f"{label}\n(n={len(subset)})")
            bin_vi.append(subset["variability_index"].mean())
            bin_pow.append(subset["mean_power"].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(len(bin_labels)), bin_vi, color="#1f77b4", edgecolor="none")
    ax1.set_xticks(range(len(bin_labels)))
    ax1.set_xticklabels(bin_labels, fontsize=9)
    ax1.set_ylabel("Mean Variability Index (NP / mean power)", fontsize=11)
    ax1.set_title("Ride Variability by TSB Bin\n"
                  "Higher = more interval-type effort", fontsize=11)
    ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="VI = 1.0 (perfectly steady)")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(range(len(bin_labels)), bin_pow, color="#ff7f0e", edgecolor="none")
    ax2.set_xticks(range(len(bin_labels)))
    ax2.set_xticklabels(bin_labels, fontsize=9)
    ax2.set_ylabel("Mean Active Power (W)", fontsize=11)
    ax2.set_title("Average Power by TSB Bin\n"
                  "Do fresh athletes ride harder?", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Ride Type Analysis — 15 Test Athletes\n"
                 "Testing whether high TSB days have more variable / harder rides",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_BINS, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved: {SAVE_BINS}")

    # --- Scatter: TSB vs variability index (all rides, all athletes) ---
    fig2, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df["tsb"], df["variability_index"], alpha=0.15, s=8,
               color="steelblue", edgecolors="none")
    slope_all, intercept_all, r_all, p_all, _ = scipy_stats.linregress(
        df["tsb"].dropna(), df["variability_index"].dropna()
    )
    x_line = np.linspace(df["tsb"].min(), df["tsb"].max(), 200)
    ax.plot(x_line, slope_all * x_line + intercept_all, color="crimson",
            linewidth=2, label=f"slope={slope_all:+.5f}  r={r_all:.3f}  p={p_all:.4f}")
    ax.set_xlabel("TSB", fontsize=12)
    ax.set_ylabel("Variability Index (NP / mean power)", fontsize=12)
    ax.set_title("TSB vs Ride Variability — All Test Athletes\n"
                 "Positive slope = fresh days have more variable rides (hypothesis supported)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_SCATTER, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {SAVE_SCATTER}")

    df.to_csv(SAVE_CSV, index=False)
    print(f"Data saved: {SAVE_CSV}")

    print(f"\n--- Summary ---")
    print(f"Hypothesis: higher TSB → more variable ride (positive slope)")
    print(f"Overall slope (all rides): {slope_all:+.5f}  r={r_all:.3f}  p={p_all:.4f}")
    print(f"Per-athlete: {n_pos}/{len(reg_results)} show positive slope")
    if slope_all > 0 and p_all < 0.05:
        print("→ Hypothesis SUPPORTED: fresh athletes do ride with more variability")
    elif slope_all > 0:
        print("→ Trend in expected direction but not statistically significant")
    else:
        print("→ Hypothesis NOT supported: no clear relationship between TSB and ride variability")


if __name__ == "__main__":
    main()
