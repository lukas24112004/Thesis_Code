"""
Thesis Visualisation — Three sections
--------------------------------------
Section 1 — HR Prediction
  Plot 1a: Actual vs predicted HR for 6 test rides (one per athlete)
  Plot 1b: Scatter plot of all predicted vs actual HR values across test set

Section 2 — Maximum Sustainable Power
  Plot 2:  Bar chart of estimated max power for all 94 athletes (reuses max_performance.csv)

Section 3 — Fatigue Direction Analysis
  Plot 3:  Within-athlete TSB vs HR residual scatter for 4 example athletes

Run on Google Colab. All outputs saved to Drive.
"""

import io
import random
import zipfile
from collections import defaultdict
from datetime import date, timedelta

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ---------------------------------------------------------------------------
# Paths — Colab
# ---------------------------------------------------------------------------

DATASET_PATH      = "/content/drive/MyDrive/Research_Project/Dataset_Reduced.zip"
ATL_CTL_TSB_PATH  = "/content/drive/MyDrive/Research_Project/tss_atl_ctl_tsb.csv"
MODEL_PATH        = "/content/drive/MyDrive/Research_Project/definitive_model.pt"
MAX_PERF_CSV      = "/content/drive/MyDrive/Research_Project/max_performance.csv"

SAVE_HR_RIDE      = "/content/drive/MyDrive/Research_Project/thesis_plot1a_hr_ride.png"
SAVE_HR_SCATTER   = "/content/drive/MyDrive/Research_Project/thesis_plot1b_hr_scatter.png"
SAVE_MAX_POWER    = "/content/drive/MyDrive/Research_Project/thesis_plot2_max_power.png"
SAVE_FATIGUE_DIR  = "/content/drive/MyDrive/Research_Project/thesis_plot3a_fatigue_direction.png"
SAVE_FATIGUE_SWEEP= "/content/drive/MyDrive/Research_Project/thesis_plot3b_fatigue_sweep.png"

# ---------------------------------------------------------------------------
# Config — must match training script exactly
# ---------------------------------------------------------------------------

SEED         = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
DOWNSAMPLE   = 10
MIN_STEPS    = 60
MAX_STEPS    = 600
LATENT_DIM   = 8
HIDDEN_SIZE  = 64
DROPOUT      = 0.2
INPUT_SIZE   = 4
ADAPT_RATIO  = 0.30
ADAPT_EPOCHS = 5
ADAPT_LR     = 0.01

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


def predict_ride(model, latent, features_norm, stats):
    feat_t = torch.from_numpy(features_norm).unsqueeze(0).to(DEVICE)
    length = torch.tensor([len(features_norm)], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pred_norm = model(feat_t, length, latent.unsqueeze(0))
    return denormalize_hr(pred_norm.squeeze(0).cpu().numpy(), stats)

# ---------------------------------------------------------------------------
# Section 1a — Actual vs Predicted HR for 1 representative ride
# ---------------------------------------------------------------------------

def plot_hr_ride(model, mean_latent, test_ids, athlete_rides, stats, save_path):
    """
    Pick the test athlete whose per-ride RMSE is closest to the overall mean,
    then show their median-length eval ride as a single clean plot.
    """
    # First pass: compute per-athlete mean RMSE on eval rides
    athlete_rmses = {}
    athlete_latents = {}
    for athlete_id in test_ids:
        rides      = athlete_rides[athlete_id]
        rides_norm = normalize_rides(rides, stats)
        n_adapt    = max(1, int(len(rides_norm) * ADAPT_RATIO))
        latent     = adapt_latent(model, mean_latent, rides_norm[:n_adapt])
        athlete_latents[athlete_id] = latent
        rmses = []
        for features_norm, hr_norm in rides_norm[n_adapt:]:
            actual = denormalize_hr(hr_norm, stats)
            pred   = predict_ride(model, latent, features_norm, stats)
            rmses.append(float(np.sqrt(np.mean((pred - actual) ** 2))))
        athlete_rmses[athlete_id] = np.mean(rmses)

    overall_mean = np.mean(list(athlete_rmses.values()))
    # Pick athlete closest to overall mean RMSE
    chosen_id = min(test_ids, key=lambda a: abs(athlete_rmses[a] - overall_mean))
    latent     = athlete_latents[chosen_id]

    rides      = athlete_rides[chosen_id]
    rides_norm = normalize_rides(rides, stats)
    n_adapt    = max(1, int(len(rides_norm) * ADAPT_RATIO))
    eval_set   = rides_norm[n_adapt:]

    # Pick the longest eval ride for a richer plot
    eval_lens  = [len(r[1]) for r in eval_set]
    pick_idx   = int(np.argmax(eval_lens))

    features_norm, hr_norm = eval_set[pick_idx]
    actual_hr = denormalize_hr(hr_norm, stats)
    pred_hr   = predict_ride(model, latent, features_norm, stats)
    time_min  = np.arange(len(actual_hr)) * 10 / 60
    rmse      = float(np.sqrt(np.mean((pred_hr - actual_hr) ** 2)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_min, actual_hr, color="#d62728", linewidth=1.8, label="Actual HR")
    ax.plot(time_min, pred_hr,   color="#1f77b4", linewidth=1.8, label="Predicted HR",
            linestyle="--", alpha=0.85)

    ax.set_ylim(0, max(actual_hr.max(), pred_hr.max()) * 1.15)
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Heart Rate (bpm)", fontsize=12)
    ax.set_title(f"Actual vs Predicted Heart Rate — Example Test Ride\nRMSE = {rmse:.1f} bpm",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

# ---------------------------------------------------------------------------
# Section 1b — Scatter: all predicted vs actual HR (test set)
# ---------------------------------------------------------------------------

def plot_hr_scatter(model, mean_latent, test_ids, athlete_rides, stats, save_path):
    all_actual = []
    all_pred   = []

    for athlete_id in test_ids:
        rides      = athlete_rides[athlete_id]
        rides_norm = normalize_rides(rides, stats)
        n_adapt    = max(1, int(len(rides_norm) * ADAPT_RATIO))
        latent     = adapt_latent(model, mean_latent, rides_norm[:n_adapt])

        for features_norm, hr_norm in rides_norm[n_adapt:]:
            actual = denormalize_hr(hr_norm, stats)
            pred   = predict_ride(model, latent, features_norm, stats)
            all_actual.extend(actual.tolist())
            all_pred.extend(pred.tolist())

    all_actual = np.array(all_actual)
    all_pred   = np.array(all_pred)
    rmse       = float(np.sqrt(np.mean((all_pred - all_actual) ** 2)))

    fig, ax = plt.subplots(figsize=(7, 7))

    # Density scatter using 2D histogram for speed
    h = ax.hist2d(all_actual, all_pred, bins=80, cmap="Blues", cmin=1)
    fig.colorbar(h[3], ax=ax, label="Sample count")

    # Perfect prediction line
    lim = [60, 220]
    ax.plot(lim, lim, color="crimson", linewidth=1.5, linestyle="--", label="Perfect prediction")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual Heart Rate (bpm)", fontsize=12)
    ax.set_ylabel("Predicted Heart Rate (bpm)", fontsize=12)
    ax.set_title(f"Predicted vs Actual HR — Test Set\nRMSE = {rmse:.2f} bpm", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

# ---------------------------------------------------------------------------
# Section 2 — Max sustainable power bar chart
# ---------------------------------------------------------------------------

def plot_max_power(csv_path, save_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values("estimated_max_power_w").reset_index(drop=True)

    split_colors = {"train": "steelblue", "val": "mediumseagreen", "test": "darkorange"}
    colors = [split_colors.get(s, "lightgray") for s in df["split"]]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(df)), df["estimated_max_power_w"], color=colors, edgecolor="none")

    mean_w = df["estimated_max_power_w"].mean()
    ax.axhline(mean_w, color="crimson", linestyle="--", linewidth=1.8,
               label=f"Mean = {mean_w:.0f} W")

    ax.set_xticks([])
    ax.set_xlabel("Athletes (sorted by estimated power)", fontsize=11)
    ax.set_ylabel("Estimated Max Sustainable Power (W)", fontsize=11)
    ax.set_title("Estimated Maximum Sustainable Power per Athlete\n"
                 "(power at which predicted HR crosses 90% of observed max HR)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=s) for s, c in split_colors.items()]
    handles.append(plt.Line2D([0], [0], color="crimson", linestyle="--",
                              label=f"Mean = {mean_w:.0f} W"))
    ax.legend(handles=handles, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

# ---------------------------------------------------------------------------
# Section 3 — Fatigue direction: TSB vs HR residual (4 athletes)
# ---------------------------------------------------------------------------

def plot_fatigue_direction(model, mean_latent, test_ids, athlete_rides, stats,
                           fatigue_df, save_path):
    """
    For 4 test athletes: scatter TSB vs power-residualized HR.
    Shows the direction the model learned (mostly positive = wrong direction).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for ax_idx, athlete_id in enumerate(test_ids[:4]):
        rides      = athlete_rides[athlete_id]
        rides_norm = normalize_rides(rides, stats)
        n_adapt    = max(1, int(len(rides_norm) * ADAPT_RATIO))
        latent     = adapt_latent(model, mean_latent, rides_norm[:n_adapt])

        ath_fatigue = fatigue_df[fatigue_df["athlete_id"] == athlete_id]

        tsb_vals, mean_power_vals, mean_hr_pred_vals = [], [], []

        for ride_idx, (features_norm, hr_norm) in enumerate(rides_norm[n_adapt:]):
            pred_hr = predict_ride(model, latent, features_norm, stats)
            # Only use second half of ride (steady-state HR)
            half    = len(pred_hr) // 2
            mean_pred_hr  = pred_hr[half:].mean()
            raw_features  = rides[n_adapt + ride_idx][0]
            mean_power    = raw_features[:, 0].mean()
            tsb           = raw_features[0, 3]  # TSB is constant within a ride

            tsb_vals.append(tsb)
            mean_power_vals.append(mean_power)
            mean_hr_pred_vals.append(mean_pred_hr)

        tsb_vals       = np.array(tsb_vals)
        power_vals     = np.array(mean_power_vals)
        hr_vals        = np.array(mean_hr_pred_vals)

        # Residualize HR on power (remove "harder ride = higher HR" effect)
        if len(power_vals) > 2:
            coeffs    = np.polyfit(power_vals, hr_vals, 1)
            hr_resid  = hr_vals - np.polyval(coeffs, power_vals)
        else:
            hr_resid = hr_vals

        # Regression line for TSB → HR residual
        ax = axes_flat[ax_idx]
        ax.scatter(tsb_vals, hr_resid, alpha=0.5, s=18, color="steelblue", edgecolors="none")

        if len(tsb_vals) > 2:
            m, b   = np.polyfit(tsb_vals, hr_resid, 1)
            x_line = np.linspace(tsb_vals.min(), tsb_vals.max(), 100)
            color  = "crimson" if m > 0 else "green"
            direction = "wrong direction ↑" if m > 0 else "correct direction ↓"
            ax.plot(x_line, m * x_line + b, color=color, linewidth=2,
                    label=f"slope = {m:.3f} ({direction})")

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"Athlete {ax_idx + 1}  (n={len(tsb_vals)} rides)", fontsize=11)
        ax.set_xlabel("TSB (Training Stress Balance)", fontsize=9)
        ax.set_ylabel("Power-residualized predicted HR (bpm)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fatigue Direction Analysis — TSB vs Power-Residualized Predicted HR\n"
                 "Expected: negative slope (higher TSB → lower HR). "
                 "Red = wrong direction learned.", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

# ---------------------------------------------------------------------------
# Section 3b — Synthetic TSB sweep: general predicted HR vs TSB across all test athletes
# ---------------------------------------------------------------------------

def plot_fatigue_sweep(model, mean_latent, test_ids, athlete_rides, stats,
                       fatigue_df, save_path):
    """
    For each test athlete: fix power at their median ride power, fix CTL at their
    median CTL, sweep TSB from -40 to +20. Record mean predicted HR (second half of
    a 10-minute synthetic ride) at each TSB value.

    Plots each athlete as a faint gray line, the mean across all athletes in bold.
    Shows the direction the model learned globally.
    """
    tsb_values  = list(range(-40, 25, 5))
    all_curves  = []

    for athlete_id in test_ids:
        rides      = athlete_rides[athlete_id]
        rides_norm = normalize_rides(rides, stats)
        n_adapt    = max(1, int(len(rides_norm) * ADAPT_RATIO))
        latent     = adapt_latent(model, mean_latent, rides_norm[:n_adapt])

        ath_df      = fatigue_df[fatigue_df["athlete_id"] == athlete_id]
        median_ctl  = float(ath_df["ctl_pre"].median()) if len(ath_df) > 0 else 60.0
        median_power = float(np.mean([r[0][:, 0].mean() for r in rides]))

        # Normalize median power using feat_mean/std (power is feature index 0)
        feat_mean = stats["feat_mean"]
        feat_std  = stats["feat_std"]

        curve = []
        for tsb in tsb_values:
            atl = median_ctl - tsb   # ATL = CTL - TSB
            T   = 60                 # 10-minute synthetic ride at 10s resolution

            features = np.stack([
                np.full(T, median_power, dtype=np.float32),
                np.full(T, atl,          dtype=np.float32),
                np.full(T, median_ctl,   dtype=np.float32),
                np.full(T, float(tsb),   dtype=np.float32),
            ], axis=1)

            features_norm = (features - feat_mean) / (feat_std + 1e-8)
            pred_hr       = predict_ride(model, latent, features_norm, stats)
            curve.append(pred_hr[T // 2:].mean())   # second half = steady state

        all_curves.append(curve)

    all_curves = np.array(all_curves)   # shape (n_athletes, n_tsb_values)
    mean_curve = all_curves.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, curve in enumerate(all_curves):
        ax.plot(tsb_values, curve, color="gray", linewidth=1.0, alpha=0.35)

    ax.plot(tsb_values, mean_curve, color="#d62728", linewidth=2.5,
            label=f"Mean across {len(test_ids)} athletes")

    ax.axvline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xlabel("TSB (Training Stress Balance)", fontsize=12)
    ax.set_ylabel("Predicted Heart Rate (bpm)", fontsize=12)
    ax.set_title("Synthetic TSB Sweep — Predicted HR vs Fatigue State\n"
                 "Power and CTL held at each athlete's median. "
                 "Expected: higher TSB → lower HR.", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate the total shift only
    delta = mean_curve[-1] - mean_curve[0]
    ax.annotate(f"Total shift: {delta:+.1f} bpm",
                xy=(0.97, 0.05), xycoords="axes fraction",
                ha="right", fontsize=10,
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


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

    # --- Section 1a: Single HR ride plot ---
    print("Generating Plot 1a: actual vs predicted HR (single ride)...")
    plot_hr_ride(model, mean_latent, test_ids, athlete_rides, stats, SAVE_HR_RIDE)

    # --- Section 1b: HR scatter ---
    print("Generating Plot 1b: predicted vs actual HR scatter...")
    plot_hr_scatter(model, mean_latent, test_ids, athlete_rides, stats, SAVE_HR_SCATTER)

    # --- Section 2: Max power ---
    print("Generating Plot 2: max sustainable power...")
    plot_max_power(MAX_PERF_CSV, SAVE_MAX_POWER)

    # --- Section 3a: Per-athlete fatigue direction ---
    print("Generating Plot 3a: per-athlete fatigue direction...")
    plot_fatigue_direction(model, mean_latent, test_ids, athlete_rides, stats,
                           fatigue_df, SAVE_FATIGUE_DIR)

    # --- Section 3b: Synthetic TSB sweep ---
    print("Generating Plot 3b: synthetic TSB sweep...")
    plot_fatigue_sweep(model, mean_latent, test_ids, athlete_rides, stats,
                       fatigue_df, SAVE_FATIGUE_SWEEP)

    print("\nAll plots saved to Drive.")


if __name__ == "__main__":
    main()
