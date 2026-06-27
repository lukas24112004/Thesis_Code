# TRIMP Fatigue Signal — Investigation Log

## Why a separate log
Both TSS and TRIMP are used to compute ATL/CTL/TSB and fed into separate model variants.
Results will be compared to determine which fatigue signal better predicts HR response to power.
See `TSS_LOG.md` for the TSS pipeline and `IMPROVED_MODEL_LOG.md` for the definitive model.

---

## What is TRIMP

TRIMP (Training Impulse) measures training load from **heart rate**, not power. Developed by
Banister (1991) — the same paper the professor cites. It is the original signal for which the
ATL/CTL/TSB model was designed. TSS was developed later as a power-meter equivalent.

The core idea: the harder the cardiovascular system works, the more fatigue accumulates.
HR intensity is weighted exponentially because the relationship between HR and blood lactate
is exponential, not linear.

**Banister formula:**
```
TRIMP = Σ( dt_min × HRr × 0.64 × e^(b × HRr) )
HRr   = (HR - HR_rest) / (HR_max - HR_rest)
b     = 1.92 (male), 1.67 (female)
```

---

## TSS vs TRIMP — key comparison

|                          | TSS                        | TRIMP                          |
|--------------------------|----------------------------|--------------------------------|
| Based on                 | Power (external load)      | Heart rate (internal load)     |
| Needs                    | FTP + power meter          | HR monitor only                |
| Original Banister model  | No                         | Yes                            |
| Directly tied to HR      | No (cross-domain)          | Yes (same domain)              |

**TSS advantage:** cross-domain signal — mechanical load explaining physiological response.
Cleaner scientific claim for "does fatigue affect HR response to power?"

**TRIMP advantage:** original theoretical framework, directly measures cardiovascular stress
which is what the model predicts.

---

## Why trimp_points and not trimp_zonal_points

Four TRIMP fields are stored per ride in the GoldenCheetah JSON:

| Field                | Description                                          |
|----------------------|------------------------------------------------------|
| `trimp_points`       | Banister original — uses HR_rest and HR_max          |
| `trimp_zonal_points` | Edwards zonal — weights time in each HR zone         |
| `trimp_100_points`   | Normalized version scaled to HR_max = 100            |
| `atiss_score`        | GoldenCheetah's own HR-based load score              |

**Selected: `trimp_points`** (Banister original) for two reasons:
1. **Theoretical:** the Banister formula uses continuous exponential weighting of HR intensity,
   matching the actual physiology (HR-to-lactate relationship is exponential, not step-wise).
   The professor cited Banister 1991 directly — this field maps to that formula.
2. **Parameter transparency:** `trimp_points` only requires HR_rest and HR_max, both of which
   are known and verifiable. The zonal version depends on zone boundaries configured per athlete
   in GoldenCheetah — if those settings are wrong, the error is silent and undetectable.

**Coverage comparison (full 318-athlete dataset):**
- `trimp_points`: 82.1% coverage (22,281 missing)
- `trimp_zonal_points`: 90.3% coverage (12,020 missing)

The lower coverage of `trimp_points` is a known tradeoff — accepted in exchange for theoretical
correctness. `trimp_zonal_points` has better coverage but less principled weighting.

---

## Verification of stored trimp_points

Before trusting pre-computed values, TRIMP was independently recomputed from raw CSV data
and compared to stored `trimp_points`.

**HR_max choice matters enormously:**
- Using session max HR as HR_max → r = 0.87, MAE = 35 (broken for easy rides)
- Using all-time HR_max (max across ALL sessions) → r = 0.9987, MAE = 6

Session max HR fails on easy rides where the athlete never pushes hard — a 120 bpm session max
makes every HR sample look like near-maximum effort, exploding TRIMP. All-time HR_max is correct.

**HR_rest assumptions (with all-time HR_max):**
| HR_rest | Median error | MAE  | r      |
|---------|-------------|------|--------|
| 40 bpm  | +6.10       | 6.08 | 0.9987 |
| 50 bpm  | -2.05       | 4.38 | 0.9986 |
| 60 bpm  | -11.51      | 6.80 | 0.9986 |

**HR_rest = 50 bpm gives median error of -2 points — essentially perfect.**

Resting HR cannot be recovered from ride data: the minimum HR during a session reflects
warm-up/cool-down intensity, not true resting HR. Using session minimums gives ~71 bpm
(too high) and underestimates TRIMP. Fixed 50 bpm is more accurate for this population.

**Conclusion:** stored `trimp_points` values are accurate and can be used directly.
No need to recompute from CSVs.

---

## Steps Completed

### Step 1 — Coverage check (`check_coverage.py`)
**Purpose:** Establish baseline coverage of `trimp_points` and `coggan_tss` on the full dataset.

**Key findings (Definitive_Dataset.zip, 318 athletes, 124,153 rides):**
- `coggan_tss`: 83.3% coverage (20,689 null), ~85.5% with fallback fields
- `trimp_points`: **82.1% coverage** (22,281 null)
- `trimp_zonal_points`: 90.3% coverage — higher but wrong field for this project

---

### Step 2 — Dataset reduction (`trimp reduction/trimp_dataset_reduction.py`)
**Purpose:** Filter 318 athletes down to a clean TRIMP cohort using principled quality cuts.

**Starting point:** Definitive_Dataset.zip — 318 athletes, 124,153 rides

**Cuts applied (athlete excluded at first failing check):**
1. **HR_MAX_HIGH:** all-time HR_max > 210 bpm — spike artifact corrupts HRr for every ride,
   causing systematic TRIMP underestimation across the athlete's entire history
2. **HR_MAX_LOW:** all-time HR_max < 140 bpm — athlete never reached a real maximum,
   HRr is inflated and TRIMP is overestimated
3. **MISSING_TRIMP:** >5% of rides have null `trimp_points`

**Results:**

| Reason          | Athletes removed |
|-----------------|-----------------|
| HR_MAX_HIGH     | 112             |
| HR_MAX_LOW      | 0               |
| MISSING_TRIMP   | 61              |
| **Kept**        | **145**         |

The HR_MAX_HIGH cut removed 112 athletes (35% of the full dataset) — spike artifacts above
210 bpm are widespread in this dataset, not an edge case.

**Output:** `TRIMP_Reduced_Dataset.zip` — 145 athletes

---

### Step 3 — Bounds filter (`trimp reduction/trimp_bounds_filter.py`)
**Purpose:** Remove athletes with physiologically impossible TRIMP/hour values.

**Thresholds:**
- `TRIMP_HIGH`: TRIMP/hour > 200 — at sustained max effort (HRr = 1.0 for a full hour),
  TRIMP/hour ≈ 262 is the theoretical ceiling. 200/hour flags any ride above what is
  realistically sustainable, indicating an HR artifact that slipped through the HR_max cap.
- `TRIMP_LOW`: TRIMP/hour < 25 — corresponds to average HRr ≈ 0.20, barely above resting.
  Real cycling rides exceed this; values below indicate sensor dropout or bad HR recording.

Both cuts use the >5% threshold: athlete excluded if more than 5% of their rides fail.

**Results (applied to TRIMP_Reduced_Dataset.zip, 145 athletes):**

| Reason      | Athletes removed |
|-------------|-----------------|
| TRIMP_LOW   | 23              |
| TRIMP_HIGH  | 4               |
| **Kept**    | **118**         |

**Output:** `TRIMP_Bounds_Filtered_Dataset.zip` — 118 athletes

---

### Step 4 — ATL/CTL/TSB computation (`4_fatigue/compute_trimp_atl_ctl_tsb.py`)
**Purpose:** Compute daily TRIMP-based ATL/CTL/TSB for each clean athlete.

**Method:**
- Uses pre-stored `trimp_points` directly from JSON
- Rides with missing trimp_points: imputed with athlete's mean TRIMP (same logic as TSS pipeline)
- ATL tau = 7 days, CTL tau = 42 days (same as TSS pipeline)
- Output values are **pre-ride** (previous day's end-of-day state)
- Gap days: TRIMP = 0, EMA decays naturally
- Multiple rides on same day: all receive same previous-day values

**Output:** `trimp_atl_ctl_tsb.csv` — columns: athlete_id, date, trimp, imputed, atl_pre, ctl_pre, tsb_pre

---

### Step 5 — Baseline LSTM on TRIMP population (`5_model/baseline_trimp_lstm.py`)
**Purpose:** Establish a fair baseline for the 118 TRIMP athletes before evaluating fatigue features.
Required because TRIMP and TSS datasets contain different athletes — direct comparison without
a population-matched baseline would be misleading.

**Result: Best test RMSE = 16.06 bpm**

The TRIMP athlete population is inherently harder to predict than the TSS population
(16.06 vs 15.18 bpm baseline). This confirms the two datasets cannot be compared directly.

---

### Step 6 — TRIMP Fatigue LSTM (`5_model/trimp_fatigue_lstm.py`)
**Purpose:** Same LSTM architecture as TSS fatigue model, with TRIMP-based ATL/CTL/TSB as inputs.

**Input:** [power, atl_pre, ctl_pre, tsb_pre] (4 features, downsampled every 10s)
**Architecture:** Single-layer LSTM, hidden=32, dropout=0.2 — identical to TSS model
**Dataset:** TRIMP_Bounds_Filtered_Dataset.zip (118 athletes)
**Split:** 70% train / 30% test by athlete

**Result: Best test RMSE = 16.26 bpm**

TRIMP fatigue features did not improve HR prediction — the model performed slightly worse
than its own baseline (16.26 vs 16.06 bpm, +0.20 bpm in the wrong direction).

---

### Step 7 — TRIMP Experiment on TSS Dataset (`8_experiment_trimp/`)
**Purpose:** Rule out the possibility that TRIMP's poor result was caused by a different
athlete population. Runs TRIMP ATL/CTL/TSB and the same LSTM architecture on the identical
94 athletes used for the baseline and TSS model.

**Method:**
- Computes TRIMP ATL/CTL/TSB from `Dataset_Reduced.zip` (94 TSS athletes)
- Missing trimp_points imputed with athlete mean (same rule as all other pipelines)
- Identical LSTM architecture and hyperparameters
- Same random seed and train/test split as all other models

**Result: Best test RMSE = 15.22 bpm** (vs baseline 15.18 bpm — no improvement)

TRIMP features add nothing even on the same athletes where TSS features gave -0.97 bpm.
The conclusion is definitive: the failure of TRIMP is about the signal, not the population.

---

## Final Results

| Model                        | Dataset            | Best test RMSE | vs baseline     |
|------------------------------|--------------------|---------------|-----------------|
| Baseline (power only)        | 94 TSS athletes    | 15.18 bpm     | —               |
| TSS LSTM                     | 94 TSS athletes    | **14.21 bpm** | **-0.97 bpm (-6.4%)** |
| TRIMP LSTM (experiment)      | 94 TSS athletes    | 15.22 bpm     | +0.04 bpm (no improvement) |
| Baseline (TRIMP population)  | 118 TRIMP athletes | 16.06 bpm     | —               |
| TRIMP LSTM                   | 118 TRIMP athletes | 16.26 bpm     | +0.20 bpm (no improvement) |

## Interpretation

**TSS-based fatigue features work; TRIMP-based features do not** — at least not with this
architecture and dataset size.

This is a counterintuitive result: TRIMP is the *same-domain* signal (HR-based, predicting HR),
while TSS is a *cross-domain* signal (power-based, predicting HR). One would expect TRIMP to
carry more relevant information. Three reasons explain why the opposite was found:

**1. TRIMP is redundant with what the LSTM already learns**

The LSTM has access to within-ride HR through its hidden state — it sees HR evolving
timestep by timestep and builds an internal representation of how that athlete's heart responds
to power. Adding TRIMP-based ATL/CTL/TSB essentially says "this athlete's HR has been high in
recent weeks" — information the model can already partially infer from the ride itself.

TSS does not have this problem. It measures power history, which the model has no other way
of knowing. That is genuinely new information.

**2. TSS is better normalized across athletes**

TSS = 100 means exactly one hour at FTP for every athlete — it is anchored to each athlete's
individual threshold. So ATL = 60 TSS carries roughly the same meaning across the whole dataset.
TRIMP has no such anchor. A TRIMP of 100 means something different for an athlete with
HR_max = 170 bpm vs one with HR_max = 200 bpm, even at the same relative effort. The global
normalization in the LSTM cannot fully compensate for this between-athlete inconsistency.

**3. TSS measures the cause; TRIMP measures the effect**

The model predicts HR (an effect) from power (a cause). TSS captures the history of mechanical
load — how hard the athlete has been pushing their legs — which directly shapes how the
cardiovascular system responds to the next power input. TRIMP captures the history of
cardiovascular stress, which is itself a HR-derived quantity. Predicting HR using past HR
patterns is circular; predicting HR using past power patterns is not.

**Summary:** TSS adds genuinely new cross-domain information about mechanical load history.
TRIMP adds information the LSTM can already partially see within the ride, making it redundant
at best and noisy at worst.

The TSS result (+0.97 bpm, -6.4%) remains the primary finding of this project.
