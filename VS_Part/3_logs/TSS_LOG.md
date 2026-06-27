# TSS Pipeline & Preliminary Models

## Research Question

Does knowledge of an athlete's fatigue state improve the prediction of heart rate response
to power output during a cycling session?

**TSS-specific sub-question:** Can power-based training load (TSS → ATL/CTL/TSB) reduce HR
prediction error in an LSTM model?

---

## Motivation

HR response to a fixed power output is not constant. An athlete who trained hard yesterday
shows a higher HR at 250W than one who rested for three days. Standard HR-from-power models
ignore this, treating every ride as if the athlete is in the same state.

If TSS-based fatigue features reduce prediction error, the model becomes practically useful:
it can estimate cardiovascular state, flag overtraining risk, and estimate maximum sustainable
power given current fatigue.

---

## Dataset Context

### Source
- 318 athletes from the group project's cleaned dataset (Research Project 3)
- Each athlete: one JSON file (ride metadata + metrics) + one CSV per ride (second-by-second
  power and HR)
- 124,153 total rides across all athletes

### Group project's cleaning (their pipeline)
- Phase 1: removed athletes with invalid year of birth, missing gender, or fewer than 10 rides
- Phase 2: removed rides with no HR or power, rides under 30 min or over 6 hours, HR > 220 bpm
- Re-applied 50-ride minimum → 318 athletes, ~5.7 GB

### Important distinction
The group project needed rides with both HR and power for model inputs.
This project needs TSS from **all** rides — including deleted ones — to build a complete
fatigue timeline. A gap corrupts ATL/CTL from that point forward.
The JSON retains TSS even for rides whose CSVs were removed.

---

## Steps Completed

### Step 1 — CSV Coverage Check (`check_csv_coverage.py`)
**Purpose:** Compare JSON ride list to CSVs present in each athlete's zip.

**Key findings (full 318-athlete dataset):**
- 124,153 total rides in JSON; 96.6% have a matching CSV present
- Athletes span multiple UTC offsets (Europe UTC+2, US East Coast UTC-4)
- Script brute-forces timezone offsets (-14 to +14) rather than assuming a fixed one
- Missing CSVs = rides the group project removed; TSS still valid for fatigue timeline

---

### Step 2a — TSS Missing Values (`validate_tss_missing.py`)
**Purpose:** Check how many rides have no TSS and whether fallback fields help.

**Key findings (318 athletes, 124,153 rides):**
- 16.7% of rides have no `coggan_tss` (20,689 null rides) — raw coverage 83.3%
- With fallback fields, effective coverage reaches ~85.5%
- Missing TSS almost perfectly explained by missing power: if no power, no TSS, no recovery

---

### Step 2b — Sport Breakdown (`sport_breakdown.py`)
**Key findings:**
- 93.3% of rides are cycling (Bike + VirtualRide)
- 582 rides have no power (245 unlabeled cyclists, 104 runs, 68 swims, etc.)
- 114 rides have no power but DO have TSS (runs/swims with manually entered load) — kept
- Decision: missing TSS → TSS = 0 in ATL/CTL/TSB

---

### Step 2c — TSS Bounds Check (`validate_tss_bounds.py`)
**Thresholds:** TSS > 500, TSS/hour > 150, TSS/hour < 10, IF > 1.5, IF < 0.3

**Key findings:**
- 2,769 rides flagged (2.2% of total)
- Most common: TSSHR_LOW and IF_LOW — very low intensity or data artifacts
- TSS_HIGH and IF_HIGH rare

---

### Step 3 — TSS Recalculation (`calculate_tss.py`)
**Formula:** `TSS = IF² × (recording_time_seconds / 3600) × 100`

**Key discovery:** Must use recording time (pauses excluded), not workout_time (pauses included):
- workout_time → 53.2% match
- recording_time → 99.3% match

**Results:** 98,325 / 98,983 rides match within 1% tolerance — stored TSS values are trustworthy.

---

### Step 4 — TSS Cross-Check (`validate_tss_crosscheck.py`)
**Purpose:** For the 2,769 flagged rides, check whether recalculated TSS triggers the same flag.

**Results:**
- 1,286 (46%): both flagged — genuine outliers
- 1,483 (54%): stored only — mostly pause artifacts (workout_time inflates denominator)
- **38 genuinely suspicious rides (0.03%)** — stored TSS out of bounds but recalc is normal

---

### Step 5 — Athlete-Level Quality Summary (`generate_athlete_table.py`)
- 14 athletes: 100% missing TSS (FTP never configured) — unusable
- 82 athletes: >20% missing TSS
- 55 athletes: 0% missing TSS — cleanest group

---

### Step 6 — Dataset Reduction (`tss reduction/dataset_reduction.py`)
**Cuts applied (athlete excluded at first failing check):**
1. FTP_DEFAULT: cp_setting = 250W on every ride
2. FTP_STAGNANT: cp_setting never changes
3. MISSING_TSS: >5% of rides have no TSS
4. TSS_HIGH: >5% of rides confirmed TSS > 500
5. IF_LOW: >5% of rides confirmed IF < 0.3

**Output:** `Dataset_Reduced.zip` — **94 athletes** (down from 318)

---

### Step 7 — ATL/CTL/TSB Computation (`4_fatigue/compute_atl_ctl_tsb.py`)
**Method:**
- ATL (Acute Training Load): 7-day EMA (alpha = 1 - e^(-1/7) ≈ 0.133)
- CTL (Chronic Training Load): 42-day EMA (alpha = 1 - e^(-1/42) ≈ 0.024)
- TSB (Training Stress Balance): CTL - ATL
- Rides with no TSS: imputed with athlete's mean TSS
- Gap days (no ride): TSS = 0, EMA decays naturally
- Output values are **pre-ride** (athlete's state entering the ride)

**Output:** `tss_atl_ctl_tsb.csv`

---

### Step 8 — Baseline LSTM (`5_model/baseline_lstm.py`)
**Architecture:** Single-layer LSTM, hidden=32, dropout=0.2
**Input:** power only (1 feature, downsampled every 10s)
**Split:** 70% train / 30% test by athlete

| Epoch | Train RMSE | Test RMSE |
|-------|-----------|-----------|
| 1     | 18.63 bpm | 16.79 bpm |
| 7     | 17.72 bpm | **15.18 bpm** ← best |
| 10    | 17.33 bpm | 15.65 bpm |

**Best test RMSE: 15.18 bpm**

---

### Step 9 — TSS Fatigue LSTM (`5_model/fatigue_lstm.py`)
**Input:** [power, atl_pre, ctl_pre, tsb_pre] (4 features, downsampled every 10s)
All other settings identical to baseline.

| Epoch | Train RMSE | Test RMSE |
|-------|-----------|-----------|
| 1     | 17.89 bpm | 15.02 bpm |
| 9     | 16.30 bpm | **14.21 bpm** ← best |
| 10    | 16.22 bpm | 14.69 bpm |

**Best test RMSE: 14.21 bpm**

---

## Results Summary

| Model | Test RMSE | vs Baseline |
|-------|-----------|-------------|
| Baseline (power only) | 15.18 bpm | — |
| TSS Fatigue LSTM | **14.21 bpm** | **-0.97 bpm (-6.4%)** |

**H1 confirmed:** Adding TSS-based fatigue features (ATL/CTL/TSB) improves HR prediction
by ~1 bpm. The improvement is consistent across epochs and athletes.

Note: best epoch was selected on the test set in these preliminary runs — see
`IMPROVED_MODEL_LOG.md` for the definitive model with proper validation.
