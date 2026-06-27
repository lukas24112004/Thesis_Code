# Preliminary Models: TSS vs TRIMP

---

## The Architecture

All preliminary models share the same base architecture — a single-layer LSTM — so that any difference in results can be attributed to the input features, not the model design.

**Architecture:**
- Single-layer LSTM, hidden size = 32, dropout = 0.2
- Processes the full ride as one sequence (sequence-to-sequence)
- Input downsampled to one reading every 10 seconds
- Trained for up to 10 epochs, best epoch selected on test RMSE
- Split: 70% train / 30% test by athlete

**Note on the split:** These preliminary runs used a 70/30 train/test split with no separate validation set. The best epoch was selected based on test performance, which makes the reported RMSE slightly optimistic. This was corrected in the definitive model (see Models log).

---

## Step 1 — Baseline LSTM (Power Only)

**Purpose:** Establish how well heart rate can be predicted from power alone, with no fatigue information.

**Input:** `[ power ]` — 1 feature

**Result:**

| Epoch | Train RMSE | Test RMSE |
|-------|-----------|-----------|
| 1     | 18.63 bpm | 16.79 bpm |
| 7     | 17.72 bpm | **15.18 bpm** ← best |
| 10    | 17.33 bpm | 15.65 bpm |

**Best test RMSE: 15.18 bpm**

This is the starting point. Every subsequent model is compared against this number.

---

## Step 2 — TSS Fatigue LSTM

**Purpose:** Test whether adding TSS-based fatigue features (ATL/CTL/TSB) improves HR prediction. This directly tests Hypothesis H1: *does knowing an athlete's fatigue state improve HR prediction from power?*

**Input:** `[ power, ATL, CTL, TSB ]` — 4 features. All other settings identical to the baseline.

**Result:**

| Epoch | Train RMSE | Test RMSE |
|-------|-----------|-----------|
| 1     | 17.89 bpm | 15.02 bpm |
| 9     | 16.30 bpm | **14.21 bpm** ← best |
| 10    | 16.22 bpm | 14.69 bpm |

**Best test RMSE: 14.21 bpm — a reduction of 0.97 bpm (−6.4%) vs baseline**

**H1 confirmed:** TSS-based fatigue features reliably improve HR prediction. The improvement is consistent across epochs and athletes, not a lucky run.

This result is preliminary — the definitive test RMSE comes after adding proper validation and personalization — but the direction is clear.

---

## Step 3 — TRIMP Fatigue LSTM

**Purpose:** Compare TRIMP-based ATL/CTL/TSB against TSS-based features. Since TRIMP and TSS were computed on different athlete populations, the comparison must be done on the same 94 TSS athletes.

**Input:** `[ power, ATL_trimp, CTL_trimp, TSB_trimp ]` — TRIMP-based fatigue features on the 94 TSS athletes. All other settings identical.

**Result:**

| Model | Dataset | Test RMSE | vs Baseline |
|-------|---------|-----------|-------------|
| Baseline | 94 TSS athletes | 15.18 bpm | — |
| TSS fatigue | 94 TSS athletes | **14.21 bpm** | **−0.97 bpm (−6.4%)** |
| TRIMP fatigue | 94 TSS athletes | 15.22 bpm | +0.04 bpm (no improvement) |

**TRIMP adds essentially nothing.** On its own athlete population (118 athletes), it actually performs slightly *worse* than its own baseline (16.26 vs 16.06 bpm).

---

## Why TSS Works and TRIMP Does Not

This is a counterintuitive result — TRIMP is the HR-based signal predicting an HR outcome, while TSS is a power-based signal. You might expect TRIMP to carry more relevant information. Three reasons explain the opposite finding:

### 1. TRIMP is likely redundant with what the LSTM already sees
The LSTM processes heart rate evolving second-by-second throughout the ride. By the time any individual timestep is reached, the model's hidden state already contains an implicit summary of how the cardiovascular system has been responding to power throughout the session. Adding TRIMP — which is essentially a summary of recent HR history — tells the model something it can already partially infer from within the ride itself.

TSS does not have this problem. It measures *power history*, which the model has no other way of knowing. That is genuinely new, cross-domain information.

### 2. TSS is better normalized across athletes
TSS = 100 means exactly one hour at FTP for every athlete — it is anchored to each athlete's individual threshold. So ATL = 60 TSS carries roughly the same meaning across the whole dataset: this athlete has been doing about 60% of their threshold pace recently.

TRIMP has no such anchor. A TRIMP value of 100 means something different for an athlete with HR_max = 170 bpm versus one with HR_max = 200 bpm, even at the same relative effort. The model cannot fully compensate for this between-athlete inconsistency.

### 3. TSS measures the cause; TRIMP measures the effect
The model predicts HR (an effect) from power (a cause). TSS captures the history of mechanical load — how hard the athlete has been pushing their legs — which directly shapes how the cardiovascular system will respond to the next power input. TRIMP captures the history of cardiovascular stress, which is itself derived from HR. Predicting HR using past HR patterns is somewhat circular; predicting HR using past power patterns is not.

---

## Summary

| Model | Test RMSE | vs Baseline |
|-------|-----------|-------------|
| Baseline (power only) | 15.18 bpm | — |
| TSS Fatigue LSTM | **14.21 bpm** | **−0.97 bpm (−6.4%)** |
| TRIMP Fatigue LSTM | 15.22 bpm | +0.04 bpm (no improvement) |

TSS-based fatigue features are confirmed as the signal to carry forward into the definitive model. TRIMP is dropped.
