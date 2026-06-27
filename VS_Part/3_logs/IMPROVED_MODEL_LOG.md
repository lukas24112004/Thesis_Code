# Improved Model — Athlete Personalization & Definitive Results

## Goal
Improve on the preliminary TSS LSTM (14.21 bpm) by adding athlete-specific personalization
via a learned latent vector, fix the validation methodology flagged by the supervisor, and
produce a clean, unbiased test RMSE.

---

## Step 10 — Improved TSS LSTM with Athlete Personalization (`8_improved_model/improved_tss_lstm.py`)

### Architecture changes over preliminary TSS LSTM

| Component         | Preliminary        | Improved                     |
|-------------------|--------------------|------------------------------|
| Hidden size       | 32                 | 64                           |
| Epochs            | 10                 | 30                           |
| Personalization   | None               | Athlete latent vector (dim=8)|
| Input features    | 4                  | 4 (same)                     |

### Athlete latent vector
Each of the 65 train athletes has a learned 8-dimensional vector. This vector is projected
via two linear layers (`h_proj`, `c_proj`) into the LSTM's initial hidden and cell states,
so the model starts each ride already "knowing" the athlete's physiology.

The model decides what each of the 8 dimensions encodes — it could implicitly capture
fitness level, resting HR, HR-power sensitivity, cardiac efficiency, etc. It is not
hand-specified.

**Why 8 dimensions?** Small enough to prevent overfitting (8 × 65 = 520 numbers total),
large enough to capture meaningful individual variation.

### Test-athlete adaptation
Test athletes have no pre-learned embedding. Their latent vector is initialized as the mean
of all 65 train-athlete vectors, then fine-tuned for 5 epochs on 30% of their rides
(LSTM weights frozen). The remaining 70% of their rides are used for evaluation.

### Sequence-to-sequence vs sliding window
Both preliminary and improved models process the full ride as one sequence (seq2seq).
The group project used sliding windows predicting one step ahead. Our approach captures
longer-range temporal dependencies (cardiac drift over a full session) without needing
explicit lagged power features — the LSTM hidden state accumulates power history implicitly.

### Training
Run on Google Colab (T4 GPU). Saved to Google Drive.

### Results (preliminary run — test set used as validation — not fully valid)
| Epoch | Train RMSE | Test RMSE |
|-------|-----------|-----------|
| 1     | 17.70 bpm | 14.11 bpm |
| 5     | 13.45 bpm | 10.91 bpm |
| 10    | 11.73 bpm | 10.46 bpm |
| 28    | 10.49 bpm | **10.08 bpm** ← best |
| 30    | 10.36 bpm | 10.19 bpm |

**Note:** best epoch selected based on test set — test set acting as validation, making
the reported RMSE slightly optimistic. Supervisor flagged this. Fixed in Step 11.

---

## Step 11 — Definitive Model with Proper Train/Val/Test Split (`8_improved_model/validation_definitive_model.py`)

### Validation fix
**Split:** 70% train (65 athletes) / 15% validation (14 athletes) / 15% test (15 athletes)
- Best epoch selected on **validation set only**
- Test set evaluated **once** at the very end on the best saved model
- Gives a clean, unbiased test RMSE

### Results
| Epoch | Train RMSE | Val RMSE  |
|-------|-----------|-----------|
| 1     | 17.70 bpm | 14.15 bpm |
| 5     | 13.63 bpm | 12.18 bpm |
| 9     | 11.87 bpm | 10.91 bpm |
| 15    | 11.15 bpm | 10.65 bpm |
| 28    | 10.61 bpm | **10.63 bpm** ← best val |
| 30    | 10.31 bpm | 10.78 bpm |

**Best validation RMSE: 10.63 bpm** (epoch 28)
**Final test RMSE: 10.57 bpm** (evaluated once on held-out test set)

The test RMSE (10.57) is very close to the validation RMSE (10.63), confirming the model
generalizes well and is not overfit.

---

## Results Summary

| Model | Test RMSE | vs Baseline | vs Preliminary TSS |
|-------|-----------|-------------|-------------------|
| Baseline (power only) | 15.18 bpm | — | — |
| Preliminary TSS LSTM | 14.21 bpm | -0.97 bpm (-6.4%) | — |
| Improved + personalization | ~10.08 bpm* | — | — |
| **Definitive model (valid)** | **10.57 bpm** | **-4.61 bpm (-30%)** | **-3.64 bpm (-26%)** |

*preliminary run, not fully valid

**Improvement over group project model: 12.58 → 10.57 bpm (-2.01 bpm, -16%)**

The improvement from 14.21 → 10.57 bpm primarily comes from the athlete latent vector —
personalization allows the model to start each ride already knowing the athlete's individual
HR-power characteristics, rather than learning from scratch.
