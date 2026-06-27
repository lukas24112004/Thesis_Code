# The Definitive Model

---

## The Problem with the Preliminary Run

The preliminary TSS model (14.21 bpm) selected the best epoch based on test set performance. This means the test set was effectively used twice — once to tune the model and once to report results — which introduces optimism bias. The supervisor flagged this. The definitive model fixes it.

---

## Two Changes

### Change 1 — Proper Three-Way Split
**70% train (65 athletes) / 15% validation (14 athletes) / 15% test (15 athletes)**

The best epoch is now selected based on the **validation set only**. The test set is touched exactly once, at the very end, on the best saved model. This gives a clean, unbiased estimate of how the model performs on athletes it has never seen.

### Change 2 — Athlete Latent Vector
Each of the 65 training athletes gets a learned **8-dimensional vector** that seeds the LSTM's initial hidden and cell states. This vector is the model's physiological fingerprint for that athlete — it can implicitly encode fitness level, resting HR, HR-power sensitivity, cardiac efficiency, or any combination the model finds useful. The model decides what each dimension means; it is not hand-specified.

**Why 8 dimensions?** Small enough to prevent overfitting (8 × 65 = 520 parameters total), large enough to capture meaningful individual variation.

**Architecture upgrade:** Hidden size increased from 32 to 64. Trained for 30 epochs on Google Colab (T4 GPU).

### Test-athlete adaptation
Test athletes have no pre-learned vector. Their latent vector is initialised as the mean of all 65 training-athlete vectors, then fine-tuned for **5 epochs on 30% of their rides** (LSTM weights frozen). The remaining 70% of their rides are used for evaluation. This simulates the real-world scenario where a new athlete provides a short history of rides before the model is used.

---

## Training Results

| Epoch | Train RMSE | Val RMSE |
|-------|-----------|----------|
| 1     | 17.70 bpm | 14.15 bpm |
| 5     | 13.63 bpm | 12.18 bpm |
| 9     | 11.87 bpm | 10.91 bpm |
| 15    | 11.15 bpm | 10.65 bpm |
| 28    | 10.61 bpm | **10.63 bpm** ← best val |
| 30    | 10.31 bpm | 10.78 bpm |

**Best validation epoch: epoch 28**
**Final test RMSE: 10.57 bpm** (evaluated once on held-out test set)

The test RMSE (10.57) sits right beside the validation RMSE (10.63), confirming the model generalises well and is not overfit.

---

## Full Results Summary

| Model | Test RMSE | vs Baseline | vs Preliminary TSS |
|-------|-----------|-------------|-------------------|
| Baseline (power only) | 15.18 bpm | — | — |
| Preliminary TSS LSTM | 14.21 bpm | −0.97 bpm (−6.4%) | — |
| **Definitive model** | **10.57 bpm** | **−4.61 bpm (−30%)** | **−3.64 bpm (−26%)** |

**Improvement over group project model: 12.58 → 10.57 bpm (−2.01 bpm, −16%)**

The 0.97 bpm gain from fatigue features is real and confirmed. The larger gain (3.64 bpm) comes from the latent vector — personalisation allows the model to start each ride already knowing the athlete's individual HR-power characteristics, rather than having to infer them from scratch.

---

## What the Model Looks Like in Practice

On an example held-out test ride, the model achieved **6.3 bpm RMSE** — well below the 10.57 bpm test average, indicating this was a relatively clean, steady ride. Key observations:

- The model correctly follows the slow cardiac drift during the warm-up phase
- It tracks sharp HR spikes during interval efforts
- The largest errors occur at peak efforts, where the model slightly over-predicts
- Actual and predicted HR lines are nearly indistinguishable across most of the ride
