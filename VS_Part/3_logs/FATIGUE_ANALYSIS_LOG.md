# Fatigue Direction Analysis — Attempts & Results

## Goal
After confirming that ATL/CTL/TSB improve HR prediction accuracy (H1 confirmed, 10.57 bpm),
the next question is whether the model actually learned the correct causal direction:
**does higher fatigue (negative TSB) predict higher HR at the same power?**

Expected: higher TSB (more rested) → lower predicted HR at same power (negative coefficient)
Found: consistently positive coefficients — model has it backwards

Each trial below documents the approach, results, and why it failed.

---

## Trial 1 — Synthetic TSB Sweep (`14_fatigue_experiments/trial1_synthetic_tsb_sweep.py`)

### Method
- Re-adapt each test athlete's latent vector on their first 30% of rides
- Use each athlete's most typical real ride as the power template
- Keep CTL fixed at athlete's median, sweep TSB from -40 to +20 (ATL = CTL - TSB)
- Record mean predicted HR over second half of ride at each TSB value
- Plot predicted HR vs TSB across all 15 test athletes

### Results (mean across test athletes)
| TSB   | Mean predicted HR |
|-------|------------------|
| -40   | 136.6 bpm        |
| -20   | 138.9 bpm        |
| 0     | 140.5 bpm        |
| +20   | 141.9 bpm        |

**Total HR shift: +5.3 bpm from TSB=-40 to TSB=+20 — wrong direction**

### Why it failed
Sweeping ATL/CTL/TSB synthetically compares different athlete types against each other.
Athletes with high ATL (fatigued, negative TSB) in this dataset tend to be the more trained
athletes — they train frequently, keeping ATL consistently high. Fit athletes have lower HR
at the same power. So the model learned "high ATL = lower HR" not because fatigue reduces HR,
but because high-ATL athletes are inherently fitter.

---

## Trial 2 — Within-Athlete Real Rides Analysis (`14_fatigue_experiments/trial2_within_athlete_real_rides.py`)

### Method
Stays within one athlete at a time to remove the between-athlete fitness confound:
1. For each test athlete: re-adapt latent vector on first 30% of rides
2. For each eval ride: run model, record mean predicted HR (second half), mean power, TSB value
3. Residualize predicted HR on mean power (removes "harder ride = higher HR" effect)
4. Regress power-adjusted HR residuals on TSB → slope = fatigue effect in bpm per TSB unit

### Results
| Athlete | TSB coef | r      | p-value | n rides | Direction |
|---------|----------|--------|---------|---------|-----------|
| 1       | +0.1443  | 0.349  | 0.000   | 801     | WRONG     |
| 2       | +0.1109  | 0.323  | 0.020   | 52      | WRONG     |
| 3       | +0.0837  | 0.298  | 0.000   | 361     | WRONG     |
| 4       | -0.0130  | -0.027 | 0.825   | 69      | correct   |
| 5       | -0.0464  | -0.059 | 0.354   | 249     | correct   |
| 6       | +0.0963  | 0.204  | 0.000   | 696     | WRONG     |
| 7       | -0.0092  | -0.021 | 0.786   | 174     | correct   |
| 8       | +0.1496  | 0.284  | 0.000   | 159     | WRONG     |
| 9       | +0.0424  | 0.238  | 0.089   | 52      | WRONG     |
| 10      | +0.1538  | 0.724  | 0.000   | 126     | WRONG     |
| 11      | +0.0367  | 0.138  | 0.229   | 78      | WRONG     |
| 12      | +0.2051  | 0.481  | 0.001   | 41      | WRONG     |
| 13      | +0.0582  | 0.156  | 0.057   | 150     | WRONG     |
| 14      | +0.1685  | 0.410  | 0.000   | 70      | WRONG     |
| 15      | +0.3187  | 0.728  | 0.000   | 93      | WRONG     |

**Mean TSB coefficient: +0.10 bpm/unit | Correct direction: 3/15 (20%)**

### Why it failed
Even within one athlete, the model still shows the wrong direction. The latent vector could
not fully absorb all the fitness information in only 5 adaptation epochs on 30% of rides.
ATL/CTL/TSB were still being used as partial fitness proxies even within an athlete's timeline.

---

## Trial 3 — TSB-Only Ablation (`10_tsb_only/` — folder deleted after analysis)

### Hypothesis
If CTL is causing the problem (fitter athletes have higher CTL and lower HR), removing
ATL/CTL and keeping only TSB should force the model to use TSB as a fatigue signal.

### Architecture
INPUT_SIZE=2: [power, tsb] only — no ATL/CTL

### Results
**Test RMSE: 10.68 bpm** (vs 10.57 bpm full model — only 0.11 bpm worse)

Fatigue direction: **2/15 correct (13%)** — worse than the full model (3/15)

| Metric | Full model | TSB-only |
|--------|------------|----------|
| Correct direction | 3/15 (20%) | 2/15 (13%) |
| Mean TSB coef | +0.10 bpm/unit | +0.05 bpm/unit |
| Effect of +40 TSB swing | +4.00 bpm | +2.11 bpm |

### Why it failed
Removing ATL/CTL made the direction slightly worse. TSB alone carries just as much
between-athlete confound as the full set. The model learned the same fitness-proxy
pattern from TSB alone.

**Key insight:** accuracy is nearly identical with or without ATL/CTL (10.57 vs 10.68 bpm),
yet ATL/CTL made the direction problem more pronounced. This confirms ATL/CTL were used
primarily as fitness proxies — they improved accuracy slightly by differentiating athlete
types, but reinforced the wrong-direction TSB signal.

---

## Trial 4 — Within-Athlete Standardization (`14_fatigue_experiments/trial3_standardized_model.py` + `trial3_standardized_direction_check.py`)

### Hypothesis
The root cause is that raw CTL values differ enormously between athletes (CTL=6 vs CTL=100).
If we standardize ATL/CTL/TSB within each athlete before training, the between-athlete
fitness signal is mathematically removed. The latent vector must carry all fitness information
and TSB can only explain within-athlete fatigue variation.

### The fix
```
ATL_std = (ATL - athlete_mean_ATL) / athlete_std_ATL
CTL_std = (CTL - athlete_mean_CTL) / athlete_std_CTL
TSB_std = (TSB - athlete_mean_TSB) / athlete_std_TSB
```
- For training athletes: stats from all their rides
- For val/test athletes: stats from adaptation rides only (no data leakage)

### Training results (10 epochs, Colab GPU)
| Epoch | Train RMSE | Val RMSE  |
|-------|-----------|-----------|
| 5     | 12.71 bpm | **11.54 bpm** ← best |
| 10    | 11.08 bpm | 12.29 bpm |

**Final test RMSE: 11.20 bpm** (vs 10.57 bpm definitive model — 0.63 bpm worse)

### Fatigue direction results
| Athlete | TSB coef   | r      | p-value | n rides | Direction |
|---------|------------|--------|---------|---------|-----------|
| 1       | +0.4436    | 0.779  | 0.000   | 801     | WRONG     |
| 2       | -0.0227    | -0.021 | 0.881   | 52      | correct   |
| 3       | +0.0937    | 0.459  | 0.000   | 361     | WRONG     |
| 4       | +0.0001    | 0.000  | 0.998   | 69      | WRONG     |
| 5       | +0.1968    | 0.178  | 0.005   | 249     | WRONG     |
| 6       | +0.1074    | 0.449  | 0.000   | 696     | WRONG     |
| 7       | -0.1656    | -0.363 | 0.000   | 174     | correct   |
| 8       | +0.3121    | 0.550  | 0.000   | 159     | WRONG     |
| 9       | -0.1663    | -0.418 | 0.002   | 52      | correct   |
| 10      | +0.0705    | 0.554  | 0.000   | 126     | WRONG     |
| 11      | -0.1448    | -0.286 | 0.011   | 78      | correct   |
| 12      | +0.9354    | 0.753  | 0.000   | 41      | WRONG     |
| 13      | +0.1599    | 0.372  | 0.000   | 150     | WRONG     |
| 14      | +0.4251    | 0.604  | 0.000   | 70      | WRONG     |
| 15      | +0.3530    | 0.709  | 0.000   | 93      | WRONG     |

**Mean TSB coefficient: +0.1732 bpm/unit | Correct direction: 4/15 (27%)**

### Why it still failed
Marginal improvement (3/15 → 4/15) but the direction is still wrong for 11/15 athletes.
Within-athlete standardization removed the static between-athlete fitness confound but the
problem persists within each athlete's own timeline. The exact mechanism is unclear —
a ride variability analysis (folder 21) confirmed that fresh athletes do not do
systematically more variable rides, so the confound is not simply explained by effort type.
The most likely remaining explanation is that within each athlete's training season, periods
of high ATL/negative TSB coincide with periods of increasing fitness, so the model learns
"lower TSB = fitter athlete = lower HR" even within one athlete's timeline.

---

## Overall Summary

| Trial | Approach | Correct direction | Mean coef |
|-------|----------|------------------|-----------|
| 1 | Synthetic TSB sweep | — | +0.13 bpm/unit |
| 2 | Within-athlete real rides | 3/15 (20%) | +0.10 bpm/unit |
| 3 | TSB-only model | 2/15 (13%) | +0.05 bpm/unit |
| 4 | Within-athlete standardization | 4/15 (27%) | +0.17 bpm/unit |
| 5 | Simple model (no latent vector) | 1/15 (7%) | +0.14 bpm/unit |

---

## Trial 5 — Simple Model Without Latent Vector (`15_simple_model_experiment/`)

### Hypothesis
The definitive model's latent vector absorbed so much of the between-athlete fitness
signal that TSB lost its influence. A simpler model without personalization would force
TSB to carry more predictive weight, potentially learning the correct direction.

### Architecture
Plain LSTM, hidden=32, no athlete embedding — identical to the preliminary TSS model
but with a proper 70/15/15 train/val/test split matching the definitive model.

### Direction results
| Athlete | TSB coef | r | p-value | n rides | Direction |
|---------|----------|---|---------|---------|-----------|
| 1 | +0.0828 | 0.159 | 0.000 | 801 | WRONG |
| 2 | +0.2898 | 0.652 | 0.000 | 52 | WRONG |
| 3 | +0.0457 | 0.220 | 0.000 | 361 | WRONG |
| 4 | +0.1777 | 0.472 | 0.000 | 69 | WRONG |
| 5 | +0.2489 | 0.270 | 0.000 | 249 | WRONG |
| 6 | +0.0344 | 0.120 | 0.002 | 696 | WRONG |
| 7 | -0.0838 | -0.142 | 0.061 | 174 | correct |
| 8 | +0.2271 | 0.478 | 0.000 | 159 | WRONG |
| 9 | +0.0417 | 0.114 | 0.420 | 52 | WRONG |
| 10 | +0.0551 | 0.295 | 0.001 | 126 | WRONG |
| 11 | +0.0630 | 0.191 | 0.094 | 78 | WRONG |
| 12 | +0.3069 | 0.512 | 0.001 | 41 | WRONG |
| 13 | +0.1535 | 0.355 | 0.000 | 150 | WRONG |
| 14 | +0.1215 | 0.206 | 0.087 | 70 | WRONG |
| 15 | +0.2811 | 0.576 | 0.000 | 93 | WRONG |

**Correct direction: 1/15 (7%) — worse than the definitive model (3/15)**

### Conclusion
Removing the latent vector made the direction worse, not better. This definitively rules
out model complexity as the cause of the wrong direction.

---

## Final Interpretation

### What the model learned
The RMSE improvements are real (15.18 → 14.21 bpm from fatigue features alone; 14.21 → 10.57 bpm
from adding the latent vector). But all five analyses consistently show the model did not learn
the causal fatigue mechanism. It learned ATL/CTL/TSB as an **athlete fitness characterization
signal** — a proxy for how trained an athlete generally is — rather than a day-to-day fatigue
indicator.

### Why the direction is wrong

**1. The between-athlete fitness confound**
Athletes with high ATL in this dataset tend to be fitter — they train frequently, keeping
ATL consistently high. Fit athletes have lower HR at any given power. The model learned
"high ATL = lower HR" as a fitness proxy, not as a fatigue signal.

**2. The within-athlete temporal confound**
Even within a single athlete's timeline, periods of high training load and negative TSB
tend to coincide with periods of improving fitness. The model picks up this seasonal
pattern: the athlete has lower HR during their heavy training block in summer not because
they are fresh, but because they are fitter than they were in spring.

**3. ATL/CTL/TSB was designed for training load management, not HR prediction**
TSB captures statistical regularity in training volume, not a clean physiological signal.
The relationship between TSB and HR at a given power is mediated by sleep, temperature,
hydration, and motivation — all invisible to the model.

**Note on the ride variability hypothesis:** An earlier explanation suggested fresh athletes
choose more variable ride types (intervals vs steady), inflating HR on high-TSB days. A
direct data analysis (folder 21) found no significant relationship between TSB and ride
variability index across the 15 test athletes (slope ≈ 0, p=0.30). This explanation was
not supported and should not be cited.

### What the RMSE improvement actually reflects
The fatigue features alone gave a modest gain (15.18 → 14.21 bpm, -0.97 bpm). The large
improvement came from the latent vector (14.21 → 10.57 bpm, -3.64 bpm), which captures
each athlete's individual HR-power characteristics. The model uses ATL/CTL/TSB primarily
to differentiate athlete types at population level, not as a day-to-day fatigue signal.
This is real and useful for prediction purposes, but not evidence that causal fatigue was learned.

### Scientific conclusion
- **H1 confirmed:** fatigue features improve HR prediction (15.18 → 14.21 bpm, -0.97 bpm); latent vector drives the larger gain (14.21 → 10.57 bpm, -3.64 bpm)
- **Causal direction not learned:** best result 4/15 correct across all five trials
- **Model complexity ruled out:** simple model (1/15) performed worse than definitive (3/15)
- **Ride variability hypothesis ruled out:** data analysis shows no TSB-variability relationship
- **Root cause:** TSB functions as a fitness proxy in this dataset, not a day-to-day fatigue signal
- **Future work:** HRV, resting HR, sleep quality needed to capture true fatigue effects
