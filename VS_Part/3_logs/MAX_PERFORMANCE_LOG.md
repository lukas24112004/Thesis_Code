# Max Performance Prediction

## Goal
Use the trained definitive model to estimate the maximum sustainable power for each test
athlete — defined as the power level where predicted HR approaches 90% of that athlete's
observed maximum HR.

## Script
`12_max_performance_prediction/max_performance.py`

---

## Method

For each of the 15 test athletes:
1. Adapt the latent vector (5 epochs, 30% of rides, LSTM weights frozen)
2. Compute the athlete's HR threshold: 90% of their 99th percentile HR across all rides
3. Run a synthetic ramp: steady-state power steps from 50W to 700W in 10W increments, holding each step for 10 min
4. For each power level, run the LSTM and record mean predicted HR over the last 5 min of each step
5. Max sustainable power = the power at which predicted HR first crosses the threshold

### Key configuration
```python
RAMP_END_W     = 700      # raised from 550 to ensure no athlete hits ceiling
HR_THRESHOLD   = 0.90     # 90% of 99th percentile HR
HR_PERCENTILE  = 99       # 99th percentile (not absolute max — removes sensor spikes)
```

---

## Fixes applied over original script

| Issue | Original | Fixed |
|-------|----------|-------|
| HR max definition | `max(all_hr)` — absolute max including spikes | `np.percentile(all_hr, 99)` — removes spike artifacts (e.g., 595 bpm readings) |
| Ramp ceiling | 550 W — Athletes 6 (500W) and 13 (488W) hit it | 700 W — no athlete reached ceiling |

The first version produced NaN results for two athletes and corrupted thresholds for others
(athlete with 595 bpm spike had an unreachable 535 bpm threshold).

---

## Results (Colab, definitive model — 10.57 bpm)

**All athletes reached the HR threshold. No NaN values.**

### Test set only (15 athletes)
| Metric | Value |
|--------|-------|
| Mean estimated max power | **175 W** |
| Range | ~120 W – 210 W |

### All 94 athletes (train + val + test)
| Metric | Value |
|--------|-------|
| Mean estimated max power | **198 W** |
| Range | ~120 W – 360 W |

The 15 test athletes happened to sit slightly below average — 198W is the more representative
figure for this population. Train, val, and test athletes are evenly distributed across the
full power range with no systematic split-based bias, confirming the model generalizes well
across the fitness spectrum.

---

## Interpretation

**The 175W mean** is on the lower end for cyclists but consistent with a recreational/amateur
population. GoldenCheetah users skew toward hobby athletes rather than competitive racers
(trained racers would typically be 250–350W).

**The range (120–210W)** is physiologically sensible — a factor of ~1.75× between the weakest
and strongest test athlete is realistic for a heterogeneous population.

**Important caveat:** This is not a true VO2max or FTP estimate. The value is derived entirely
from the model's HR predictions, which carry the full 10.57 bpm RMSE uncertainty. The estimate
reflects "at what power does this model predict this athlete's HR will approach their ceiling"
given their observed HR profile from adaptation rides.

**Practical use case:** Given an athlete's training history (ATL/CTL/TSB) and a sensor reading
HR during a ride, the model can estimate current HR at any power level. The max performance
function extends this to estimate the upper sustainable limit — useful for pacing guidance or
overtraining risk assessment.

---

## Extension — Optimal Fatigue State for Maximum Performance

### Script
`14_fatigue_experiments/extension_optimal_fatigue_performance.py`

### Goal
Connect the max performance and fatigue sections: sweep TSB from -40 to +20 (CTL fixed at
each athlete's median) and find the TSB at which estimated max power is highest.

### Method
For each of the 15 test athletes:
1. Fix CTL at their median value
2. Sweep TSB from -40 to +20 in steps of 5
3. At each TSB value, compute ATL = CTL - TSB and run the full power ramp
4. Record estimated max power at each TSB
5. Optimal TSB = TSB value where estimated max power is highest

### Results

**Line plot:** Mean estimated max power decreases from ~188W at TSB=-40 to ~174W at TSB=+20.
The model predicts higher max power when the athlete is more fatigued (negative TSB).

**Bar chart:** 11/14 athletes show optimal TSB at -40 or -35 (most fatigued end of sweep).
Only 3 athletes show optimal TSB in positive territory. Mean optimal TSB = **-25**.

### Interpretation

The result is physiologically backwards — nobody performs best when heavily fatigued.
This is a direct consequence of the same confound identified in the fatigue direction analysis:
the model learned high ATL as a signal for a fitter athlete (lower HR at any power), not as
a fatigue signal. Artificially increasing ATL (lowering TSB) makes the model treat the athlete
as fitter, which lowers predicted HR and pushes the estimated max power ceiling higher.

This confirms the fatigue direction finding from a different angle: the model uses
ATL/CTL/TSB as athlete fitness characterization, not day-to-day fatigue. The optimal fatigue
state result is therefore a model artifact, not a physiological prediction.

The inconsistency is worth noting in the thesis as a limitation and as further evidence
that more direct physiological signals (HRV, resting HR) would be needed to capture
true fatigue effects.

---

## Simple Model Comparison (`16_optimal_fatigue_simple/optimal_fatigue_simple.py`)

### Purpose
Test whether the backwards optimal fatigue result was caused by the latent vector
absorbing the fitness signal. The simple model has no latent vector — if the result
is the same, the problem is in the data.

### Results

| Model | Mean optimal TSB | Conclusion |
|-------|-----------------|------------|
| Definitive (latent vector) | **-25.0** | most athletes at -40 |
| Simple (no latent vector) | **-32.9** | even more extreme |

Both models give negative mean optimal TSB — both say "most fatigued = highest max power."
The simple model is slightly more extreme (-32.9 vs -25.0) but the direction is identical.

### Conclusion
Model complexity is not the cause. Removing the latent vector made the result worse,
not better. This is consistent with the direction check result (1/15 correct vs 3/15).
The root cause is the data: TSB correlates with athlete fitness level, not day-to-day
fatigue, so lower TSB (higher ATL) always looks like a fitter athlete to the model.
true fatigue effects.
