# Maximum Performance Prediction

---

## The Idea

The definitive model answers: *given this power output, what heart rate will this athlete reach?*

The extension flips the question: *given this athlete's HR ceiling, what is the maximum power they can sustain before hitting it?*

This turns the model into a practical tool — instead of predicting HR from power, it estimates an upper performance limit from the athlete's observed physiology.

---

## Method

For each of the 15 test athletes:

1. **Adapt the latent vector** — fine-tune the athlete's 8-dimensional fingerprint on 30% of their rides (5 epochs, LSTM weights frozen), same procedure as the main model evaluation.

2. **Set an HR ceiling** — defined as 90% of the athlete's 99th percentile HR across all their rides. The 99th percentile is used instead of the absolute maximum to remove sensor spike artifacts (some athletes have readings of 595 bpm which would produce an unreachable ceiling). 90% of near-maximum corresponds roughly to a very hard threshold effort — sustainable for a trained athlete but close to their physiological limit.

3. **Run a synthetic power ramp** — steady-state power steps from 50W to 700W in 10W increments. Each step is held for 10 minutes to allow HR to stabilise.

4. **Average HR over the last 5 minutes** of each step — this gives the steady-state predicted HR at that power level, filtering out any transient response at the start of the step.

5. **Find the crossing point** — the power level at which predicted HR first crosses the athlete's HR ceiling, found by linear interpolation between the last step below and first step above the threshold.

---

## Results

**All 15 test athletes reached the HR threshold — no missing values.**

| Group | Mean estimated max power | Range |
|-------|------------------------|-------|
| Test athletes (15) | **175 W** | ~120 W – 210 W |
| All athletes — train + val + test (94) | **198 W** | ~120 W – 360 W |

The 15 test athletes happened to sit slightly below the population average. The 198W figure is more representative of the full dataset. Train, validation, and test athletes are evenly distributed across the full power range with no systematic split-based bias, confirming the model generalises across the fitness spectrum.

---

## Interpretation

**175–198W is physiologically realistic for this population.** GoldenCheetah users skew toward recreational and hobby cyclists rather than competitive racers. Trained club racers typically sit at 250–350W; values in the 120–210W range are consistent with amateur athletes training for fitness rather than competition.

**The range (120–360W across all athletes) makes sense.** A factor of ~3× between the weakest and strongest athlete in a heterogeneous recreational population is physiologically plausible.

---

## Limitations

- **This is a model extrapolation, not a measured value.** The estimate is derived from synthetic power inputs fed to a model trained on real rides — it is not a lab test or a measured FTP.
- **It inherits the full 10.57 bpm RMSE.** A small HR prediction error shifts where the ceiling is crossed, which can meaningfully change the estimated max power especially at the extremes of the power range.
- **The model was trained on power up to typical training levels.** The strongest athletes near the top of the range are extrapolating beyond the model's training distribution and are therefore the least reliable estimates.

---

## Practical Use Case

Given an athlete's training history (ATL/CTL/TSB) and a few adaptation rides, the model can estimate the power level at which their heart rate will approach its ceiling. This is useful for:
- **Pacing guidance** — knowing where the physiological limit is before a race or hard effort
- **Overtraining risk assessment** — monitoring whether max sustainable power is declining over a training block
