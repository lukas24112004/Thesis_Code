# Did the Model Learn Why? — Fatigue Direction Analysis

---

## The Question

A lower RMSE proves the fatigue features are useful. It does not prove the model learned the correct physiological relationship. The model could be exploiting a spurious correlation that happens to reduce error without understanding the underlying mechanism.

The correct direction is clear from physiology:
- **Higher TSB (more rested)** → heart works efficiently → **lower HR at the same power**
- **Lower TSB (more fatigued)** → heart works harder → **higher HR at the same power**

If the model learned this correctly, regressing TSB against predicted HR (holding power fixed) should give a **negative coefficient**. A positive coefficient means the model learned the relationship backwards.

Five trials tested this from different angles.

---

## Trial 1 — Optimal Fatigue State for Maximum Performance

**Method:** Using the max performance extension, TSB was swept from −40 (deeply fatigued) to +20 (very fresh) for each test athlete, with CTL fixed at each athlete's median. At each TSB value, ATL = CTL − TSB was set and the full power ramp was run. The TSB at which estimated max power was highest is the state the model considers optimal.

**If the model understood fatigue correctly:** A fresher athlete (higher TSB) should hold more power before HR hits its ceiling. Optimal TSB should be positive.

**Result:**

- 10 of 14 athletes show optimal TSB in the deeply fatigued zone (negative TSB)
- Mean optimal TSB = **−25**
- As TSB rises from −40 to +20, estimated max power *falls* from ~188W to ~174W

**The model says athletes perform best when heavily fatigued — physiologically backwards.**

---

## Trial 2 — Within-Athlete Real Rides

**Method:** To remove the confound that fitter athletes simply train more (and therefore have higher ATL), the test was run within one athlete at a time.

1. Adapt the athlete's latent vector on their first 30% of rides
2. For each remaining ride, record predicted HR, mean power, and pre-ride TSB
3. Residualise predicted HR on mean power (removes the "harder ride = higher HR" effect)
4. Regress those power-adjusted residuals on TSB — the slope is the fatigue direction signal

A **negative slope** = correct (fresh → lower HR). A **positive slope** = backwards.

**Result across 15 test athletes:**

| Metric | Value |
|--------|-------|
| Athletes with correct direction | **3 / 15** |
| Mean TSB coefficient | **+0.10 bpm per TSB unit** |

The wrong direction persists even within a single athlete's own ride history.

---

## Sub-test A — TSB-Only Model (Is CTL the Problem?)

**Hypothesis:** CTL is a 42-day average — it directly encodes how fit an athlete is. Maybe CTL is the source of the fitness-proxy confound, and removing it will force TSB to act as a pure fatigue signal.

**Method:** Retrain the model with only `[ power, TSB ]` as input — ATL and CTL removed.

**Result:**
- Correct direction: **2 / 15** — got *worse*, not better
- Mean TSB coefficient: +0.05 bpm per unit

TSB alone carries just as much fitness-proxy information as the full feature set. Removing CTL does not fix the direction.

---

## Sub-test B — No Latent Vector (Did Personalisation Hide It?)

**Hypothesis:** Maybe the athlete latent vector absorbed all the fitness information, leaving TSB with no role except as a spurious signal. A plain LSTM with no personalisation should force TSB to carry more predictive weight.

**Method:** Plain LSTM, hidden size = 32, no athlete embedding — trained on the same 94 athletes with the same split.

**Result:**
- Correct direction: **1 / 15** — got *worse* again
- Mean TSB coefficient: +0.14 bpm per unit

Removing personalisation made the direction problem worse, not better. Model complexity is not the cause.

---

## Trial 3 — Within-Athlete Standardisation

**Hypothesis:** Raw ATL/CTL/TSB values differ enormously between athletes (CTL = 6 for a casual rider vs CTL = 100 for a serious one). The model might be reading the larger number as a fitness signal rather than a fatigue signal. Rescaling each value to a z-score against that athlete's own history removes the between-athlete scale difference.

**Method:**
```
ATL_std = (ATL − athlete_mean_ATL) / athlete_std_ATL
CTL_std = (CTL − athlete_mean_CTL) / athlete_std_CTL
TSB_std = (TSB − athlete_mean_TSB) / athlete_std_TSB
```
For test athletes, statistics are computed from adaptation rides only (no data leakage).

**Result:**
- Correct direction: **4 / 15** — marginal improvement, still 11 athletes wrong
- Mean TSB coefficient: +0.17 bpm per unit (slightly larger in wrong direction)
- Test RMSE: 11.20 bpm (0.63 bpm worse than the definitive model)

Standardisation is the best result across all five trials, but the direction remains wrong for the large majority of athletes.

---

## Summary Table

| Trial | Approach | Correct Direction |
|-------|----------|-----------------|
| 1 | Optimal-fatigue power ramp | 3 / 15 |
| 2 | Within-athlete real rides | 3 / 15 |
| 2a | TSB-only model | 2 / 15 |
| 2b | No latent vector | 1 / 15 |
| 3 | Within-athlete standardisation | **4 / 15** |

**Best case: 4/15 — still 11 athletes wrong.** The pattern holds across every method, ruling out any single architectural or feature choice as the cause.

---

## Root Cause — The Leading Theory

The most likely explanation is a structural confound in how cyclists actually train:

**Periods of high training load (negative TSB) coincide with periods of improving fitness.**

When a cyclist is in a dedicated training block — high ATL, negative TSB — they are also prioritising cycling, managing recovery, eating well, sleeping consistently, and making fitness gains. Their HR at a fixed power is *falling* during this period because they are getting fitter.

When a cyclist has a long rest or takes time off — positive TSB, low ATL — fitness erodes, habits slip, and HR at the same power *rises*.

The model sees this pattern in the data and learns it correctly from a predictive standpoint: high ATL correlates with lower HR. But it has learned it as a **fitness characterisation signal**, not a **day-to-day fatigue signal**. The model uses TSB to identify what kind of athlete this is, not to track how tired they are today.

This is not a modelling failure — it is a data structure problem. The signal needed to capture true day-to-day fatigue (HRV, resting HR, sleep quality, subjective wellness) is simply not present in the dataset.

---

## What the RMSE Improvement Actually Reflects

The fatigue features do improve accuracy — but for a different reason than originally hypothesised:

- **−0.97 bpm from fatigue features alone** (15.18 → 14.21 bpm): ATL/CTL/TSB help the model distinguish athlete types at population level — a proxy for fitness that adds information the LSTM cannot infer from power alone.
- **−3.64 bpm from the latent vector** (14.21 → 10.57 bpm): Explicit personalisation is far more powerful, directly encoding each athlete's individual HR-power characteristics.

The accuracy gain is real and useful. But it is not evidence that the model learned the causal fatigue mechanism.

---

## Future Work

### Signals that could capture true fatigue
- **HRV (Heart Rate Variability):** Morning HRV is a validated physiological marker of recovery state — a low HRV indicates the nervous system is still under stress from previous training.
- **Resting HR:** Elevated resting HR is a well-established sign of accumulated fatigue and poor recovery.
- **Sleep quality:** Depth and duration of overnight recovery directly affect next-day cardiovascular response to exercise.
- **Subjective wellness scores:** Self-reported fatigue, mood, and motivation are strong predictors of performance readiness.

### The experiment that would settle it
The fundamental problem is that field data confounds fatigue with fitness. A controlled lab experiment would isolate fatigue cleanly:

1. Same athlete, same bike, same lab
2. Fixed effort: 30 minutes at 150W, held constant by a power-controlled trainer
3. Ride it once fresh (after 3+ days rest), once fatigued (after a hard training block)
4. Days apart within the same week — so fitness cannot meaningfully change between sessions
5. Compare heart rate

Same power → different HR = fatigue, isolated. In a controlled setting like this, the model's TSB-based fatigue signal would be expected to perform markedly better than in the observational field data used here.
