# Master Presentation Document
## Fatigue-Aware Heart Rate Prediction in Cycling — Complete Research Narrative

*This document covers everything needed to build a final presentation. It is organized as a
story from beginning to end: what we did, why we did it, what we found, and what it means.
Every number, every result, and every connection between experiments is included.*

---

# PART 1 — THE PROBLEM

## What This Research Is About

When a cyclist rides at 200 watts, their heart rate response is not fixed. The same person,
at the same power level, can show very different heart rates depending on how fatigued they
are, how fit they currently are, and how far into the ride they already are.

This is a well-known physiological reality: fatigue elevates heart rate. A cyclist who has
been training heavily for two weeks will have a higher heart rate at 200 watts on a Friday
afternoon than they would after a rest week. The heart has to work harder to deliver the
same mechanical output when the body is under accumulated stress.

The research question is: **can a machine learning model learn to predict a cyclist's
heart rate from their power output, and does adding fatigue information actually improve
those predictions?**

This is not just an academic question. If the answer is yes, the model could be used in
practical applications: real-time pacing guidance (slow down, your predicted HR at this pace
is approaching your ceiling), overtraining detection (your HR is higher than the model
expects at this power — something is wrong), and individualized training load management.

## The Setup

The model takes as input a ride's second-by-second (or every-10-second) power output and
predicts what the athlete's heart rate should be at each moment. The key scientific question
across the whole project is: **does adding fatigue features — a measure of how much training
stress the athlete has accumulated — improve this prediction?**

We expect yes. A fatigued athlete (high recent training load) should show higher predicted HR
at any given power. A rested athlete should show lower predicted HR. If the model can learn
this relationship, it has genuinely captured a physiological signal.

---

# PART 2 — THE DATASET

## The Raw Dataset

The dataset comes from GoldenCheetah — open-source cycling training software that lets
athletes log their rides. The raw dataset contains **318 athletes** and **124,153 rides**.

Each ride is a CSV file with second-by-second recordings of power (watts) and heart rate
(bpm). Alongside each ride, GoldenCheetah stores pre-computed ride summaries as JSON metadata
— including things like total distance, average power, normalized power, and importantly,
**pre-computed training load scores** for each ride.

## Reducing the Dataset — The TSS Pipeline

Not all 318 athletes could be used. The primary fatigue signal we test is **TSS
(Training Stress Score)**, a power-based metric. To compute TSS, you need an FTP (Functional
Threshold Power) for each athlete — the power output they can sustain for approximately one
hour. If an athlete has no reliable FTP recorded, their TSS values are meaningless.

Additionally, we need rides that have both a power recording AND a heart rate recording.
Many athletes in GoldenCheetah only have heart rate (no power meter) or only have power
(no HR monitor). We need both.

**Five quality cuts were applied, in order:**

1. **NO_FTP** — athlete has no FTP recorded in their history. TSS cannot be computed without it.
2. **FTP_HIGH** — FTP > 500 watts. Physiologically possible for elite athletes, but unrealistic
   for the recreational/amateur GoldenCheetah population. Likely a data entry error (e.g.,
   watts entered as kilograms of body weight).
3. **FTP_LOW** — FTP < 50 watts. Clearly erroneous.
4. **LOW_PAIRED_RIDES** — fewer than 30 rides with both power AND heart rate recorded.
   Insufficient data to train or adapt a personalized model.
5. **LOW_TSS_COVERAGE** — more than 5% of rides are missing TSS values (i.e., rides where the
   power meter was not recording or FTP was missing for that session).

**Result after all 5 cuts: 94 athletes remained.** This is the dataset used for all TSS
experiments, the definitive model, and all fatigue direction analyses.

The 70/15/15 train/validation/test split by athlete (with random seed 42) gives:
- **65 train athletes** — used to fit the model
- **14 validation athletes** — used to select the best epoch during training
- **15 test athletes** — held out, never seen until the final evaluation

---

# PART 3 — THE FATIGUE SIGNAL: TSS, ATL, CTL, AND TSB

## What TSS Measures

TSS (Training Stress Score) is a number that quantifies how hard a single ride was,
relative to the athlete's fitness. The formula normalizes every ride to the same scale:

- **TSS = 100** means: you rode for exactly one hour at your FTP. That is by definition one
  "unit" of training stress.
- A two-hour ride at half your FTP would also be TSS = 100.
- A 30-minute all-out effort above FTP might give TSS = 80.

Because TSS is anchored to each athlete's individual FTP, **TSS = 60 means roughly the same
thing for a recreational rider (FTP = 150W) and a trained cyclist (FTP = 350W)**. This
cross-athlete consistency is one reason TSS works well as a machine learning feature.

## ATL, CTL, and TSB — The Fitness-Fatigue Model

The Banister fitness-fatigue model (1991) uses TSS history to estimate three quantities.
These are computed as exponentially-weighted moving averages of daily TSS:

- **ATL (Acute Training Load)** — the short-term average of TSS over approximately 7 days.
  This represents recent acute stress — how fatigued the athlete is right now. High ATL
  means the athlete has been training hard in the past week.

- **CTL (Chronic Training Load)** — the long-term average of TSS over approximately 42 days.
  This represents the athlete's fitness level — how much training load they have sustained
  over the past month and a half. High CTL means a well-trained, fit athlete.

- **TSB (Training Stress Balance)** = CTL − ATL. This is the "form" measure. Positive TSB
  means CTL > ATL — the athlete has backed off recently and is rested and fresh. Negative TSB
  means ATL > CTL — the athlete has been training hard above their baseline level and is
  carrying accumulated fatigue.

**Important:** these values are computed as **pre-ride values** — the state at the start of
each ride, before it contributes to the running totals. This prevents data leakage.

These three features — ATL, CTL, TSB — are appended to every timestep in the ride sequence.
A ride at power = [180, 195, 210, ...] watt becomes a sequence of:
`[power, ATL, CTL, TSB]` at every timestep. The fatigue state is constant throughout the
ride (it reflects the pre-ride state) while power varies second by second.

---

# PART 4 — THE TRIMP PIPELINE

## What TRIMP Is

TRIMP (Training Impulse) was developed by Banister in 1991 — the same paper that introduced
the ATL/CTL/TSB fitness-fatigue model. While TSS measures **external load** (mechanical power
output), TRIMP measures **internal load** (cardiovascular stress from heart rate).

The TRIMP formula is:

```
TRIMP per ride = Σ( dt_minutes × HRr × 0.64 × e^(1.92 × HRr) )
HRr = (HR - HR_rest) / (HR_max - HR_rest)
```

Where HRr is the fractional heart rate reserve — how hard the heart is working relative to
its range from rest to maximum. The exponential weighting (e^1.92) reflects the fact that
the HR-to-blood-lactate relationship is exponential: spending time near HR_max accumulates
fatigue much faster than spending the same time near resting HR.

**Key parameters used:**
- HR_rest = 50 bpm (fixed, because session minimum HR reflects warm-up intensity, not true
  resting HR — a detailed verification showed HR_rest = 50 bpm gives median error of only
  −2 TRIMP points vs the GoldenCheetah-stored values)
- HR_max = all-time maximum HR across all sessions (not session maximum — using session max
  breaks TRIMP for easy rides where the athlete never reaches maximum effort)
- Formula: Banister `trimp_points` from GoldenCheetah (not the zonal variant, because it
  uses continuous exponential weighting matching the actual physiology)

## Why TRIMP Was Worth Testing

TRIMP was an attractive alternative to TSS for three reasons:
1. It is the **original** signal from the Banister 1991 paper — theoretically motivated
2. It only needs a heart rate monitor (no power meter required)
3. It directly measures cardiovascular stress, which is exactly what the model predicts

## The TRIMP Dataset: 318 → 118 Athletes

TRIMP requires clean HR data for every ride. Three quality cuts were applied:

1. **HR_MAX_HIGH** — all-time HR_max > 210 bpm: clearly a sensor spike artifact. This corrupts
   HRr for every single ride that athlete has ever done, making their entire TRIMP history
   unreliable. **This alone removed 112 athletes** — 35% of the original 318. HR spike
   artifacts are extremely widespread in the dataset, not an edge case.

2. **HR_MAX_LOW** — all-time HR_max < 140 bpm: athlete apparently never reached a real
   cardiovascular maximum in any recorded session. HRr would be artificially inflated. Zero
   athletes were removed by this cut.

3. **MISSING_TRIMP** — more than 5% of rides missing `trimp_points`: data quality filter.
   61 additional athletes removed.

**After quality cuts: 145 athletes.** A second-pass bounds filter then removed athletes with
physiologically impossible TRIMP/hour values (either below 25/hour — barely above resting —
or above 200/hour — above what is sustainably possible). This left **118 athletes** in the
final TRIMP dataset.

Note that the TSS dataset (94 athletes) and the TRIMP dataset (118 athletes) **do not
contain the same athletes**. This makes direct comparison unreliable, which is why a
separate experiment tested TRIMP on the exact same 94 TSS athletes.

---

# PART 5 — THE PRELIMINARY MODELS

## Baseline Model

The simplest possible model: a single-layer LSTM with hidden size 32, taking only power as
input. No fatigue information. No athlete personalization.

This establishes the floor: what prediction accuracy is achievable from power alone, without
any knowledge of who the athlete is or how fatigued they are?

**Baseline test RMSE: 15.18 bpm** (on the 94 TSS athletes, 70/30 train/test split)

## Preliminary TSS Fatigue LSTM

Same architecture (LSTM, hidden=32) but with four input features: [power, ATL, CTL, TSB].
Same 94 athletes, same split.

**Preliminary TSS test RMSE: 14.21 bpm**

This is a **0.97 bpm improvement** (−6.4%) over the baseline. The fatigue features help.
H1 — that fatigue information improves HR prediction — is confirmed.

Note: at this stage, there was no validation set. The model was evaluated directly on the
test set, with early stopping guided by the test set, which makes the result slightly
optimistic. This was flagged by the supervisor and fixed in the definitive model.

## TRIMP Models — Results and Why They Failed

**On the TRIMP population (118 athletes):**
- Baseline (power only): 16.06 bpm
- TRIMP fatigue LSTM: 16.26 bpm — **+0.20 bpm worse than baseline**

**On the TSS population (94 athletes, head-to-head comparison):**
- Baseline: 15.18 bpm
- TSS fatigue features: 14.21 bpm (−0.97 bpm, working)
- TRIMP fatigue features: 15.22 bpm — **+0.04 bpm, essentially no improvement**

The head-to-head comparison on the same 94 athletes is decisive: TSS improves prediction by
0.97 bpm; TRIMP improves it by essentially zero.

**Why TRIMP failed — three reasons:**

**Reason 1: TRIMP is redundant with what the LSTM already sees.**
The LSTM processes the ride second-by-second, observing HR as it evolves. After the first few
minutes, the model has already built an internal picture of how that athlete's HR responds to
the current power level. TRIMP-based ATL/CTL/TSB essentially tells the model: "this athlete's
HR has been high in recent weeks." But the model can partially infer this from the current
ride's HR sequence itself. TSS, by contrast, measures *power* history — information about the
athlete's mechanical training load that the model has absolutely no other way of knowing from
the ride data alone. TSS is genuinely new cross-domain information; TRIMP is not.

**Reason 2: TSS is better normalized across athletes.**
TSS = 100 means exactly one hour at FTP for every athlete. A CTL of 60 TSS means roughly
the same thing for any athlete in the dataset. TRIMP has no such anchor. A TRIMP of 100 means
different things depending on the athlete's HR_max, resting HR, and cardiac characteristics.
The model has to learn these individual scaling factors from scratch, reducing the signal-to-noise ratio.

**Reason 3: TSS measures the cause; TRIMP measures the effect.**
The model predicts HR (an effect) from power (a cause). TSS captures the history of
*mechanical* stress — how hard the athlete has been pushing their legs. That history directly
shapes how the cardiovascular system responds to the next power input. TRIMP captures the
history of cardiovascular stress, which is itself HR-derived. Using past HR patterns to
predict future HR is circular. Using past power patterns to predict future HR is not.

**Decision:** TSS is the stronger signal. All subsequent experiments use the TSS pipeline
and the 94 TSS athletes.

---

# PART 6 — THE DEFINITIVE MODEL

## The Problem With the Preliminary Model

The supervisor flagged that using the test set as a validation proxy introduces optimism bias.
When early stopping is guided by test-set performance, you are implicitly selecting the epoch
that happens to fit the test set best — giving an overly favorable final RMSE.

## Architecture

The definitive model adds two major changes over the preliminary LSTM:

**1. Proper three-way split: 70% train / 15% validation / 15% test**
- 65 train athletes, 14 validation athletes, 15 test athletes
- Best epoch is chosen based on **validation set only**
- Test set is evaluated exactly **once**, at the very end, on the best saved model
- This gives a clean, unbiased estimate of out-of-sample performance

**2. Athlete latent vector (personalization)**
Each of the 65 training athletes has a learned 8-dimensional vector — a compact numerical
"fingerprint" of that athlete's physiology. This vector is projected via two linear layers
(h_proj, c_proj) into the LSTM's initial hidden and cell states. Before the ride even starts,
the model is initialized with knowledge of who the athlete is.

The model decides what each of the 8 dimensions encodes. It might implicitly learn fitness
level, resting HR, HR-power sensitivity, cardiac efficiency — we don't specify it. We just
give the model 8 free parameters per athlete and let it use them however it finds useful.

Why 8 dimensions? Small enough to prevent overfitting (8 × 65 = 520 parameters total for
all athlete fingerprints combined), large enough to capture meaningful individual variation.

**Other hyperparameters:**
- Hidden size: 64 (doubled from preliminary's 32)
- Dropout: 0.2
- Training: 30 epochs, batch size 64, on Google Colab T4 GPU
- Input features: [power, ATL, CTL, TSB] — same 4 features as preliminary model

## Test-Athlete Adaptation

Test athletes were never seen during training. They have no pre-learned vector. At evaluation
time, their latent vector is initialized as the **average of all 65 train-athlete vectors**
(a reasonable neutral starting point), then fine-tuned for 5 epochs on the first 30% of
their rides — LSTM weights are frozen during this step, only the latent vector is updated.
The remaining 70% of each test athlete's rides are used for evaluation.

This mirrors a realistic deployment scenario: a new user provides a calibration period,
and the model adapts to them.

## Training Results

| Epoch | Train RMSE | Val RMSE  |
|-------|-----------|-----------|
| 1     | 17.70 bpm | 14.15 bpm |
| 5     | 13.63 bpm | 12.18 bpm |
| 9     | 11.87 bpm | 10.91 bpm |
| 15    | 11.15 bpm | 10.65 bpm |
| 28    | 10.61 bpm | **10.63 bpm** ← best val |
| 30    | 10.31 bpm | 10.78 bpm |

**Best validation RMSE: 10.63 bpm (epoch 28)**
**Final test RMSE: 10.57 bpm** — close to validation RMSE, confirming the model generalizes
well and is not overfit.

## Full RMSE Progression

| Model                          | Test RMSE | Change vs previous |
|--------------------------------|-----------|--------------------|
| Baseline (power only)          | 15.18 bpm | —                  |
| Preliminary TSS LSTM           | 14.21 bpm | −0.97 bpm (−6.4%)  |
| **Definitive model**           | **10.57 bpm** | −3.64 bpm (−26%) vs preliminary TSS |

Total improvement from baseline: **−4.61 bpm (−30%)**

The two major drivers of improvement:
- **Fatigue features alone** (TSS → preliminary model): −0.97 bpm. Adding ATL/CTL/TSB gives
  the model information about training history that improves predictions.
- **Latent vector** (preliminary → definitive): −3.64 bpm. Knowing *who* the athlete is
  allows the model to start each ride already calibrated to that individual's physiology.
  This is the largest single improvement in the project.

**Comparison to the group project (12.58 bpm):** the definitive model (10.57 bpm) improves
on the group result by 2.01 bpm (−16%). The group model had no athlete personalization and
no fatigue features; both additions contributed to the improvement.

---

# PART 7 — MAXIMUM PERFORMANCE PREDICTION

## What This Extension Does

The definitive model predicts HR from power for any ride. This can be inverted: instead of
asking "what HR would this power produce?", we can ask "what is the maximum power this athlete
can sustain before their heart rate reaches its ceiling?"

This is a practical application of the model — estimating an athlete's maximum sustainable
power output under their current physiological state (as captured by their adapted latent
vector and ATL/CTL/TSB values).

## Method

For each of the 15 test athletes:

1. **Adapt the latent vector** (5 epochs, 30% of rides, LSTM frozen) — same as before.

2. **Define the HR ceiling**: 90% of the athlete's 99th percentile heart rate across all
   their recorded rides. Using the 99th percentile (not the absolute maximum) removes sensor
   spike artifacts — one rogue 595 bpm reading would otherwise create an unreachable threshold.

3. **Run a synthetic power ramp**: feed the model a sequence of constant-power steps from
   50W to 700W (in 1W increments, each held for a full 60-second window). At each power
   level, record the mean predicted HR over the final 30 seconds of that step (allowing the
   HR to stabilize).

4. **Find the crossing point**: the first power level where mean predicted HR ≥ HR ceiling.
   Use linear interpolation between the last step below threshold and the first step above
   to get a precise estimate. This is the estimated **maximum sustainable power**.

## Results

**15 test athletes:**
- Mean estimated max power: **175 watts**
- Range: approximately 120 W to 210 W

**All 94 athletes (train + val + test):**
- Mean estimated max power: **198 watts**
- Range: approximately 120 W to 360 W

The 15 test athletes happen to sit slightly below average (175W vs 198W overall) — this
reflects natural variation in the split, not a systematic problem. The full 94-athlete range
(120–360W) is physiologically realistic: a factor of 3× between the weakest and strongest
athlete is typical for a heterogeneous recreational population.

**Why 198W, not 350W?** GoldenCheetah users skew toward hobby/recreational athletes, not
competitive racers. Trained racers would typically be 250–350W. A population average of 198W
is consistent with motivated amateur cyclists.

## Important Caveat

This estimate is not a true FTP or VO2max test result. It carries the full 10.57 bpm RMSE
uncertainty of the underlying model. It is an answer to: "according to this model, given this
athlete's adaptation rides, at what power does predicted HR approach their observed ceiling?"
Not a laboratory measurement.

---

# PART 8 — FATIGUE DIRECTION ANALYSIS

## Why This Section Exists

The RMSE improvement from 15.18 to 14.21 bpm confirms that ATL/CTL/TSB help predict HR.
But this only tells us the model is more accurate — it does not tell us *why* or *how*.

The deeper question: **did the model actually learn the correct causal direction?**

Higher TSB (more rested) should produce lower predicted HR at the same power — the athlete
is fresh and their heart is working efficiently. Lower TSB (more fatigued) should produce
higher predicted HR at the same power — the heart must work harder under accumulated stress.

If the model learned this correctly, the coefficient in a regression of TSB vs predicted HR
(holding power constant) should be **negative**: higher TSB → lower HR.

If it learned it backwards, the coefficient would be **positive**.

**Across all five analyses, the coefficient was consistently positive — the wrong direction.**

This section documents five attempts to explain and fix this, why each failed, and what
the root cause is.

---

## Trial 1 — Optimal Fatigue State via Power Ramp

### What We Did

Using the same power ramp method developed for maximum performance prediction, we swept TSB
from −40 to +20 while holding CTL fixed at each athlete's median value (ATL = CTL − TSB).
At each TSB value, we ran the full ramp and found the maximum sustainable power.

If the model had learned fatigue correctly, a fresher athlete (positive TSB) should be able
to sustain more power — the fresh athlete can push harder before hitting their HR ceiling.

### Result

**Mean optimal TSB = −25. The model says athletes perform best when most fatigued.**

11 out of 14 athletes showed their highest estimated max power at the most fatigued end of
the sweep (TSB = −40 or −35). As TSB increased from −40 to +20, mean estimated max power
decreased from approximately 188W to approximately 174W.

This is physiologically backwards. Nobody performs best when heavily fatigued.

### What This Tells Us

The model learned "lower TSB (higher ATL)" as a signal for a fitter athlete (lower HR at
any given power) rather than as a signal for a fatigued athlete (higher HR at any given
power). Artificially increasing ATL (by lowering TSB) makes the model treat the athlete as
more trained, which lowers predicted HR and pushes the estimated max power ceiling higher.

This was the first clear evidence that the model has the fatigue direction wrong.

**Connection to Trial 2:** Trial 1 used synthetic TSB sweeps across athletes, which confounds
fatigue with fitness. Trial 2 addresses this by staying within one athlete at a time.

---

## Trial 2 — Within-Athlete Real Rides Analysis

### The Problem With Trial 1

Trial 1 compared athletes with high ATL against athletes with low ATL — but in this dataset,
athletes with consistently high ATL tend to be fitter athletes. They train frequently, so ATL
is always elevated. Fit athletes have lower HR at the same power. The model learned:
"high ATL = lower HR" not because fatigue reduces HR, but because high-ATL athletes are
inherently fitter.

This is the **between-athlete fitness confound**: across different athletes, high ATL
correlates with high fitness, which correlates with low HR. The causal direction
(within one athlete, high ATL means more fatigued, which means higher HR) is drowned out.

### The Fix in Trial 2

Stay within one athlete at a time. For each test athlete:
1. Adapt their latent vector on the first 30% of rides
2. For every remaining ride, run the model and record: mean predicted HR (second half of ride),
   mean power, and TSB value for that ride
3. **Residualize predicted HR on mean power**: fit a linear regression of predicted HR vs
   mean power, and keep the residuals. This removes the "harder rides have higher HR" effect —
   what remains is the portion of predicted HR that cannot be explained by the power level alone.
4. Regress the power-adjusted residuals on TSB: the slope is the model's implicit TSB coefficient.

A negative slope = model predicts lower HR at higher TSB (correct).
A positive slope = model predicts higher HR at higher TSB (wrong).

### Results

| Athlete | TSB coef | r      | p-value | n rides | Direction |
|---------|----------|--------|---------|---------|-----------|
| 1       | +0.1443  | 0.349  | 0.000   | 801     | WRONG     |
| 2       | +0.1109  | 0.323  | 0.020   | 52      | WRONG     |
| 3       | +0.0837  | 0.298  | 0.000   | 361     | WRONG     |
| 4       | −0.0130  | −0.027 | 0.825   | 69      | correct   |
| 5       | −0.0464  | −0.059 | 0.354   | 249     | correct   |
| 6       | +0.0963  | 0.204  | 0.000   | 696     | WRONG     |
| 7       | −0.0092  | −0.021 | 0.786   | 174     | correct   |
| 8       | +0.1496  | 0.284  | 0.000   | 159     | WRONG     |
| 9       | +0.0424  | 0.238  | 0.089   | 52      | WRONG     |
| 10      | +0.1538  | 0.724  | 0.000   | 126     | WRONG     |
| 11      | +0.0367  | 0.138  | 0.229   | 78      | WRONG     |
| 12      | +0.2051  | 0.481  | 0.001   | 41      | WRONG     |
| 13      | +0.0582  | 0.156  | 0.057   | 150     | WRONG     |
| 14      | +0.1685  | 0.410  | 0.000   | 70      | WRONG     |
| 15      | +0.3187  | 0.728  | 0.000   | 93      | WRONG     |

**Correct direction: 3 out of 15 athletes (20%)**
**Mean TSB coefficient: +0.10 bpm per TSB unit**

The within-athlete analysis still shows the wrong direction. Even within a single athlete's
timeline, higher TSB predicts higher — not lower — modeled HR.

### Sub-test: TSB-Only Model

**Hypothesis:** Maybe CTL is causing the problem. CTL is the long-term average — it directly
encodes how fit the athlete is. If we remove ATL and CTL and keep only TSB, the model must use
TSB purely as a fatigue signal rather than a fitness proxy.

**Architecture:** INPUT_SIZE = 2: [power, TSB] only. Everything else the same.

**Results:**
- Test RMSE: 10.68 bpm (vs 10.57 bpm full model — only 0.11 bpm worse)
- Correct direction: **2 out of 15 (13%) — worse than the full model**
- Mean TSB coefficient: +0.05 bpm/unit (effect halved but still wrong direction)

Removing CTL made the direction result worse, not better. TSB alone carries just as much
fitness-proxy information as the full set. The model still learns it backwards.

A striking finding: the accuracy is almost identical with or without ATL/CTL/TSB (10.57 vs
10.68 bpm), but the direction problem persists. This tells us the model uses these features
more as athlete-type characterizers than as session-to-session fatigue signals.

### Sub-test: Simple Model Without Latent Vector

**Hypothesis:** Perhaps the latent vector absorbed so much fitness information that TSB lost
all influence. A simpler model without personalization would force TSB to carry more
predictive weight and might learn the correct direction.

**Architecture:** Plain LSTM, hidden=32, no athlete embedding. Same data, same split.

**Results:**
- Correct direction: **1 out of 15 (7%) — worse than the personalized model (3/15)**
- Mean TSB coefficient: +0.14 bpm/unit

Removing the latent vector made the direction worse. The personalized model actually gets
slightly more athletes right (3/15 vs 1/15), even though it has a larger architecture.
Model complexity is not the cause of the wrong direction.

**Connection to Trial 3:** Trial 2 showed that even within-athlete regressions give the
wrong direction. Maybe the latent vector didn't fully absorb the between-athlete fitness
confound because it was only adapted for 5 epochs. Trial 3 attacks the problem differently:
mathematically remove the between-athlete fitness signal from ATL/CTL/TSB before training.

---

## Trial 3 — Within-Athlete Standardization

### The Hypothesis

The root cause of the between-athlete confound is that raw ATL/CTL/TSB values differ
enormously between athletes. An athlete with CTL = 100 is in a different universe from one
with CTL = 6. The model sees the raw numbers and learns that "CTL = 100 → lower HR" —
because high-CTL athletes happen to be fitter.

If we **standardize ATL/CTL/TSB within each athlete** (z-score: subtract each athlete's
own mean, divide by their own standard deviation), the numbers can no longer carry
between-athlete fitness information. A standardized ATL of +1.5 means "this athlete's ATL
is 1.5 standard deviations above their own average" — it tells you something about their
current state relative to their personal baseline, not where they sit relative to other athletes.

### The Fix

For training athletes: compute mean and standard deviation of ATL, CTL, TSB from all their rides.
For test athletes: compute mean and standard deviation from adaptation rides only (to prevent
any leakage of test-ride information into the normalization).

Then feed standardized values to the model: the features become
[power, ATL_std, CTL_std, TSB_std] at each timestep.

### Results

**Test RMSE: 11.20 bpm** (vs 10.57 bpm definitive model — 0.63 bpm worse)

The standardization costs 0.63 bpm in accuracy. This makes sense: by removing the
between-athlete scaling, we removed some information the model was correctly using
(even if for the wrong reason).

**Fatigue direction:**

| Athlete | TSB coef   | r      | p-value | n rides | Direction |
|---------|------------|--------|---------|---------|-----------|
| 1       | +0.4436    | 0.779  | 0.000   | 801     | WRONG     |
| 2       | −0.0227    | −0.021 | 0.881   | 52      | correct   |
| 3       | +0.0937    | 0.459  | 0.000   | 361     | WRONG     |
| 4       | +0.0001    | 0.000  | 0.998   | 69      | WRONG     |
| 5       | +0.1968    | 0.178  | 0.005   | 249     | WRONG     |
| 6       | +0.1074    | 0.449  | 0.000   | 696     | WRONG     |
| 7       | −0.1656    | −0.363 | 0.000   | 174     | correct   |
| 8       | +0.3121    | 0.550  | 0.000   | 159     | WRONG     |
| 9       | −0.1663    | −0.418 | 0.002   | 52      | correct   |
| 10      | +0.0705    | 0.554  | 0.000   | 126     | WRONG     |
| 11      | −0.1448    | −0.286 | 0.011   | 78      | correct   |
| 12      | +0.9354    | 0.753  | 0.000   | 41      | WRONG     |
| 13      | +0.1599    | 0.372  | 0.000   | 150     | WRONG     |
| 14      | +0.4251    | 0.604  | 0.000   | 70      | WRONG     |
| 15      | +0.3530    | 0.709  | 0.000   | 93      | WRONG     |

**Correct direction: 4 out of 15 (27%)**

A marginal improvement (3/15 → 4/15), but the direction is still wrong for 11 out of 15
athletes. Removing the between-athlete fitness confound mathematically helped slightly —
but the direction problem persists strongly within each athlete's own data.

### Why It Still Failed — The Within-Athlete Temporal Confound

Even after standardization, within each athlete's own training timeline, periods of high
training load (negative TSB) tend to coincide with periods of improving fitness. A cyclist
who trains intensively through July has lower HR in July than in March — not because they
are fatigued, but because that intensive training block is making them fitter. The model
picks up this seasonal pattern even within one athlete's data.

This is the **within-athlete temporal confound**: on a short timescale (week to week),
higher TSB should mean fresher athlete → lower HR. But over a training season, high-load
periods (negative TSB) coincide with fitness gains, which also lower HR. The two effects
point in opposite directions, and the seasonal fitness gain wins.

---

## All Five Trials — Summary

| Trial | Approach | Correct direction | Mean TSB coef |
|-------|----------|------------------|---------------|
| 1 | Optimal fatigue via power ramp | — (backwards result) | — |
| 2 | Within-athlete real rides regression | 3/15 (20%) | +0.10 bpm/unit |
| 2a | TSB-only model (no ATL/CTL) | 2/15 (13%) | +0.05 bpm/unit |
| 2b | Simple model (no latent vector) | 1/15 (7%) | +0.14 bpm/unit |
| 3 | Within-athlete standardization | 4/15 (27%) | +0.17 bpm/unit |

No approach got more than 4/15 correct. The direction problem is consistent and robust.

---

# PART 9 — ROOT CAUSE AND CONCLUSIONS

## What the Model Actually Learned

The RMSE improvements are genuine. The model is better at predicting heart rate when it
has ATL/CTL/TSB, and even better with the athlete latent vector. But the five direction
analyses reveal that the model did not learn the *causal fatigue mechanism*.

What it learned instead: **ATL/CTL/TSB as a fitness characterization signal**.

High CTL in this dataset correlates with being a trained, fit athlete. Fit athletes have
lower HR at any given power. The model learned this correlation. It does not know the
difference between "this athlete has CTL = 80 because they are a consistently trained cyclist"
and "this athlete's CTL just jumped from 40 to 80 because of a brutal training block and they
are exhausted."

## The Two Confounds

**1. Between-athlete fitness confound**
Across the dataset, athletes with high ATL tend to be fitter. They train frequently by habit,
keeping ATL consistently elevated. Fit athletes have lower HR at any power level. The model
learned this population-level pattern rather than the within-athlete fatigue signal.

**2. Within-athlete temporal confound**
Even within one athlete's timeline, periods of heavy training and negative TSB tend to align
with periods of rising fitness. The cyclist's fitness trajectory across a season creates a
spurious within-athlete correlation: high load → lower HR, but through fitness improvement,
not fatigue. The model cannot distinguish "I have lower HR now because I'm fit" from "I have
lower HR now because I've been training hard." The seasonal pattern dominates the week-to-week
fatigue signal.

## What Was Ruled Out

**Ride variability hypothesis (tested and rejected):** An early explanation suggested that
fresh athletes (positive TSB) choose more variable rides — intervals and hard efforts — while
fatigued athletes choose steadier, easier rides. If so, the apparent positive TSB coefficient
would just reflect athletes doing harder rides on fresh days. This was tested directly by
computing a variability index (NP / mean power) for every ride across all 15 test athletes.
Linear regression of TSB vs variability index gave slope ≈ 0, r = −0.015, p = 0.299. There
is no relationship. This explanation does not hold in the data.

**Model complexity (tested and rejected):** The simple model with no latent vector performed
*worse* on the direction check (1/15) than the full definitive model (3/15). Removing
personalization did not fix the direction. The problem is in the data structure, not the
model architecture.

## Scientific Conclusions

**H1 confirmed:** Fatigue features improve HR prediction. Baseline 15.18 → TSS model 14.21 bpm
(−0.97 bpm, −6.4%). This is a real, measurable improvement.

**Personalization confirmed:** Athlete latent vector drives the largest gain. 14.21 → 10.57 bpm
(−3.64 bpm, −26%). Knowing who the athlete is matters far more than knowing their fatigue state.

**Causal direction not learned:** Best result across all analyses was 4/15 athletes correct
(27%). The model uses ATL/CTL/TSB primarily to differentiate athlete types at population
level, not as a day-to-day fatigue signal. The RMSE improvement from fatigue features is real
but reflects fitness characterization, not causal fatigue learning.

**Future work:** To capture true day-to-day fatigue effects, the model would need direct
physiological fatigue signals that change daily without being confounded by fitness trends.
The most promising candidates: **HRV (heart rate variability)** — depressed on fatigued days
regardless of fitness level; **resting morning HR** — elevated under true fatigue; and
**sleep quality scores** — directly capture recovery state. With these features, the
within-athlete daily variation that ATL/CTL/TSB cannot cleanly separate would become
genuinely distinguishable.

---

# APPENDIX — KEY NUMBERS QUICK REFERENCE

| Metric | Value |
|--------|-------|
| Raw dataset | 318 athletes, 124,153 rides |
| TSS dataset | 94 athletes |
| TRIMP dataset | 118 athletes |
| Train / Val / Test split | 70% / 15% / 15% (by athlete) |
| Train athletes | 65 |
| Val athletes | 14 |
| Test athletes | 15 |
| Baseline RMSE | 15.18 bpm |
| Preliminary TSS RMSE | 14.21 bpm (−0.97 bpm) |
| Definitive model RMSE | 10.57 bpm (−4.61 bpm vs baseline) |
| TRIMP on TSS population | 15.22 bpm (no improvement) |
| TRIMP on TRIMP population | 16.26 bpm (worse than baseline) |
| Latent vector dimensions | 8 |
| Adaptation: epochs | 5 |
| Adaptation: % of rides used | 30% |
| Best val epoch | 28 |
| Mean max power (15 test athletes) | 175 W |
| Mean max power (94 athletes) | 198 W |
| HR threshold for max power | 90% of 99th percentile HR |
| Optimal TSB (definitive model) | −25 (backwards) |
| Optimal TSB (simple model) | −32.9 (worse, also backwards) |
| Best fatigue direction result | 4/15 correct (Trial 3, standardization) |
| Ride variability p-value | p = 0.299 (not significant) |
| HR_rest for TRIMP verification | 50 bpm (median error = −2 points) |
| HR_MAX_HIGH cutoff | 210 bpm |
| Athletes removed by HR_MAX_HIGH | 112 (35% of original 318) |
