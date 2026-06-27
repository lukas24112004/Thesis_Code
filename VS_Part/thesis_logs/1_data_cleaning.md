# Data Cleaning

---

## Part A — Group Project: From Raw Dataset to 318 Athletes

### The original dataset
The data comes from GoldenCheetah, an open-source cycling analytics platform where users voluntarily upload their training files. The raw dataset was approximately 100 GB and contained ride files from hundreds of athletes spanning multiple years of training.

Each athlete's data is stored as a folder containing:
- One `.json` file per ride: ride-level summary statistics including distance, duration, normalized power, and pre-computed training load scores (TSS, TRIMP, etc.)
- One `.csv` file per ride: second-by-second recordings of power output and heart rate during the session

### Group project cleaning steps
The group project (Research Project 3) applied the following filters to reduce the raw dataset to a usable size:

1. **Removed athletes with invalid year of birth, missing gender, or fewer than 10 rides** — basic data quality, ensures demographic fields are usable and athletes have enough history.
2. **Removed rides with no HR or power data** — these rides cannot be used for model training since both signals are required.
3. **Removed rides shorter than 30 minutes or longer than 6 hours** — very short rides are likely warm-ups or partial recordings; very long rides may be data artifacts.
4. **Removed rides with HR above 220 bpm** — physiologically impossible values indicating sensor errors.
5. **Re-applied a 50-ride minimum** — after the above cuts, athletes who no longer had 50 qualifying rides were dropped.

**Result: 318 athletes, approximately 5.7 GB, 124,153 rides.**

---

## Part B — Why This Thesis Needed an Additional Step

The group project deleted rides that lacked HR or power data. This was correct for their purposes — they needed both signals for training. However, this created a problem for computing fatigue features.

**The issue:** ATL and CTL are exponential moving averages that accumulate from all training, not just rides with complete sensor data. If an athlete did a hard two-hour ride on a day their heart rate monitor failed and that ride was deleted, the athlete still accumulated fatigue from that session — their body does not know the sensor was missing. Dropping the ride from the fatigue timeline creates a gap that corrupts ATL and CTL from that point forward.

**The solution:** TSS and TRIMP scores are stored in each ride's `.json` file, even for rides whose `.csv` was removed by the group project. By reading from the JSON rather than the CSV, this project can reconstruct a complete, gap-free fatigue timeline for every athlete using all their historical rides — not just the ones with complete sensor files.

---

## Part C — TSS Data Pipeline: 318 → 94 Athletes

### What TSS is
TSS (Training Stress Score) is a power-based measure of how hard a ride was, defined as:

```
TSS = (duration_seconds × NP × IF) / (FTP × 3600) × 100
```

Where:
- **NP** (Normalized Power): average power adjusted for effort variability
- **FTP** (Functional Threshold Power): the highest power the athlete can sustain for ~1 hour
- **IF** (Intensity Factor): NP ÷ FTP — how hard the ride was relative to threshold
- **TSS = 100** means exactly one hour at FTP — the same meaning for every athlete

TSS is cross-domain: it measures mechanical load (power) to predict a physiological response (heart rate). This is scientifically clean because it avoids circularity.

### Coverage and verification
- 85.5% of rides carry a stored TSS value (the rest have missing power data)
- Missing values filled from a 4-field fallback hierarchy in the JSON
- When TSS was independently recalculated from raw power files, **99.3% of stored values matched within 1%** — confirming the stored values are trustworthy

### Quality cuts applied (in sequence)

| # | Cut | Reason |
|---|-----|---------|
| 1 | FTP never configured (stuck at default 250W) | TSS would be meaningless — every athlete anchored to the same wrong threshold |
| 2 | FTP set once and never updated | A stale FTP drifts away from true fitness, making TSS comparisons across time unreliable |
| 3 | >5% of rides missing TSS | Too many gaps to reconstruct a reliable fatigue timeline |
| 4 | >5% of rides with TSS > 500 | Implausibly high loads indicating bad FTP configuration or sensor errors |
| 5 | >5% of rides with IF < 0.3 | Near-coasting effort, likely data artifacts or non-cycling activities |

**Result: 94 athletes**

**Train/validation/test split:** 65 train / 14 validation / 15 test (70/15/15 by athlete)

---

## Part D — TRIMP Data Pipeline: 318 → 118 Athletes

### What TRIMP is
TRIMP (Training Impulse) is a heart-rate-based measure of training load, defined by Banister (1991) as:

```
TRIMP = Σ( Δt × HRr × 0.64 × e^(b × HRr) )

HRr = (HR − HR_rest) / (HR_max − HR_rest)
b   = 1.92 (male), 1.67 (female)
```

Where HRr is the athlete's heart rate expressed as a fraction between resting and maximum. The exponential weighting means that high-intensity minutes count far more than easy ones — reflecting the fact that the HR-to-blood-lactate relationship is exponential, not linear.

TRIMP measures internal load (how hard the cardiovascular system worked), while TSS measures external load (how hard the athlete pushed mechanically).

### Verification challenges
Three issues had to be resolved before TRIMP values could be trusted:

1. **No resting HR in the data** — session minimums give ~71 bpm (too high, athletes are still moving). Fixed HR_rest = 50 bpm was used instead, giving a median error of only −2 TRIMP points against independently recalculated values.
2. **Session HR_max breaks the formula** — on easy days an athlete might only reach 120 bpm, making every HR sample look like near-maximum effort. All-time HR_max (the highest HR ever recorded across all sessions) was used instead (r = 0.9987 match).
3. **Sensor spikes** — values of 219–595 bpm appear in the data. Athletes whose all-time HR_max exceeded 210 bpm were excluded entirely, as these spikes corrupt HRr for every ride in that athlete's history.

### Quality cuts applied

| Cut | Athletes removed |
|-----|-----------------|
| HR_max > 210 bpm (sensor spikes) | 112 |
| >5% of rides missing TRIMP | 61 |
| TRIMP/hour < 25 or > 200 (physiologically impossible) | 27 |
| **Kept** | **118** |

**Important:** The 94 TSS athletes and 118 TRIMP athletes are different populations — different quality requirements selected different subsets. Direct comparison between TSS and TRIMP model results requires running both on the same athletes (see Models log).

---

## Part E — ATL, CTL, and TSB: The Three Fatigue Features

Once a daily training load score exists for every athlete (either TSS or TRIMP), three features are computed and fed into the model alongside power at every timestep.

### ATL — Acute Training Load (recent fatigue)
ATL is a 7-day exponential moving average of daily training load. It rises quickly when an athlete trains hard and fades quickly when they rest. It captures how much stress has accumulated in the past week.

```
ATL_today = ATL_yesterday × (1 − α_atl) + TSS_today × α_atl
α_atl = 1 − e^(−1/7) ≈ 0.133
```

### CTL — Chronic Training Load (fitness base)
CTL is a 42-day exponential moving average of daily training load. It changes slowly — building fitness over weeks of consistent training and declining during extended rest. It captures the athlete's overall training base.

```
CTL_today = CTL_yesterday × (1 − α_ctl) + TSS_today × α_ctl
α_ctl = 1 − e^(−1/42) ≈ 0.024
```

### TSB — Training Stress Balance (form / freshness)
TSB = CTL − ATL. A positive TSB means the athlete is rested (fitness exceeds recent fatigue). A negative TSB means the athlete is carrying fatigue (recent load exceeds their base). TSB is commonly called "form" in training practice.

### How they enter the model
All three values describe the athlete's state **entering the ride** — they are computed from the previous day's end-of-day state so there is no data leakage. The model receives them as constant features alongside power at every timestep:

```
Input at each timestep: [ power, ATL, CTL, TSB ]
```

Gap days (no ride recorded) contribute TSS = 0, allowing the EMAs to decay naturally. Multiple rides on the same day all receive the same previous-day values.
