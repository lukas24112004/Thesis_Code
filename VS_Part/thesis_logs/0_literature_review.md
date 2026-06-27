# Literature Review

This thesis sits at the intersection of exercise physiology and machine learning. The four key works below together motivate every methodological choice.

---

## Banister (1991) — Modeling Elite Athletic Performance

**What it contributes:** The origin of the fatigue-fitness framework used throughout this project.

Banister proposed that an athlete's performance at any point in time is the net result of two competing physiological processes: fitness (the positive adaptation to training, which builds slowly) and fatigue (the acute cost of recent effort, which rises and fades quickly). He modelled both as exponential moving averages of training load and defined the balance between them — now called TSB (Training Stress Balance) — as a proxy for readiness to perform.

He also defined TRIMP (Training Impulse), the heart-rate-based measure of training load used in this project's TRIMP pipeline. The formula weights each minute of exercise by how hard the cardiovascular system was working, using an exponential function because the relationship between heart rate and blood lactate is exponential, not linear.

**Why it matters here:** The ATL/CTL/TSB framework and the TRIMP formula both come directly from this paper. The professor supervising this thesis cited Banister 1991 explicitly, making it the theoretical anchor for the fatigue features.

---

## Allen & Coggan (2010) — Training and Racing with a Power Meter

**What it contributes:** The power-based alternative to TRIMP — TSS, FTP, and Normalized Power.

Allen and Coggan translated the Banister framework from heart rate into power. They defined:
- **FTP** (Functional Threshold Power): the highest average power an athlete can sustain for approximately one hour.
- **Normalized Power (NP)**: a weighted version of average power that accounts for the metabolic cost of variability — hard intervals cost more than the same average effort held steady.
- **TSS** (Training Stress Score): a single number summarising how hard a ride was, anchored so that one hour at exactly FTP = TSS 100. This gives TSS a consistent meaning across all athletes.

**Why it matters here:** TSS is the fatigue signal used in the primary model. Unlike TRIMP, it is cross-domain — it measures mechanical load (power) and predicts a physiological response (heart rate), which is a cleaner scientific claim. Its athlete-anchored normalization (TSS 100 means the same thing for every rider) also makes it more useful in a multi-athlete model.

---

## Barsumyan et al. (2025) — Cardiovascular Drift and Machine Learning in Cycling

**What it contributes:** Direct empirical motivation for this project.

Barsumyan et al. showed that heart rate rises progressively at a fixed power output as fatigue accumulates within and across sessions — a phenomenon called cardiovascular drift. Standard HR-from-power models ignore this, treating every ride as if the athlete is in the same physiological state. This paper provides the empirical basis for the central claim of this thesis: that knowing an athlete's fatigue state should improve HR prediction from power.

**Why it matters here:** This is the direct motivation for the research question. It establishes that the effect we are trying to capture is real and measurable, not a theoretical assumption.

---

## Smiley & Finkelstein (2024) — LSTM for Physiological Time Series

**What it contributes:** Architectural justification for using an LSTM.

Smiley and Finkelstein benchmarked recurrent neural networks — specifically LSTMs — against traditional time-series models (linear regression, random forests, simple RNNs) on physiological prediction tasks including heart rate. LSTMs consistently outperformed the alternatives because they can maintain memory of earlier timesteps within a session, capturing phenomena like cardiac drift that only become visible over minutes of sustained effort.

**Why it matters here:** This paper justifies choosing an LSTM over simpler architectures. The model in this thesis processes an entire ride as one sequence, allowing the LSTM hidden state to accumulate a representation of how power has accumulated over time — exactly the kind of temporal dependency LSTMs are designed for.
