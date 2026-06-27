"""
ATL / CTL / TSB Computation
----------------------------
Computes daily fatigue metrics for each athlete from the cleaned dataset.

  ATL (Acute Training Load)    : 7-day exponential moving average of TSS
  CTL (Chronic Training Load)  : 42-day exponential moving average of TSS
  TSB (Training Stress Balance): CTL - ATL  (fitness minus fatigue)

TSS source:
  - Resolved per ride via fallback chain: coggan_tss -> a_coggan_tss ->
    coggan_tssperhour -> a_coggan_tssperhour
  - Rides where TSS is still missing (no power data): imputed with that
    athlete's mean TSS across all rides that do have a value. Treating
    these as 0 would falsely imply a rest day; the athlete trained but
    the load is unknown.
  - Gaps between rides (genuine rest days): TSS = 0, EMA decays naturally.

Output values (atl_pre, ctl_pre, tsb_pre) represent the athlete's state
ENTERING the ride — i.e. end-of-day values from the previous day. This
is the correct input for the LSTM: the model predicts HR during a ride
given the fatigue state the athlete carries into it.

Reads  : Dataset_Reduced.zip
Writes : tss_atl_ctl_tsb.csv
"""

import csv
import io
import json
import math
import zipfile
from datetime import date, timedelta

INPUT_PATH  = r"C:\Users\Gebruiker\Desktop\The_Project\Dataset_Reduced.zip"
OUTPUT_PATH = r"C:\Users\Gebruiker\Desktop\The_Project\tss_atl_ctl_tsb.csv"

TSS_CHAIN = ["coggan_tss", "a_coggan_tss"]

ATL_TAU   = 7
CTL_TAU   = 42
ALPHA_ATL = 1 - math.exp(-1 / ATL_TAU)   # ~0.1331
ALPHA_CTL = 1 - math.exp(-1 / CTL_TAU)   # ~0.0235


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_float(value, index=None):
    try:
        if index is not None:
            value = value[index]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return None


def parse_date(date_str):
    """Parse GoldenCheetah UTC date string: 'YYYY/MM/DD HH:MM:SS UTC'."""
    try:
        parts = date_str.strip().split(" ")
        return date.fromisoformat(parts[0].replace("/", "-"))
    except (AttributeError, ValueError, IndexError):
        return None


def resolve_tss(m):
    """Return TSS from fallback chain, or None if all fields missing."""
    for field in TSS_CHAIN:
        v = get_float(m.get(field))
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Per-athlete computation
# ---------------------------------------------------------------------------

def process_athlete(athlete_zip_bytes):
    with zipfile.ZipFile(io.BytesIO(athlete_zip_bytes)) as az:
        json_files = [n for n in az.namelist() if n.endswith(".json")]
        if not json_files:
            return []
        with az.open(json_files[0]) as f:
            data = json.load(f)

    rides = data.get("RIDES", [])
    if not rides:
        return []

    # -- Step 1: resolve TSS per ride, collect valid values for mean imputation
    ride_records = []
    tss_values   = []

    for ride in rides:
        d = parse_date(ride.get("date", ""))
        if d is None:
            continue
        tss = resolve_tss(ride.get("METRICS", {}))
        ride_records.append({"date": d, "tss": tss, "imputed": False})
        if tss is not None:
            tss_values.append(tss)

    if not ride_records:
        return []

    # -- Step 2: impute missing TSS with athlete's mean
    mean_tss = sum(tss_values) / len(tss_values) if tss_values else 0.0
    for r in ride_records:
        if r["tss"] is None:
            r["tss"]     = mean_tss
            r["imputed"] = True

    # -- Step 3: build daily TSS map (sum multiple rides on same day)
    daily_tss = {}
    for r in ride_records:
        daily_tss[r["date"]] = daily_tss.get(r["date"], 0.0) + r["tss"]

    # -- Step 4: fill every calendar day from first to last ride, gaps = 0
    sorted_days = sorted(daily_tss)
    first_day   = sorted_days[0]
    last_day    = sorted_days[-1]

    full_timeline = {}
    current = first_day
    while current <= last_day:
        full_timeline[current] = daily_tss.get(current, 0.0)
        current += timedelta(days=1)

    # -- Step 5: compute daily ATL/CTL via EMA
    daily_atl = {}
    daily_ctl = {}
    atl = 0.0
    ctl = 0.0

    prev_day = first_day - timedelta(days=1)
    daily_atl[prev_day] = 0.0
    daily_ctl[prev_day] = 0.0

    for d in sorted(full_timeline):
        tss  = full_timeline[d]
        atl  = atl * (1 - ALPHA_ATL) + tss * ALPHA_ATL
        ctl  = ctl * (1 - ALPHA_CTL) + tss * ALPHA_CTL
        daily_atl[d] = atl
        daily_ctl[d] = ctl

    # -- Step 6: output one row per ride using previous day's ATL/CTL/TSB
    rows = []
    for r in ride_records:
        d        = r["date"]
        prev     = d - timedelta(days=1)
        atl_pre  = daily_atl.get(prev, 0.0)
        ctl_pre  = daily_ctl.get(prev, 0.0)
        tsb_pre  = ctl_pre - atl_pre
        rows.append({
            "date":     d.isoformat(),
            "tss":      round(r["tss"], 2),
            "imputed":  r["imputed"],
            "atl_pre":  round(atl_pre, 4),
            "ctl_pre":  round(ctl_pre, 4),
            "tsb_pre":  round(tsb_pre, 4),
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_rows       = []
    total_athletes = 0
    total_imputed  = 0

    with zipfile.ZipFile(INPUT_PATH, "r") as src:
        inner_zips = sorted(n for n in src.namelist() if n.endswith(".zip"))
        n_athletes = len(inner_zips)
        print(f"Processing {n_athletes} athletes...")

        for idx, name in enumerate(inner_zips, 1):
            if idx % 20 == 0:
                print(f"  {idx}/{n_athletes}")

            athlete_id = name.replace(".zip", "")
            data       = src.read(name)
            rows       = process_athlete(data)

            if not rows:
                continue

            imputed_n = sum(1 for r in rows if r["imputed"])
            total_imputed  += imputed_n
            total_athletes += 1

            for r in rows:
                all_rows.append({
                    "athlete_id": athlete_id,
                    **r,
                })

    # Write CSV
    fieldnames = ["athlete_id", "date", "tss", "imputed", "atl_pre", "ctl_pre", "tsb_pre"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone.")
    print(f"  Athletes processed : {total_athletes}")
    print(f"  Total ride rows    : {len(all_rows)}")
    print(f"  Imputed TSS rides  : {total_imputed}  ({total_imputed/len(all_rows)*100:.1f}%)" if all_rows else "  Imputed TSS rides  : 0")
    print(f"\nOutput written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
