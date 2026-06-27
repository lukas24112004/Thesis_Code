"""
TRIMP ATL / CTL / TSB Computation
-----------------------------------
Computes daily fatigue metrics for each athlete in TRIMP_Bounds_Filtered_Dataset.zip
using the pre-stored trimp_points from the JSON (Banister formula).

  ATL (Acute Training Load)    : 7-day EMA of daily TRIMP   (alpha = 1 - e^(-1/7))
  CTL (Chronic Training Load)  : 42-day EMA of daily TRIMP  (alpha = 1 - e^(-1/42))
  TSB (Training Stress Balance): CTL - ATL

Output values (atl_pre, ctl_pre, tsb_pre) represent the athlete's state ENTERING
the ride — end-of-day values from the previous calendar day.

Rules:
  - Multiple rides on the same day: daily TRIMP is summed; all rides receive the same pre-ride values
  - Gap days (no ride): TRIMP = 0, EMA decays naturally
  - Rides with null or zero trimp_points: excluded from output rows but gap days still decay EMA

Reads  : TRIMP_Bounds_Filtered_Dataset.zip  (118 athletes)
Writes : trimp_atl_ctl_tsb.csv
"""

import csv
import io
import json
import math
import zipfile
from datetime import date, timedelta

INPUT_PATH  = r"C:\Users\Gebruiker\Desktop\The_Project\TRIMP_Bounds_Filtered_Dataset.zip"
OUTPUT_PATH = r"C:\Users\Gebruiker\Desktop\The_Project\trimp_atl_ctl_tsb.csv"

ATL_TAU   = 7
CTL_TAU   = 42
ALPHA_ATL = 1 - math.exp(-1 / ATL_TAU)
ALPHA_CTL = 1 - math.exp(-1 / CTL_TAU)


def parse_date(date_str):
    """Parse GoldenCheetah UTC date string: 'YYYY/MM/DD HH:MM:SS UTC'."""
    try:
        return date.fromisoformat(date_str.strip().split(" ")[0].replace("/", "-"))
    except (AttributeError, ValueError, IndexError):
        return None


def process_athlete(athlete_zip_bytes):
    with zipfile.ZipFile(io.BytesIO(athlete_zip_bytes)) as az:
        json_file = next((n for n in az.namelist() if n.endswith(".json")), None)
        if json_file is None:
            return []
        with az.open(json_file) as f:
            data = json.load(f)

    rides = data.get("RIDES", [])
    if not rides:
        return []

    # Step 1 — collect all rides, resolve trimp_points where present
    ride_records  = []
    trimp_values  = []

    for ride in rides:
        d = parse_date(ride.get("date", ""))
        if d is None:
            continue
        trimp = ride.get("METRICS", {}).get("trimp_points")
        if trimp is not None:
            try:
                trimp = float(trimp)
                if trimp <= 0:
                    trimp = None
            except (TypeError, ValueError):
                trimp = None
        ride_records.append({"date": d, "trimp": trimp, "imputed": False})
        if trimp is not None:
            trimp_values.append(trimp)

    if not ride_records:
        return []

    # Step 2 — impute missing trimp with athlete's mean (same logic as TSS pipeline)
    mean_trimp = sum(trimp_values) / len(trimp_values) if trimp_values else 0.0
    for r in ride_records:
        if r["trimp"] is None:
            r["trimp"]   = mean_trimp
            r["imputed"] = True

    # Step 3 — sum multiple rides on same day
    daily_trimp = {}
    for r in ride_records:
        daily_trimp[r["date"]] = daily_trimp.get(r["date"], 0.0) + r["trimp"]

    # Step 4 — fill every calendar day from first to last ride; gaps get TRIMP = 0
    first_day = min(daily_trimp)
    last_day  = max(daily_trimp)
    full_timeline = {}
    current = first_day
    while current <= last_day:
        full_timeline[current] = daily_trimp.get(current, 0.0)
        current += timedelta(days=1)

    # Step 5 — compute end-of-day ATL and CTL via EMA
    atl = 0.0
    ctl = 0.0
    daily_atl = {first_day - timedelta(days=1): 0.0}
    daily_ctl = {first_day - timedelta(days=1): 0.0}

    for d in sorted(full_timeline):
        atl = atl * (1 - ALPHA_ATL) + full_timeline[d] * ALPHA_ATL
        ctl = ctl * (1 - ALPHA_CTL) + full_timeline[d] * ALPHA_CTL
        daily_atl[d] = atl
        daily_ctl[d] = ctl

    # Step 6 — one output row per ride; values are from end of previous day (pre-ride state)
    rows = []
    for r in ride_records:
        prev    = r["date"] - timedelta(days=1)
        atl_pre = daily_atl.get(prev, 0.0)
        ctl_pre = daily_ctl.get(prev, 0.0)
        rows.append({
            "date":     r["date"].isoformat(),
            "trimp":    round(r["trimp"], 2),
            "imputed":  r["imputed"],
            "atl_pre":  round(atl_pre, 4),
            "ctl_pre":  round(ctl_pre, 4),
            "tsb_pre":  round(ctl_pre - atl_pre, 4),
        })

    return rows


def main():
    all_rows = []
    skipped  = 0

    with zipfile.ZipFile(INPUT_PATH, "r") as src:
        inner_zips = sorted(n for n in src.namelist() if n.endswith(".zip"))
        n_athletes = len(inner_zips)
        print(f"Processing {n_athletes} athletes...")

        for idx, name in enumerate(inner_zips, 1):
            if idx % 20 == 0:
                print(f"  {idx}/{n_athletes}  ({len(all_rows)} rows so far)")

            athlete_id = name.replace(".zip", "")
            rows       = process_athlete(src.read(name))

            if not rows:
                skipped += 1
                continue

            for r in rows:
                all_rows.append({"athlete_id": athlete_id, **r})

    fieldnames = ["athlete_id", "date", "trimp", "imputed", "atl_pre", "ctl_pre", "tsb_pre"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_imputed = sum(1 for r in all_rows if r["imputed"])
    print(f"\nDone.")
    print(f"  Athletes processed : {n_athletes - skipped}")
    print(f"  Athletes skipped   : {skipped}")
    print(f"  Total ride rows    : {len(all_rows):,}")
    print(f"  Imputed TRIMP rides: {n_imputed} ({n_imputed/len(all_rows)*100:.1f}%)" if all_rows else "")

    if all_rows:
        trimps = [r["trimp"] for r in all_rows]
        atls   = [r["atl_pre"] for r in all_rows]
        ctls   = [r["ctl_pre"] for r in all_rows]
        print(f"  TRIMP range        : {min(trimps):.1f} – {max(trimps):.1f}  (mean {sum(trimps)/len(trimps):.1f})")
        print(f"  ATL range          : {min(atls):.1f} – {max(atls):.1f}")
        print(f"  CTL range          : {min(ctls):.1f} – {max(ctls):.1f}")

    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
