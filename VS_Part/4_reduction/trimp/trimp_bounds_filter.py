"""
TRIMP Bounds Filter
--------------------
Applies TRIMP_HIGH and TRIMP_LOW bounds checks to TRIMP_Reduced_Dataset.zip
(the already HR_max + missing-filtered dataset, 145 athletes).

Cuts applied:
  TRIMP_HIGH : >5% of rides with TRIMP/hour > 300  (above physiological ceiling)
  TRIMP_LOW  : >5% of rides with TRIMP/hour < 10   (HR barely recorded / data artifact)

Duration used: workout_time from JSON METRICS (seconds).
Rides with no workout_time or workout_time <= 0 are skipped for bounds check only.

Output: TRIMP_Bounds_Filtered_Dataset.zip
"""

import io
import json
import zipfile
from collections import Counter

DATASET_PATH = r"C:\Users\Gebruiker\Desktop\The_Project\TRIMP_Reduced_Dataset.zip"
OUTPUT_PATH  = r"C:\Users\Gebruiker\Desktop\The_Project\TRIMP_Bounds_Filtered_Dataset.zip"

TRIMP_FIELD       = "trimp_points"
MISSING_THRESHOLD = 0.05
TRIMP_HIGH_PER_HR = 200
TRIMP_LOW_PER_HR  = 25


def check_athlete(az_bytes):
    with zipfile.ZipFile(io.BytesIO(az_bytes)) as inner:
        json_files = [f for f in inner.namelist() if f.endswith(".json")]
        if not json_files:
            return None, "NO_JSON"
        with inner.open(json_files[0]) as f:
            data = json.load(f)

    rides = data.get("RIDES", [])
    if not rides:
        return None, "NO_RIDES"

    n_checkable = 0
    n_high      = 0
    n_low       = 0

    for ride in rides:
        metrics = ride.get("METRICS", {})
        trimp   = metrics.get(TRIMP_FIELD)
        dur_sec = metrics.get("workout_time")

        if trimp is None or dur_sec is None:
            continue
        try:
            trimp   = float(trimp)
            dur_sec = float(dur_sec)
        except (TypeError, ValueError):
            continue

        if dur_sec <= 0:
            continue

        dur_hr = dur_sec / 3600
        trimp_per_hr = trimp / dur_hr
        n_checkable += 1

        if trimp_per_hr > TRIMP_HIGH_PER_HR:
            n_high += 1
        if trimp_per_hr < TRIMP_LOW_PER_HR:
            n_low += 1

    if n_checkable == 0:
        return None, "NO_CHECKABLE_RIDES"

    pct_high = n_high / n_checkable
    pct_low  = n_low  / n_checkable

    if pct_high > MISSING_THRESHOLD:
        return None, f"TRIMP_HIGH ({pct_high*100:.1f}% of rides > {TRIMP_HIGH_PER_HR}/hr)"

    if pct_low > MISSING_THRESHOLD:
        return None, f"TRIMP_LOW ({pct_low*100:.1f}% of rides < {TRIMP_LOW_PER_HR}/hr)"

    return (len(rides), n_checkable, n_high, n_low), None


def main():
    kept    = []
    removed = []

    print(f"Reading {DATASET_PATH.split(chr(92))[-1]}...")

    with zipfile.ZipFile(DATASET_PATH, "r") as outer:
        athlete_zips = sorted(n for n in outer.namelist() if n.endswith(".zip"))
        n_total = len(athlete_zips)
        print(f"  {n_total} athletes\n")

        athlete_data = {}
        for i, az_name in enumerate(athlete_zips, 1):
            if i % 30 == 0:
                print(f"  Scanning athlete {i}/{n_total}...")

            athlete_id = az_name.replace(".zip", "")
            az_bytes   = outer.read(az_name)
            result, reason = check_athlete(az_bytes)

            if reason:
                removed.append((athlete_id, reason))
            else:
                kept.append((athlete_id, *result))
                athlete_data[az_name] = az_bytes

    print(f"\nResults:")
    print(f"  Kept   : {len(kept)} athletes")
    print(f"  Removed: {len(removed)} athletes\n")

    reason_counts = Counter(r.split("(")[0].strip() for _, r in removed)
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:<20} {count}")

    if removed:
        print(f"\nRemoved athletes:")
        print(f"  {'Athlete':<40} Reason")
        print("  " + "-" * 70)
        for athlete_id, reason in removed:
            print(f"  {athlete_id:<40} {reason}")

    print(f"\nWriting {OUTPUT_PATH.split(chr(92))[-1]}...")
    with zipfile.ZipFile(OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as out_zip:
        for az_name, az_bytes in athlete_data.items():
            out_zip.writestr(az_name, az_bytes)

    total_rides = sum(r[1] for r in kept)
    print(f"\nKept dataset summary:")
    print(f"  Athletes   : {len(kept)}")
    print(f"  Total rides: {total_rides:,}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
