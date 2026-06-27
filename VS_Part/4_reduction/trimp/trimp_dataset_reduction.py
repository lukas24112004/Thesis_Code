"""
TRIMP Dataset Reduction
------------------------
Starting from Definitive_Dataset.zip (318 athletes), applies three cuts
in order — an athlete is excluded at the first failing check.

Cuts applied:
  1. HR_MAX_HIGH  : all-time HR_max > 210 bpm  (spike artifact corrupts all TRIMP values)
  2. HR_MAX_LOW   : all-time HR_max < 140 bpm  (never reached real max, HRr inflated)
  3. MISSING_TRIMP: >5% of rides have null trimp_points

Uses trimp_points (Banister formula) — NOT trimp_zonal_points.

Output: TRIMP_Reduced_Dataset.zip
"""

import io
import json
import zipfile

DATASET_PATH = r"C:\Users\Gebruiker\Desktop\The_Project\Definitive_Dataset.zip"
OUTPUT_PATH  = r"C:\Users\Gebruiker\Desktop\The_Project\TRIMP_Reduced_Dataset.zip"

TRIMP_FIELD       = "trimp_points"
MISSING_THRESHOLD = 0.05   # >5% missing trimp_points → excluded
HRMAX_HIGH        = 210    # bpm — spike artifact threshold
HRMAX_LOW         = 140    # bpm — never reached real max


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

    # All-time HR_max: max of per-ride max_heartrate across all rides
    hr_maxes = []
    for ride in rides:
        metrics = ride.get("METRICS", {})
        val = metrics.get("max_heartrate")
        if val is not None:
            try:
                hr_maxes.append(float(val))
            except (TypeError, ValueError):
                pass

    if not hr_maxes:
        return None, "NO_HRMAX"

    hrmax_alltime = max(hr_maxes)

    if hrmax_alltime > HRMAX_HIGH:
        return None, f"HR_MAX_HIGH ({hrmax_alltime:.0f} bpm)"

    if hrmax_alltime < HRMAX_LOW:
        return None, f"HR_MAX_LOW ({hrmax_alltime:.0f} bpm)"

    n_rides   = len(rides)
    n_missing = sum(
        1 for ride in rides
        if ride.get("METRICS", {}).get(TRIMP_FIELD) is None
    )
    pct_missing = n_missing / n_rides

    if pct_missing > MISSING_THRESHOLD:
        return None, f"MISSING_TRIMP ({pct_missing*100:.1f}%)"

    return (n_rides, n_missing, hrmax_alltime), None


def main():
    kept    = []
    removed = []

    print(f"Reading {DATASET_PATH.split(chr(92))[-1]}...")

    with zipfile.ZipFile(DATASET_PATH, "r") as outer:
        athlete_zips = sorted(n for n in outer.namelist() if n.endswith(".zip"))
        n_total = len(athlete_zips)
        print(f"  {n_total} athletes found\n")

        athlete_data = {}
        for i, az_name in enumerate(athlete_zips, 1):
            if i % 50 == 0:
                print(f"  Scanning athlete {i}/{n_total}...")

            athlete_id = az_name.replace(".zip", "")
            az_bytes   = outer.read(az_name)
            result, reason = check_athlete(az_bytes)

            if reason:
                removed.append((athlete_id, reason))
            else:
                n_rides, n_missing, hrmax = result
                kept.append((athlete_id, n_rides, n_missing, hrmax))
                athlete_data[az_name] = az_bytes

    print(f"\nResults:")
    print(f"  Kept   : {len(kept)} athletes")
    print(f"  Removed: {len(removed)} athletes\n")

    # Count by reason
    from collections import Counter
    reason_counts = Counter(r.split("(")[0].strip() for _, r in removed)
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:<25} {count}")

    print(f"\nWriting {OUTPUT_PATH.split(chr(92))[-1]}...")
    with zipfile.ZipFile(OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as out_zip:
        for az_name, az_bytes in athlete_data.items():
            out_zip.writestr(az_name, az_bytes)

    total_rides   = sum(r[1] for r in kept)
    total_missing = sum(r[2] for r in kept)
    coverage      = 100 * (1 - total_missing / total_rides) if total_rides else 0

    print(f"\nKept dataset summary:")
    print(f"  Athletes           : {len(kept)}")
    print(f"  Total rides        : {total_rides:,}")
    print(f"  trimp_points coverage: {coverage:.1f}%")
    print(f"  HR_max range       : {min(r[3] for r in kept):.0f} – {max(r[3] for r in kept):.0f} bpm")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
