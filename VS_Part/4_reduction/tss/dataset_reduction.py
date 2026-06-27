"""
Dataset Reduction
-----------------
Single-pass reduction of Definitive_Dataset.zip (318 athletes).

Cuts applied in order (athlete excluded at first failing check):
  1. FTP_DEFAULT  : cp_setting == 250W on every ride (GoldenCheetah default, never configured)
  2. FTP_STAGNANT : cp_setting never changes across all rides (set once, never updated)
  3. MISSING_TSS  : >5% of rides have no TSS value (block-structured gaps corrupt ATL/CTL)
  4. TSS_HIGH     : >5% of rides confirmed TSS > 500 (both stored and recalculated)
  5. IF_LOW       : >5% of rides confirmed IF < 0.3  (both stored and recalculated, likely inflated FTP)

TSS resolved via fallback chain: coggan_tss -> a_coggan_tss -> coggan_tssperhour -> a_coggan_tssperhour

Reads  : Definitive_Dataset.zip  (318 athletes)
Writes : Dataset_Reduced.zip
"""

import io
import json
import zipfile

INPUT_PATH  = r"C:\Users\Gebruiker\Desktop\The_Project\Definitive_Dataset.zip"
OUTPUT_PATH = r"C:\Users\Gebruiker\Desktop\The_Project\Dataset_Reduced.zip"

DEFAULT_FTP            = 250.0
MISSING_TSS_THRESHOLD  = 0.05   # 5%
TSS_HIGH_THRESHOLD     = 0.05   # 5%
IF_LOW_THRESHOLD       = 0.05   # 5%

TSS_CHAIN = ["coggan_tss", "a_coggan_tss", "coggan_tssperhour", "a_coggan_tssperhour"]
TSS_MAX   = 500
IF_MIN    = 0.3


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


def load_rides(athlete_zip_bytes):
    with zipfile.ZipFile(io.BytesIO(athlete_zip_bytes)) as az:
        json_files = [n for n in az.namelist() if n.endswith(".json")]
        if not json_files:
            return None
        with az.open(json_files[0]) as f:
            data = json.load(f)
    return data.get("RIDES", [])


# ---------------------------------------------------------------------------
# Cut 1 & 2 — FTP reliability
# ---------------------------------------------------------------------------

def check_ftp(rides):
    ftp_values = []
    for ride in rides:
        cp = get_float(ride.get("METRICS", {}).get("cp_setting"))
        if cp is not None:
            ftp_values.append(cp)

    if not ftp_values:
        return None

    if all(abs(v - DEFAULT_FTP) < 0.01 for v in ftp_values):
        return "FTP_DEFAULT"

    unique = set(round(v, 2) for v in ftp_values)
    if len(unique) == 1:
        return "FTP_STAGNANT"

    return None


# ---------------------------------------------------------------------------
# Cut 3 — Missing TSS
# ---------------------------------------------------------------------------

def check_missing_tss(rides):
    missing = sum(
        1 for ride in rides
        if not any(get_float(ride.get("METRICS", {}).get(f)) is not None for f in TSS_CHAIN)
    )
    return missing / len(rides)


# ---------------------------------------------------------------------------
# Cut 4 & 5 — Anomaly rates (both stored AND recalculated confirmed)
# ---------------------------------------------------------------------------

def stored_flags(m):
    flags  = set()
    tss    = get_float(m.get("coggan_tss"))
    if_val = get_float(m.get("coggan_if"), index=0)
    if tss is not None and tss > TSS_MAX:
        flags.add("TSS_HIGH")
    if if_val is not None and if_val < IF_MIN:
        flags.add("IF_LOW")
    return flags


def calc_flags(m):
    if_val   = get_float(m.get("coggan_if"), index=0)
    rec_time = get_float(m.get("coggan_if"), index=1)
    if if_val is None or rec_time is None:
        return set()
    flags    = set()
    calc_tss = if_val ** 2 * (rec_time / 3600) * 100
    if calc_tss > TSS_MAX:
        flags.add("TSS_HIGH")
    if if_val < IF_MIN:
        flags.add("IF_LOW")
    return flags


def check_anomalies(rides):
    tss_high = 0
    if_low   = 0
    for ride in rides:
        m  = ride.get("METRICS", {})
        sf = stored_flags(m)
        if sf:
            confirmed = sf & calc_flags(m)
            if "TSS_HIGH" in confirmed:
                tss_high += 1
            if "IF_LOW" in confirmed:
                if_low += 1
    total = len(rides)
    return tss_high / total, if_low / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def should_exclude(athlete_zip_bytes):
    """Returns (reason_string, rates_dict) or (None, rates_dict) if kept."""
    rides = load_rides(athlete_zip_bytes)
    if not rides:
        return "NO_RIDES", {}

    rates = {}

    # Cut 1 & 2: FTP
    ftp_reason = check_ftp(rides)
    if ftp_reason:
        return ftp_reason, rates

    # Cut 3: Missing TSS
    miss = check_missing_tss(rides)
    rates["missing_tss"] = miss
    if miss > MISSING_TSS_THRESHOLD:
        return f"MISSING_TSS={miss*100:.1f}%", rates

    # Cut 4 & 5: Anomalies
    tss_high_rate, if_low_rate = check_anomalies(rides)
    rates["tss_high"] = tss_high_rate
    rates["if_low"]   = if_low_rate

    if tss_high_rate > TSS_HIGH_THRESHOLD:
        return f"TSS_HIGH={tss_high_rate*100:.1f}%", rates
    if if_low_rate > IF_LOW_THRESHOLD:
        return f"IF_LOW={if_low_rate*100:.1f}%", rates

    return None, rates


def main():
    kept     = []
    excluded = []

    reason_counts = {
        "FTP_DEFAULT":  0,
        "FTP_STAGNANT": 0,
        "MISSING_TSS":  0,
        "TSS_HIGH":     0,
        "IF_LOW":       0,
        "NO_RIDES":     0,
    }

    with zipfile.ZipFile(INPUT_PATH, "r") as src:
        inner_zips = [n for n in src.namelist() if n.endswith(".zip")]
        total = len(inner_zips)
        print(f"Reading {total} athletes from {INPUT_PATH}\n")

        with zipfile.ZipFile(OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for idx, name in enumerate(inner_zips, 1):
                if idx % 50 == 0:
                    print(f"  {idx}/{total}  (kept {len(kept)}, excluded {len(excluded)})")

                data   = src.read(name)
                reason, _ = should_exclude(data)

                if reason:
                    excluded.append((name, reason))
                    key = reason.split("=")[0]
                    reason_counts[key] = reason_counts.get(key, 0) + 1
                else:
                    kept.append(name)
                    dst.writestr(name, data)

    print(f"\nDone.")
    print(f"  Input    : {total} athletes")
    print(f"  Kept     : {len(kept)} athletes")
    print(f"  Excluded : {len(excluded)} athletes")
    print(f"\nExclusion breakdown:")
    for key, count in reason_counts.items():
        if count:
            print(f"  {key:<15} : {count}")
    print(f"\nExcluded athletes:")
    for name, reason in excluded:
        print(f"  {name:<55} {reason}")
    print(f"\nOutput written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
