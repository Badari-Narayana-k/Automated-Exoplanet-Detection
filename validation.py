# ==========================================================
# KEPLER CANDIDATE VALIDATION PIPELINE
# ==========================================================

import pandas as pd
import numpy as np

# ================= FILE PATHS =============================
TLS_CANDIDATES_FILE = "/content/drive/MyDrive/Research_Project/outputs/detectedPlanets.csv"

KOI_FILE = "/content/drive/MyDrive/Research_Project/datasets/Kepler_KOI_table.csv"
CONFIRMED_FILE = "/content/drive/MyDrive/Research_Project/datasets/Kepler_confirmed.csv"
FALSE_POS_FILE = "/content/drive/MyDrive/Research_Project/datasets/Kepler_false_positives.csv"
STELLAR_FILE = "/content/drive/MyDrive/Research_Project/datasets/Kepler_stellar.csv"
OUTPUT_FILE = "/content/drive/MyDrive/Research_Project/outputs/validated_candidates.csv"
# ================= LOAD DATA ==============================
print("Loading catalogs...")

tls_df = pd.read_csv(TLS_CANDIDATES_FILE, comment='#')

# Map star names to Kepler KIC IDs
STAR_NAME_TO_KIC = {
    "Kepler-11": "6541920"
}
tls_df["star"] = tls_df["star"].replace(STAR_NAME_TO_KIC)

koi_df = pd.read_csv(KOI_FILE, comment='#')
confirmed_df = pd.read_csv(CONFIRMED_FILE, comment='#')
fp_df = pd.read_csv(FALSE_POS_FILE, comment='#')
stellar_df = pd.read_csv(STELLAR_FILE, comment='#')

# Convert to string for safe matching
tls_df["star"] = tls_df["star"].astype(str)
stellar_df["kepid"] = stellar_df["kepid"].astype(str)
koi_df["kepid"] = koi_df["kepid"].astype(str)
confirmed_df["kepid"] = confirmed_df["kepid"].astype(str)
fp_df["kepid"] = fp_df["kepid"].astype(str)

print("Loaded TLS candidates:", len(tls_df))

# ================= PERIOD DUPLICATE FILTER ================
def remove_duplicate_periods(df, tolerance=0.01):

    df = df.sort_values("period_days")
    selected = []

    for _, row in df.iterrows():
        p = row["period_days"]

        if not any(abs(p - s["period_days"]) / s["period_days"] < tolerance for s in selected):
            selected.append(row)

    result = pd.DataFrame(selected)
    print("Final candidates after filtering:", len(result))
    return result

# ================= PERIOD CLUSTERING ======================
def cluster_periods(df, tolerance=0.01):

    df = df.sort_values("period_days")
    clusters = []

    for _, row in df.iterrows():
        p = row["period_days"]
        placed = False

        for cluster in clusters:
            cluster_periods = [r["period_days"] for r in cluster]

            if abs(p - np.mean(cluster_periods)) / np.mean(cluster_periods) < tolerance:
                cluster.append(row)
                placed = True
                break

        if not placed:
            clusters.append([row])

    representative_rows = []

    for cluster in clusters:
        best = max(cluster, key=lambda r: r["depth"] / r["period_days"])
        representative_rows.append(best)

    return pd.DataFrame(representative_rows)

# ================= MATCH KNOWN PLANETS ====================
def match_known_planet(row, catalog, tol=0.01):

    if "period" in catalog.columns:
        period_col = "period"
    elif "period_days" in catalog.columns:
        period_col = "period_days"
    else:
        return False

    matches = catalog[catalog["kepid"] == row["star"]]

    for _, m in matches.iterrows():
        if abs(row["period_days"] - m[period_col]) / m[period_col] < tol:
            return True

    return False

#================= Computing period error =================

def compute_period_error(row, catalog):
    if "period" not in catalog.columns:
        return np.nan

    matches = catalog[catalog["kepid"] == row["star"]]
    if len(matches) == 0:
        return np.nan

    diffs = abs(matches["period"] - row["period_days"]) / matches["period"]
    return diffs.min()

# ================= VALIDATION FUNCTION ====================
def validate_candidates(df):

    df = df.copy()

    # Safety check
    required_cols = ["star", "period_days", "transit_depth"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("Clustering similar period detections...")
    df = cluster_periods(df)

    #print("Removing duplicate period detections...")
    #df = remove_duplicate_periods(df)
    print("Skipping duplicate removal (handled in MCMC stage)")

    print("False positive filtering not applied in current pipeline")
    print("Future work will include cross-matching with false positive catalogs and centroid validation")

    print("Checking KOI catalog...")
    df["is_KOI"] = df["star"].isin(koi_df["kepid"])

    print("Checking confirmed planets...")
    df["is_confirmed"] = df.apply(lambda r: match_known_planet(r, confirmed_df), axis=1)

    print("Attaching stellar parameters...")

    stellar_sub = stellar_df[["kepid", "teff", "mass", "radius"]]
    stellar_sub = stellar_sub.drop_duplicates(subset=["kepid"])
    stellar_sub = stellar_sub.rename(columns={
        "mass": "stellar_mass",
        "radius": "stellar_radius"
    })

    df = df.merge(
        stellar_sub,
        left_on="star",
        right_on="kepid",
        how="left"
    )

    df.drop(columns=["kepid"], inplace=True)

    if df["teff"].isna().any():
        print("Warning: Missing stellar parameters for some stars")

    print("Flagging potential new candidates...")
    df["new_candidate"] = ~(df["is_KOI"] | df["is_confirmed"])

    recovered = df[df["is_confirmed"] == True]
    print("Recovered known planets:", len(recovered))

    df["period_error"] = df.apply(lambda r: compute_period_error(r, confirmed_df), axis=1)

    return df

# ================= RUN VALIDATION =========================
print("Starting validation...")

validated_df = validate_candidates(tls_df)

print("Saving results...")
validated_df.to_csv(OUTPUT_FILE, index=False)

print("Validation complete.")
print(validated_df.head())



print("\nValidation Summary:")
print(f"Total candidates: {len(validated_df)}")
print(f"Confirmed matches: {validated_df['is_confirmed'].sum()}")
print(f"New candidates: {validated_df['new_candidate'].sum()}")
