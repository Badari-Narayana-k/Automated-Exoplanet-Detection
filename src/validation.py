# ==========================================================
# validation.py — KEPLER CANDIDATE VALIDATION PIPELINE (v2)
#
# BUGS FIXED vs v1:
#   [CRITICAL] cluster_periods used r["depth"] → now r["transit_depth"]
#   [HIGH]     False positive catalog loaded but never used → now cross-matched
#   [MEDIUM]   compute_period_error had implicit global scope → explicit param
#   [MEDIUM]   stellar_sub kept arbitrary duplicate row → sorted by teff first
#   [MEDIUM]   Period tolerance 1% too tight for TTV systems → 2% (config)
# ==========================================================

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ==========================================================
# DATA LOADING
# ==========================================================

def load_catalogs():
    """Load all Kepler reference catalogs. Returns dict of DataFrames."""
    catalogs = {}
    catalog_files = {
        "koi":       config.KOI_FILE,
        "confirmed": config.CONFIRMED_FILE,
        "fp":        config.FALSE_POS_FILE,
        "stellar":   config.STELLAR_FILE,
    }

    for name, path in catalog_files.items():
        try:
            df = pd.read_csv(path, comment="#")
            # Ensure kepid is string for safe matching
            if "kepid" in df.columns:
                df["kepid"] = df["kepid"].astype(str).str.strip()
            catalogs[name] = df
            log.info(f"Loaded {name}: {len(df)} rows")
        except FileNotFoundError:
            log.warning(f"Catalog not found: {path} — skipping {name}")
            catalogs[name] = pd.DataFrame()

    return catalogs


def load_candidates(path=config.CATALOG_FILE):
    df = pd.read_csv(path, comment="#")
    df["star"] = df["star"].replace(config.STAR_NAME_TO_KIC)
    df["star"] = df["star"].astype(str).str.strip()
    log.info(f"Loaded {len(df)} TLS candidates")
    return df


# ==========================================================
# PERIOD CLUSTERING — removes near-identical detections
# ==========================================================

def cluster_periods(df, tolerance=config.PERIOD_TOLERANCE):
    """
    Group detections with similar periods into clusters; keep strongest signal.

    BUG FIX: was using r["depth"] → column is "transit_depth" in output CSV.
    Strongest = highest SDE (detection significance), not depth/period ratio.
    """
    if df.empty:
        return df

    df = df.sort_values("period_days").reset_index(drop=True)
    clusters = []

    for _, row in df.iterrows():
        p = row["period_days"]
        placed = False
        for cluster in clusters:
            mean_p = np.mean([r["period_days"] for r in cluster])
            if abs(p - mean_p) / mean_p < tolerance:
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])

    # Pick best representative per cluster: highest SDE (or depth if SDE absent)
    representatives = []
    for cluster in clusters:
        if "SDE" in cluster[0].index and not pd.isna(cluster[0]["SDE"]):
            best = max(cluster, key=lambda r: r["SDE"])
        else:
            best = max(cluster, key=lambda r: r["transit_depth"])   # FIX: was r["depth"]
        representatives.append(best)

    result = pd.DataFrame(representatives).reset_index(drop=True)
    log.info(f"Period clustering: {len(df)} → {len(result)} candidates")
    return result


# ==========================================================
# CATALOG CROSS-MATCHING
# ==========================================================

def match_known_planet(row, catalog, tol=config.PERIOD_TOLERANCE):
    """Check if (star, period) matches any entry in a Kepler catalog."""
    if catalog.empty or "kepid" not in catalog.columns:
        return False

    period_col = next(
        (c for c in ["period", "koi_period", "pl_orbper", "period_days"] if c in catalog.columns),
        None
    )
    if period_col is None:
        return False

    matches = catalog[catalog["kepid"] == row["star"]]
    if matches.empty:
        return False

    for _, m in matches.iterrows():
        try:
            ref_p = float(m[period_col])
            if ref_p > 0 and abs(row["period_days"] - ref_p) / ref_p < tol:
                return True
        except (ValueError, TypeError):
            continue
    return False


def match_false_positive(row, fp_catalog, tol=config.PERIOD_TOLERANCE):
    """Check if candidate matches a known false positive."""
    return match_known_planet(row, fp_catalog, tol=tol)


def compute_period_error(row, catalog):
    """
    BUG FIX: catalog now passed explicitly (was implicit global).
    Returns fractional period error vs closest match in catalog.
    """
    period_col = next(
        (c for c in ["period", "koi_period", "pl_orbper"] if c in catalog.columns),
        None
    )
    if period_col is None or catalog.empty:
        return np.nan

    matches = catalog[catalog["kepid"] == row["star"]]
    if matches.empty:
        return np.nan

    diffs = []
    for _, m in matches.iterrows():
        try:
            ref_p = float(m[period_col])
            if ref_p > 0:
                diffs.append(abs(row["period_days"] - ref_p) / ref_p)
        except (ValueError, TypeError):
            continue
    return min(diffs) if diffs else np.nan


# ==========================================================
# STELLAR PARAMETER ATTACHMENT
# ==========================================================

def attach_stellar_params(df, stellar_df):
    """
    BUG FIX: sort by teff (not-null) before drop_duplicates so we keep
    the highest-quality row when a star has multiple stellar entries.
    """
    if stellar_df.empty:
        log.warning("Stellar catalog empty — skipping parameter attachment")
        df[["teff", "stellar_mass", "stellar_radius"]] = np.nan
        return df

    cols_needed = [c for c in ["kepid", "teff", "mass", "radius"] if c in stellar_df.columns]
    stellar_sub = stellar_df[cols_needed].copy()

    if "teff" in stellar_sub.columns:
        stellar_sub = stellar_sub.sort_values("teff", ascending=False, na_position="last")

    stellar_sub = stellar_sub.drop_duplicates(subset=["kepid"], keep="first")
    stellar_sub = stellar_sub.rename(columns={"mass": "stellar_mass", "radius": "stellar_radius"})

    df = df.merge(stellar_sub, left_on="star", right_on="kepid", how="left")
    if "kepid" in df.columns:
        df.drop(columns=["kepid"], inplace=True)

    missing = df["teff"].isna().sum() if "teff" in df.columns else len(df)
    if missing > 0:
        log.warning(f"{missing} candidates missing stellar parameters")
    return df


# ==========================================================
# VALIDATION SUMMARY STATS
# ==========================================================

def validation_summary(df):
    print("\n" + "="*50)
    print("  VALIDATION SUMMARY")
    print("="*50)
    print(f"  Total candidates:        {len(df)}")
    if "is_KOI" in df.columns:
        print(f"  KOI matches:             {df['is_KOI'].sum()}")
    if "is_confirmed" in df.columns:
        print(f"  Confirmed planet matches:{df['is_confirmed'].sum()}")
    if "is_false_positive" in df.columns:
        print(f"  False positive flags:    {df['is_false_positive'].sum()}")
    if "new_candidate" in df.columns:
        print(f"  New candidates:          {df['new_candidate'].sum()}")
    if "period_error" in df.columns and not df["period_error"].isna().all():
        print(f"  Median period error:     {df['period_error'].dropna().median()*100:.3f}%")
    print("="*50 + "\n")


# ==========================================================
# VISUALIZATION
# ==========================================================

def plot_validation_overview(df, save=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Period distribution
    axes[0].hist(df["period_days"], bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Period (days)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Period Distribution")

    # Radius vs Orbital Distance
    if "radius_RE" in df.columns and "orbit_AU" in df.columns:
        colors = df.get("is_confirmed", pd.Series([False]*len(df))).map(
            {True: "gold", False: "steelblue"}
        )
        axes[1].scatter(df["orbit_AU"], df["radius_RE"], c=colors, s=40, alpha=0.8)
        axes[1].set_xlabel("Orbital distance (AU)")
        axes[1].set_ylabel("Planet radius (R⊕)")
        axes[1].set_xscale("log")
        axes[1].set_title("Planet Radius–Orbit Space")

    # SDE histogram
    if "SDE" in df.columns:
        axes[2].hist(df["SDE"].dropna(), bins=15, color="coral", edgecolor="white")
        axes[2].axvline(config.TLS_SDE_THRESHOLD, color="red", ls="--", label="SDE threshold")
        axes[2].set_xlabel("SDE")
        axes[2].set_title("Detection Significance (SDE)")
        axes[2].legend()

    plt.suptitle("Validation Overview", fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {save}")
    plt.close()


# ==========================================================
# MAIN VALIDATION FUNCTION
# ==========================================================

def validate_candidates(df, catalogs):
    """
    Full validation pipeline.

    Parameters
    ----------
    df       : DataFrame of TLS/BLS candidates from exoplanets.py
    catalogs : dict from load_catalogs()
    """
    df = df.copy()

    required = ["star", "period_days", "transit_depth"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    log.info("Step 1/6: Clustering similar period detections...")
    df = cluster_periods(df)

    log.info("Step 2/6: Cross-matching KOI catalog...")
    df["is_KOI"] = df["star"].isin(catalogs["koi"].get("kepid", pd.Series(dtype=str)))

    log.info("Step 3/6: Matching confirmed planets...")
    df["is_confirmed"] = df.apply(
        lambda r: match_known_planet(r, catalogs["confirmed"]), axis=1
    )

    log.info("Step 4/6: Flagging false positives...")   # BUG FIX: now actually used
    df["is_false_positive"] = df.apply(
        lambda r: match_false_positive(r, catalogs["fp"]), axis=1
    )

    log.info("Step 5/6: Attaching stellar parameters...")
    df = attach_stellar_params(df, catalogs["stellar"])

    log.info("Step 6/6: Flagging new candidates...")
    df["new_candidate"] = ~(df["is_KOI"] | df["is_confirmed"] | df["is_false_positive"])

    log.info("Computing period errors for confirmed matches...")
    df["period_error"] = df.apply(
        lambda r: compute_period_error(r, catalogs["confirmed"]),   # FIX: explicit param
        axis=1
    )

    validation_summary(df)
    return df


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    tls_df    = load_candidates()
    catalogs  = load_catalogs()

    validated = validate_candidates(tls_df, catalogs)

    plot_validation_overview(
        validated,
        save=f"{config.PLOTS_DIR}/validation_overview.png"
    )

    validated.to_csv(config.VALIDATED_FILE, index=False)
    log.info(f"Validated candidates saved: {config.VALIDATED_FILE}")
