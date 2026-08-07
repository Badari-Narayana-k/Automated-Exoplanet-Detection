# ==========================================================
# config.py — pipeline configuration
# Values match the methodology described in the paper
# (Savitzky-Golay window ~401 cadences, 5-sigma clipping,
#  period grid 0.5-300 d, SDE threshold 7, 2% period tolerance).
# ==========================================================

import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR   = os.path.join(BASE_DIR, "results", "plots")
OUTPUTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR   = os.path.join(BASE_DIR, "data", "cache")

for d in (PLOTS_DIR, OUTPUTS_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)

# ---- Preprocessing ----
WINDOW_LENGTH = 401      # Savitzky-Golay window, cadences (~8.3 d at LC)
OUTLIER_SIGMA = 5
BIN_SIZE_DAYS = None

# ---- Period search ----
MIN_PERIOD = 0.5
MAX_PERIOD = 300.0
TLS_OVERSAMPLING = 3
MAX_PLANETS = 6
TLS_SDE_THRESHOLD = 7.0
PERIOD_TOLERANCE = 0.02   # 2%, widened for TTV systems per paper
TRANSIT_PADDING = 0.15

# ---- Habitable zone (Kopparapu+2013 approx, conservative) ----
HZ_INNER_COEFF = 1.10
HZ_OUTER_COEFF = 0.53

# ---- Catalog files (single-target validation; see docs/BUGS_AND_FINDINGS.md) ----
KOI_FILE       = os.path.join(BASE_DIR, "data", "koi_catalog.csv")
CONFIRMED_FILE = os.path.join(BASE_DIR, "data", "confirmed_catalog.csv")
FALSE_POS_FILE = os.path.join(BASE_DIR, "data", "fp_catalog.csv")
STELLAR_FILE   = os.path.join(BASE_DIR, "data", "stellar_catalog.csv")
CATALOG_FILE   = os.path.join(OUTPUTS_DIR, "detectedPlanets_real.csv")
VALIDATED_FILE = os.path.join(OUTPUTS_DIR, "validatedCandidates_real.csv")

STAR_NAME_TO_KIC = {"Kepler-11": "6541920"}

# ---- MCMC ----
MCMC_N_STEPS       = 2000
MCMC_BURN_IN       = 500
MCMC_RESULTS_FILE  = os.path.join(OUTPUTS_DIR, "mcmcResults_real.csv")
