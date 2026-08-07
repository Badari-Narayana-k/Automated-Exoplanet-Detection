# ============================================================
# BLIND multi-planet search on real Kepler-11 photometry (LOCAL version).
# Unlike the paper's Section 3.1 (targeted +/-5% windows around
# already-known periods), this scans the FULL 0.5-300 day grid
# with no prior knowledge of where a planet's period sits - the
# task the original methods section actually describes.
#
# Run locally: pip install lightkurve transitleastsquares astropy scipy
# then: python local_blind_search.py
# Speed depends on your CPU's core count (check with `nproc` on
# Linux, `sysctl -n hw.ncpu` on Mac, or Task Manager on Windows) -
# transitleastsquares uses multiprocessing.Pool internally and
# scales roughly with core count, not with GPU.
# ============================================================

# --- Cell 1: installs ---
# Run this in your terminal first, not in the script:
#   pip install lightkurve transitleastsquares astropy


# --- Cell 2: download real Kepler-11 data (Q1-Q17 long cadence) ---
import lightkurve as lk
import numpy as np

search = lk.search_lightcurve("KIC 6541920", mission="Kepler", cadence="long")
print(search)  # sanity check: should list ~17 quarters
lc_collection = search.download_all()
print(f"Downloaded {len(lc_collection)} quarters")

# --- Cell 3: per-quarter detrend and stitch (same method as the paper) ---
from scipy.signal import savgol_filter

all_time, all_flux = [], []
for lc in lc_collection:
    t = np.asarray(lc.time.value, dtype=float)
    f = np.asarray(lc.flux.value, dtype=float)
    quality = np.asarray(lc.quality.value)

    good = np.isfinite(t) & np.isfinite(f) & (quality == 0)
    t, f = t[good], f[good]
    if len(f) == 0:
        continue

    med = np.nanmedian(f)
    f_norm = f / med

    wl = min(401, len(f_norm) - 1)
    if wl % 2 == 0:
        wl -= 1
    if wl >= 5:
        trend = savgol_filter(f_norm, wl, 3)
        f_norm = f_norm / trend

    all_time.append(t)
    all_flux.append(f_norm)

time = np.concatenate(all_time)
flux = np.concatenate(all_flux)
order = np.argsort(time)
time, flux = time[order], flux[order]
time = time - time[0]

# global 5-sigma clip
med, std = np.median(flux), np.std(flux)
keep = np.abs(flux - med) < 5 * std
time, flux = time[keep], flux[keep]

print(f"Final light curve: {len(time)} points, baseline {time[-1]-time[0]:.1f} days")
print(f"Residual scatter: {np.std(flux)*1e6:.0f} ppm")

# --- Cell 4: BLIND iterative multi-planet search ---
# This is the real, unconstrained search: full 0.5-300 day period
# grid, no literature period given, standard iterative peel-off.
# SAVE PROGRESS AFTER EACH PLANET so a Colab disconnect doesn't
# lose everything.

from transitleastsquares import transitleastsquares, transit_mask
import json, time as timer

SDE_THRESHOLD = 7.0
MAX_PLANETS = 6
PERIOD_MIN = 0.5
PERIOD_MAX = 300.0

residual_time = time.copy()
residual_flux = flux.copy()
found_planets = []

for iteration in range(1, MAX_PLANETS + 1):
    print(f"\n=== Iteration {iteration}: searching {PERIOD_MIN}-{PERIOD_MAX} d ===")
    t0 = timer.time()

    model = transitleastsquares(residual_time, residual_flux)
    r = model.power(
        period_min=PERIOD_MIN,
        period_max=PERIOD_MAX,
        oversampling_factor=3,       # raised from Colab's 2 - your 14 threads can absorb it
        duration_grid_step=1.15,     # raise (coarser) if too slow, lower for precision
        use_threads=14,              # you have 16 logical cores; leaving 2 free for the OS
        show_progress_bar=True,
    )

    elapsed = timer.time() - t0
    print(f"Best period: {r.period:.5f} d, SDE={r.SDE:.2f}, "
          f"depth={((1-r.depth)*1e6):.0f} ppm, elapsed={elapsed/60:.1f} min")

    if r.SDE < SDE_THRESHOLD:
        print(f"SDE {r.SDE:.2f} < threshold {SDE_THRESHOLD}, stopping.")
        break

    # check for duplicate period (within 2% of an already-found planet)
    is_dup = any(abs(r.period - p["period"]) / p["period"] < 0.02 for p in found_planets)
    if is_dup:
        print("Duplicate period detected, masking and continuing without recording.")
    else:
        found_planets.append({
            "iteration": iteration,
            "period": r.period,
            "T0": r.T0,
            "duration": r.duration,
            "depth_ppm": float((1 - r.depth) * 1e6),
            "SDE": r.SDE,
            "odd_even_mismatch": r.odd_even_mismatch,
            "elapsed_min": elapsed / 60,
        })
        # save progress immediately
        with open("blind_search_results.json", "w") as fh:
            json.dump(found_planets, fh, indent=2, default=float)
        print(f"Saved. {len(found_planets)} planet(s) found so far.")

    # mask this transit out of the residual and continue
    mask = transit_mask(residual_time, r.period, r.duration, r.T0)
    residual_flux = residual_flux[~mask]
    residual_time = residual_time[~mask]

print("\n=== DONE ===")
for p in found_planets:
    print(p)

# --- Cell 5: package results for upload ---

print("blind_search_results.json is ready in this folder - upload it to me directly, no need to zip.")
