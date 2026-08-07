"""
load_real_data.py — Load and stitch real Kepler-11 long-cadence FITS quarters
downloaded from MAST via lightkurve (Q1-Q17), replacing the original pipeline's
generate_synthetic_lc().

IMPORTANT (see docs/BUGS_AND_FINDINGS.md): each Kepler quarter has a different
instrumental flux baseline (measured jumps of 12-648 ppm at quarter boundaries
in this dataset). A single global detrending filter applied across the
stitched 4-year baseline -- fine for continuous synthetic data -- cannot
correct these step discontinuities and produced a spurious non-physical
detection in an early version of this analysis. This module detrends each
quarter independently before stitching.
"""
import glob
import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter


def load_kepler11_real(fits_dir="data/kepler11_llc"):
    files = sorted(glob.glob(f"{fits_dir}/*.fits"))

    all_time, all_flux = [], []
    quarter_report = []

    for f in files:
        d = fits.getdata(f, 1)
        h0 = fits.getheader(f, 0)
        t = np.asarray(d["TIME"], dtype=float)
        flux = np.asarray(d["FLUX"], dtype=float)
        quality = np.asarray(d["SAP_QUALITY"]) if "SAP_QUALITY" in d.columns.names else np.zeros(len(t))

        good = np.isfinite(t) & np.isfinite(flux) & (quality == 0)
        t, flux = t[good], flux[good]
        if len(flux) == 0:
            continue

        # Per-quarter detrend: median-normalise, then remove the quarter's own
        # long-term trend with Savitzky-Golay before stitching. Required for
        # real multi-quarter data -- see module docstring.
        med = np.nanmedian(flux)
        flux_norm = flux / med
        wl = min(401, len(flux_norm) - 1)
        if wl % 2 == 0:
            wl -= 1
        if wl >= 5:
            trend = savgol_filter(flux_norm, wl, 3)
            flux_norm = flux_norm / trend

        quarter_report.append({
            "file": f.split("/")[-1],
            "quarter": h0.get("QUARTER"),
            "n_pts": len(flux),
            "t_min": t.min(), "t_max": t.max(),
            "median_raw_flux": med,
        })

        all_time.append(t)
        all_flux.append(flux_norm)

    time = np.concatenate(all_time)
    flux = np.concatenate(all_flux)

    order = np.argsort(time)
    time, flux = time[order], flux[order]
    time = time - time[0]

    return time, flux, quarter_report


if __name__ == "__main__":
    time, flux, report = load_kepler11_real()
    print(f"Stitched {len(report)} quarters, {len(time)} total cadences")
    print(f"Baseline: {time[-1] - time[0]:.2f} days")
    for r in report:
        print(r)
