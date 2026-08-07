"""
targeted_search.py — targeted period-search validation using the REAL
transitleastsquares package.

BUG FIX (see docs/BUGS_AND_FINDINGS.md #1): the original codebase's
detect_multiple_planets() called a hand-written matched-filter proxy, not
the transitleastsquares package the paper's methods section describes. On
synthetic data this proxy performed adequately. On real data it returned a
confident, non-physical false positive (period=198.3d, depth~5ppm, SDE>7),
because its detection statistic normalises by local_std/sqrt(n) but only
requires n>=2 in-transit points to accept a candidate. This script replaces
it with the actual transitleastsquares package.

This performs a TARGETED search (+/-5% window around each literature
period) -- not a blind search. See blind_search.py for that, and
docs/BUGS_AND_FINDINGS.md for why the distinction matters.
"""
import sys, time as timer, json, os
sys.path.insert(0, os.path.dirname(__file__))
from load_real_data import load_kepler11_real
import exoplanets as ex
from transitleastsquares import transitleastsquares

KNOWN = {
    "b": {"name": "Kepler-11b", "period": 10.30375, "depth_ppm": 310},
    "c": {"name": "Kepler-11c", "period": 13.02502, "depth_ppm": 790},
    "d": {"name": "Kepler-11d", "period": 22.68719, "depth_ppm": 940},
    "e": {"name": "Kepler-11e", "period": 31.9959,  "depth_ppm": 1630},
    "f": {"name": "Kepler-11f", "period": 46.68876, "depth_ppm": 540},
    "g": {"name": "Kepler-11g", "period": 118.37743,"depth_ppm": 890},
}


def search_one_planet(key, t, f):
    planet = KNOWN[key]
    p0 = planet["period"]

    t0 = timer.time()
    model = transitleastsquares(t, f)
    r = model.power(period_min=p0 * 0.95, period_max=p0 * 1.05,
                     oversampling_factor=3, duration_grid_step=1.1,
                     show_progress_bar=False)
    elapsed = timer.time() - t0

    return {
        "name": planet["name"],
        "period_truth": p0,
        "depth_truth_ppm": planet["depth_ppm"],
        "n_transits_possible": round((t[-1] - t[0]) / p0, 1),
        "period_found": r.period,
        "period_err_pct": abs(r.period - p0) / p0 * 100,
        "SDE": r.SDE,
        "depth_found_ppm": float((1 - r.depth) * 1e6),
        "duration_found_hr": r.duration * 24,
        "T0_found": r.T0,
        "odd_even_mismatch": r.odd_even_mismatch,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    t, f, report = load_kepler11_real()
    t, f = ex.clean_and_flatten(t, f)
    print(f"Cleaned light curve: {len(t)} points, baseline {t[-1]-t[0]:.1f} d")

    results = [search_one_planet(k, t, f) for k in KNOWN]

    out_path = "results/real_tls_targeted_results.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    for r in results:
        print(f"{r['name']}: P_true={r['period_truth']:.5f}d P_found={r['period_found']:.5f}d "
              f"err={r['period_err_pct']:.4f}% SDE={r['SDE']:.2f} depth_found={r['depth_found_ppm']:.0f}ppm")

    return results


if __name__ == "__main__":
    main()
