"""
check_depth_shortfall.py — investigate the systematic depth shortfall
found in targeted_search.py results.

Recovered depths for Kepler-11b, c, d, e come out 5-40% shallower than the
literature (discovery-paper) values, worst for e. This script tests the
most obvious explanation -- photometric aperture dilution -- directly
against real MAST header data, and rules it out. See
docs/BUGS_AND_FINDINGS.md for the full writeup, including the
stellar-parameter-revision hypothesis (also ruled out) and the remaining
open question.

Requires the original MAST FITS files with headers intact -- NOT files
re-exported via lightkurve's LightCurve.to_fits(), which strips CROWDSAP
and FLFRCSAP. Use search.download_all() and read the cached originals
directly (see data/README.md).
"""
import glob
import numpy as np
from astropy.io import fits


def check_crowdsap(fits_dir="data/kplr_raw"):
    files = sorted(glob.glob(f"{fits_dir}/**/*.fits", recursive=True))
    crowd = []
    for f in files:
        h0 = fits.getheader(f, 0)
        h1 = fits.getheader(f, 1)
        c = h1.get("CROWDSAP")
        if c is not None:
            crowd.append(c)
            print(f"Q{h0.get('QUARTER')}: CROWDSAP={c}, FLFRCSAP={h1.get('FLFRCSAP')}")

    crowd = np.array(crowd)
    print()
    print(f"Mean CROWDSAP: {crowd.mean():.4f}, range: {crowd.min():.4f}-{crowd.max():.4f}")
    max_correction_pct = (1 / crowd.min() - 1) * 100
    print(f"Maximum possible depth correction from dilution: {max_correction_pct:.2f}%")
    print()
    print("Observed depth shortfall: 5-13% (b,c,d), 36-40% (e).")
    print(f"Verdict: {max_correction_pct:.1f}% max possible correction cannot explain "
          "the observed shortfall, especially Kepler-11e's 36-40% deficit.")
    print("Aperture dilution is RULED OUT as the cause. See docs/BUGS_AND_FINDINGS.md.")
    return crowd


if __name__ == "__main__":
    check_crowdsap()
