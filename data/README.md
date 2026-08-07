# Data

Raw Kepler FITS files are not committed to this repo (too large, and
redistributing Kepler data directly is unnecessary when MAST hosts it).

## To reproduce

Run `download_kepler11_llc_only.sh` on a machine with normal internet access
(MAST is not reachable from some sandboxed/restricted environments — see
`docs/BUGS_AND_FINDINGS.md` for why that mattered during this analysis). It
downloads the 17 long-cadence quarters (Q1-Q17) for Kepler-11 (KIC 6541920)
directly from MAST.

Alternatively, via `lightkurve`:

```python
import lightkurve as lk
search = lk.search_lightcurve("KIC 6541920", mission="Kepler", cadence="long")
lc_collection = search.download_all(download_dir="./kplr_raw")
```

Using `download_all()` (not `.to_fits()`) preserves the original MAST FITS
headers, including `CROWDSAP`/`FLFRCSAP` — these were needed later to test
(and rule out) aperture dilution as an explanation for a depth anomaly; see
`docs/BUGS_AND_FINDINGS.md`.

Place the resulting FITS files in `data/kepler11_llc/` before running
`src/load_real_data.py`.
