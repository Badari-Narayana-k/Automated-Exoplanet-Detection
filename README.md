# Kepler-11 Multi-Planet Detection Pipeline: Real-Data Validation

An open-source pipeline for multi-planet transit detection and joint
Bayesian characterisation in Kepler long-cadence photometry, validated
against **real MAST-hosted Kepler-11 photometry** (KIC 6541920, Quarters
1–17, 58,785 cleaned cadences, 1459.5-day baseline) — not just the
synthetic benchmark the project started with.

**This repo's commit history is itself part of the documentation.** Rather
than squashing everything into one commit, it's structured to show the
actual sequence: original synthetic-only pipeline → real data ingestion →
two real bugs found and fixed → a wrong conclusion drawn, then retracted
once a blind search disagreed with it → an anomaly investigated and
honestly left open. If you only read one thing before the code, read
[`docs/BUGS_AND_FINDINGS.md`](docs/BUGS_AND_FINDINGS.md).

## Headline results

- **Two functional bugs found and fixed**, neither visible on synthetic
  data: a fake TLS detector that returned a confident false positive on
  real data, and an MCMC sampler with 0% acceptance on the real dataset.
- **Targeted search**: all 6 known Kepler-11 planets recovered, 5/6 to
  <0.005% period accuracy.
- **Blind search** (no period given in advance): 5/6 planets recovered at
  SDE 78–226. In the process, revealed that the targeted search's
  Kepler-11f result was wrong — retracted explicitly, not silently fixed.
- **A depth-shortfall anomaly** investigated and two hypotheses (aperture
  dilution, revised stellar parameters) ruled out with real data; the
  actual cause is reported as still open.

Full details, numbers, and the paper: [`paper/paper.pdf`](paper/paper.pdf).

## Repo structure

```
src/
  exoplanets.py          preprocessing, BLS, iterative TLS peel-off (original)
  jointMultiplanet.py    joint MCMC characterisation (original)
  validation.py          catalog cross-matching (original)
  config.py               pipeline configuration
  load_real_data.py       real MAST data ingestion + per-quarter detrending
  targeted_search.py      targeted TLS search (bug-fixed) + results
  run_real_mcmc.py         joint MCMC fit (bug-fixed, adaptive step-size)
  check_depth_shortfall.py investigates the depth anomaly (CROWDSAP check)
  blind_search.py          full blind period search, no prior knowledge

data/
  README.md                how to get the real Kepler-11 FITS data
  download_kepler11_llc_only.sh

results/
  real_tls_targeted_results.json
  mcmcResults_real.csv
  blind_search_results.json
  crowdsap_by_quarter.csv

paper/
  paper.tex, paper.pdf     LaTeX manuscript + compiled PDF
  figures/

docs/
  BUGS_AND_FINDINGS.md     full diagnostic writeup — start here
```

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Get real Kepler-11 data (see data/README.md)
bash data/download_kepler11_llc_only.sh
# place resulting FITS files in data/kepler11_llc/

# 2. Targeted search
python src/targeted_search.py

# 3. Joint MCMC fit
python src/run_real_mcmc.py

# 4. Blind search (slow — ~2hrs on 14 cores; see docs for timing notes)
python src/blind_search.py

# 5. Depth-shortfall check (needs original MAST headers, not lightkurve's
#    stripped .to_fits() export — see data/README.md)
python src/check_depth_shortfall.py
```

## What's still open

- The depth-shortfall cause (Finding 5 in `docs/BUGS_AND_FINDINGS.md`).
- The SDE-normalisation artefact behind the Kepler-11f mistake has not been
  systematically characterized — how narrow a window triggers it, whether
  it scales with SNR — flagged as a natural standalone follow-up.
- Single-target validation (n=1 star). Whether these findings generalize to
  other multi-planet systems is untested.

## License

MIT — see [LICENSE](LICENSE).
