# Bugs Found and Findings — Full Writeup

This document exists because a paper's Discussion section compresses this
story into a few paragraphs; the full diagnostic trail is more useful as a
standalone reference, both for anyone hitting the same bugs and as a record
of how each conclusion was actually reached.

## Bug 1: the native TLS detector was a false-positive-prone proxy

**File:** `src/targeted_search.py` (fix), original bug was in the pipeline's
`exoplanets.py :: run_tls_native()`.

The original codebase's iterative multi-planet search called a hand-written
matched-filter proxy, not the `transitleastsquares` package the paper's
methods section describes using. On synthetic data (clean Gaussian noise,
no instrumental systematics) this proxy performed adequately — it happened
to agree with the paper's synthetic-validation claims.

On real data it failed outright: the proxy's detection statistic normalises
by `local_std / sqrt(n)`, but only requires `n >= 2` in-transit points to
accept a candidate. At long trial periods with sparse phase coverage, pure
noise produces a spuriously high "SDE" (we saw 9.7–14.3) for a signal with
negligible actual depth (~5 ppm — not a real transit shape at all). This
specific failure mode cannot occur with idealized synthetic noise at any
meaningful rate, which is why the original synthetic-only validation never
caught it.

**Fix:** use the actual `transitleastsquares` package.

## Bug 2: the MCMC sampler had 0% acceptance on real data

**File:** `src/run_real_mcmc.py` (fix), original bug in `jointMultiplanet.py`.

The original single-chain Metropolis–Hastings sampler initialises its
proposal step size as `|theta_0| * 0.005`. On the real, full 58,785-point,
four-planet dataset — log-posterior magnitude of order 4×10⁵, reflecting
up to 142 observed transits per planet, far more information than the
synthetic case implicitly assumed — this scale is orders of magnitude too
large. The first real-data run returned **0% acceptance across all 2000
steps**: the chain never moved from its starting point.

**Fix:** a two-stage adaptive scheme — a tuning phase (2500 iterations)
adjusting each parameter's step size individually toward a 25–50% target
acceptance band, followed by production sampling (1500 steps, 500 burn-in)
at the tuned scale. Final acceptance rates: 31–45%.

A side effect worth noting: once working, the real posterior period
precisions came out roughly 50× tighter than the pipeline's original
synthetic-data claims (tens of seconds vs. ~0.03–0.09 days), simply because
the real 4-year baseline contains far more transits per planet than the
synthetic benchmark assumed.

## Finding 3: the Kepler-11f "TTV explanation" was wrong (retracted)

**Files:** `src/targeted_search.py` (produced the wrong claim),
`src/blind_search.py` (produced the correction).

The targeted search (±5% window around the literature period) placed
Kepler-11f's period at 45.375 d against a literature value of 46.689 d — a
2.81% discrepancy, independently reproduced by an `astropy` BoxLeastSquares
cross-check. This was attributed to unmodelled transit-timing variations,
supported by two real, independently verified citations: Ford et al. (2012)
flags Kepler-11f as one of only 39 Kepler-wide systems with a documented
long-term secular TTV trend, and Lissauer et al. (2011)'s own discovery
paper reports Kepler-11f's TTV mass detection as the weakest (2σ) of the
three it measured directly.

**Both of those facts are true. The conclusion built on them was wrong.**

A subsequent full blind search (the entire 0.5–300 day grid, no period
given in advance) finds Kepler-11f at **46.6857 d — 0.0066% from the
literature value, at SDE=192** (vs. SDE=15.7 for the targeted search's
wrong answer). A signal recovered at 12× higher significance and ~400×
better period accuracy is the correct detection.

**Root cause (best current understanding, not yet independently confirmed):**
`transitleastsquares`' SDE statistic is normalised against the mean/std of
power computed *only over the searched period grid*. A narrow ±5% window
has different noise statistics than the full grid — it is plausible that
within that narrow window, a spurious local peak had higher relative SDE
than the true signal, while the full grid correctly identifies the global
maximum. **This is a real, general methodological finding about narrow-window
period searches, not specific to Kepler-11f** — see "Open question" below.

We are retracting the original claim explicitly, in the same place it was
made, rather than silently revising it. The TTV literature cited remains
factually accurate about Kepler-11f's real TTV history; it was simply the
wrong explanation for this particular number.

## Finding 4: Kepler-11b was not recovered by blind search (expected)

Kepler-11b — the shallowest planet in the system (310 ppm literature
depth) — was the only planet not recovered by the blind search, despite
being recovered by the targeted search (SDE=16.0, already the second-lowest
in that table). This is consistent with the **look-elsewhere effect**: a
blind search over the full 0.5–300 day grid tests orders of magnitude more
trial periods than a narrow ±5% window, so the same physical signal needs
correspondingly higher peak power to clear the same nominal SDE≥7
threshold. Not a defect — an expected sensitivity-floor effect, worth
knowing about if you're running this kind of search near a detection
threshold.

## Finding 5: systematic depth shortfall — cause still open

Recovered transit depths for Kepler-11b, c, d, and e come out systematically
shallower than literature values: 5–13% for b/c/d, 36–40% for e — consistent
across targeted TLS, targeted BLS, and the blind search. **Kepler-11g does
not share this pattern** (+1.8% to +7.1%, the opposite direction).

Two hypotheses were tested directly and ruled out:

- **Aperture dilution** — real `CROWDSAP` values recovered from the original
  MAST headers (0.948–0.995 across all 17 quarters) permit a maximum 5.5%
  depth correction, an order of magnitude too small to explain the deficit.
  Separately, dilution is a per-star property and should affect all six
  planets similarly; Kepler-11g's exception is independent evidence against
  it even before the CROWDSAP numbers are considered.
- **Revised stellar parameters** (Bedell et al. 2017) — revises the derived
  *planet radius*, not the measured *transit depth in ppm*, so it cannot
  explain a photometric-depth discrepancy regardless of which stellar
  radius is assumed.

**This remains genuinely unresolved.** The one untested hypothesis with a
plausible mechanism is TTV-induced phase-smearing in a period-folded search
that doesn't account for timing variations — but a duration-inflation check
(MCMC-fitted durations are 75–95% inflated for b/c/d but only 15% for e,
the *opposite* ranking from the depth deficit) argues against a single
clean mechanism. See paper Section 3.6 for the full account, including what
this check does and doesn't establish.

## Open question worth its own follow-up: the SDE-normalisation artefact

Finding 3 above revealed something more general than a Kepler-11f-specific
mistake: a narrow-window `transitleastsquares` search can return a
confidently wrong answer (SDE=15.7, well above the conventional SDE≥7
threshold) rather than a low-confidence one. This means **SDE alone is not
a reliable indicator of correctness when the search window is narrow** —
a real, characterizable failure mode of a widely-used package that other
targeted-search users could hit without knowing it.

This hasn't been systematically characterized yet: how narrow does a window
have to be before this appears, does it scale with SNR or transit depth,
and can it be detected automatically (e.g., by comparing a targeted result
against a wider sanity-check window before trusting it)? This is flagged as
a natural standalone follow-up, independent of the rest of this project.
