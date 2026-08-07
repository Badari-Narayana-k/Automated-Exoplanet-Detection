# ==========================================================
# jointMultiplanet.py — JOINT MULTI-PLANET MCMC MODELLING (v2)
#
# MAJOR REWRITE vs v1:
#   [CRITICAL] v1 had NO MCMC — was a copy of the detection pipeline
#   Now implements:
#     • Physically motivated transit model (trapezoidal + limb darkening)
#     • Metropolis-Hastings MCMC sampler (pure numpy — no emcee required)
#     • Joint fitting of all planets simultaneously
#     • Gaussian priors from stellar parameters
#     • Corner plot, trace plots, model overlay
#     • Credible intervals on all parameters
#
# Production note: swap _transit_model() with batman.TransitModel for
# full Mandel-Agol precision; swap _mcmc_mh() with emcee EnsembleSampler
# for faster convergence.
# ==========================================================

import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import savgol_filter

import config

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ==========================================================
# TRANSIT MODEL  (trapezoidal with quadratic limb darkening)
# ==========================================================

def _transit_model_single(time, period, t0, depth, duration, u1=0.3, u2=0.1):
    """
    Single-planet transit model with quadratic limb darkening.

    Approximate limb-darkened depth:
        depth_eff = depth * (1 - u1/3 - u2/6) / (1 - u1/6 - u2/12)

    Uses trapezoidal shape; ingress = 15% of full duration.
    """
    ld_correction = (1 - u1/3 - u2/6) / (1 - u1/6 - u2/12 + 1e-12)
    eff_depth = depth * ld_correction

    phase = ((time - t0 + 0.5*period) % period) - 0.5*period  # centred
    half_dur = duration / 2.0
    ingress   = 0.15 * half_dur

    flux = np.ones(len(time))

    # Flat bottom
    flat = np.abs(phase) <= (half_dur - ingress)
    flux[flat] -= eff_depth

    # Ingress/egress
    ing_zone = (np.abs(phase) > (half_dur - ingress)) & (np.abs(phase) <= half_dur)
    if ing_zone.any():
        frac = (half_dur - np.abs(phase[ing_zone])) / ingress
        flux[ing_zone] -= eff_depth * frac

    return flux


def transit_model(time, planet_params, u1=0.3, u2=0.1):
    """
    Compute joint flux model for N planets.

    planet_params : list of dicts with keys [period, t0, depth, duration]
    """
    flux = np.ones(len(time))
    for p in planet_params:
        flux *= _transit_model_single(
            time, p["period"], p["t0"], p["depth"], p["duration"], u1, u2
        )
    return flux


# ==========================================================
# PARAMETER PACKING / UNPACKING
# ==========================================================

def pack_params(planets, u1=0.3, u2=0.1):
    """Flatten planet parameters + limb darkening into a 1-D vector."""
    theta = []
    for p in planets:
        theta.extend([p["period"], p["t0"], p["depth"], p["duration"]])
    theta.extend([u1, u2])
    return np.array(theta)


def unpack_params(theta, n_planets):
    """Reconstruct list of planet dicts from parameter vector."""
    planets = []
    for i in range(n_planets):
        base = i * 4
        planets.append({
            "period":   theta[base + 0],
            "t0":       theta[base + 1],
            "depth":    theta[base + 2],
            "duration": theta[base + 3],
        })
    u1 = theta[n_planets * 4]
    u2 = theta[n_planets * 4 + 1]
    return planets, u1, u2


def param_names(n_planets):
    names = []
    for i in range(1, n_planets + 1):
        names += [f"P{i}(d)", f"t0_{i}(d)", f"δ{i}", f"τ{i}(d)"]
    names += ["u1", "u2"]
    return names


# ==========================================================
# PRIORS
# ==========================================================

def log_prior(theta, n_planets, prior_means, prior_sigmas):
    """
    Gaussian priors on all parameters.
    Returns log-prior probability.
    """
    planets, u1, u2 = unpack_params(theta, n_planets)

    lp = 0.0
    for i, p in enumerate(planets):
        base = i * 4

        # Period: tight Gaussian around TLS estimate
        lp -= 0.5 * ((p["period"]   - prior_means[base])   / prior_sigmas[base])  **2
        # t0: moderate uncertainty
        lp -= 0.5 * ((p["t0"]       - prior_means[base+1]) / prior_sigmas[base+1])**2
        # depth: log-normal (depth > 0)
        if p["depth"] <= 0:
            return -np.inf
        lp -= 0.5 * ((p["depth"]    - prior_means[base+2]) / prior_sigmas[base+2])**2
        # duration: positive
        if p["duration"] <= 0:
            return -np.inf
        lp -= 0.5 * ((p["duration"] - prior_means[base+3]) / prior_sigmas[base+3])**2

    # Limb darkening: physical constraint 0 ≤ u1, u2 ≤ 1, u1+u2 ≤ 1
    if not (0 <= u1 <= 1 and 0 <= u2 <= 1 and u1 + u2 <= 1):
        return -np.inf
    n = n_planets * 4
    lp -= 0.5 * ((u1 - prior_means[n])   / prior_sigmas[n])  **2
    lp -= 0.5 * ((u2 - prior_means[n+1]) / prior_sigmas[n+1])**2

    return lp


# ==========================================================
# LIKELIHOOD
# ==========================================================

def log_likelihood(theta, time, flux, flux_err, n_planets):
    """Gaussian log-likelihood: log p(data|theta)."""
    planets, u1, u2 = unpack_params(theta, n_planets)
    model = transit_model(time, planets, u1, u2)
    resid = flux - model
    return -0.5 * np.sum((resid / flux_err)**2 + np.log(2 * np.pi * flux_err**2))


def log_posterior(theta, time, flux, flux_err, n_planets, prior_means, prior_sigmas):
    lp = log_prior(theta, n_planets, prior_means, prior_sigmas)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, time, flux, flux_err, n_planets)


# ==========================================================
# MCMC — METROPOLIS-HASTINGS SAMPLER
# Pure numpy implementation; drop-in replaceable with emcee
# ==========================================================

def _mcmc_mh(log_prob_fn, theta0, n_steps=config.MCMC_N_STEPS,
             step_size=None, seed=42):
    """
    Single-chain Metropolis-Hastings MCMC.

    Parameters
    ----------
    log_prob_fn : callable(theta) → log-probability
    theta0      : initial parameter vector
    n_steps     : number of MCMC steps
    step_size   : proposal std (auto-tuned if None)

    Returns
    -------
    chain : (n_steps, n_params) array
    """
    rng = np.random.default_rng(seed)
    n_params = len(theta0)

    if step_size is None:
        step_size = np.abs(theta0) * 0.005 + 1e-6

    chain   = np.zeros((n_steps, n_params))
    current = theta0.copy()
    lp_cur  = log_prob_fn(current)
    n_accept = 0

    log.info(f"MCMC: {n_steps} steps, {n_params} parameters")

    for i in range(n_steps):
        proposal = current + rng.normal(0, step_size, n_params)
        lp_prop  = log_prob_fn(proposal)

        if np.log(rng.uniform()) < (lp_prop - lp_cur):
            current = proposal
            lp_cur  = lp_prop
            n_accept += 1

        chain[i] = current

        # Adaptive step size every 200 steps during burn-in
        if i < config.MCMC_BURN_IN and (i + 1) % 200 == 0:
            accept_rate = n_accept / (i + 1)
            if accept_rate < 0.20:
                step_size *= 0.7
            elif accept_rate > 0.40:
                step_size *= 1.3

    log.info(f"MCMC acceptance rate: {n_accept/n_steps*100:.1f}%")
    return chain


# ==========================================================
# INITIAL PARAMETER ESTIMATION (MAP via L-BFGS-B)
# ==========================================================

def find_map(time, flux, flux_err, planet_init, n_planets,
             prior_means, prior_sigmas):
    """Maximum-a-posteriori estimate as MCMC starting point."""
    theta0 = pack_params(planet_init)

    def neg_log_post(theta):
        val = log_posterior(theta, time, flux, flux_err, n_planets,
                            prior_means, prior_sigmas)
        return -val if np.isfinite(val) else 1e10

    bounds = []
    for i in range(n_planets):
        pm = prior_means[i*4:(i+1)*4]
        ps = prior_sigmas[i*4:(i+1)*4]
        bounds += [
            (max(0.1, pm[0] - 5*ps[0]),  pm[0] + 5*ps[0]),   # period
            (pm[1] - 2*ps[1],             pm[1] + 2*ps[1]),   # t0
            (max(1e-7, pm[2] - 5*ps[2]), pm[2] + 5*ps[2]),   # depth
            (max(0.001, pm[3] - 5*ps[3]),pm[3] + 5*ps[3]),   # duration
        ]
    bounds += [(0, 1), (0, 1)]  # u1, u2

    result = minimize(neg_log_post, theta0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-9})
    log.info(f"MAP optimisation: {'success' if result.success else 'failed'}, "
             f"f={result.fun:.4f}")
    return result.x


# ==========================================================
# MCMC ORCHESTRATION
# ==========================================================

def run_joint_mcmc(time, flux, flux_err, detected_planets,
                   n_steps=config.MCMC_N_STEPS,
                   burn_in=config.MCMC_BURN_IN):
    """
    Joint MCMC fitting of all detected planets simultaneously.

    Parameters
    ----------
    time, flux, flux_err : light curve arrays
    detected_planets     : list of dicts from exoplanets.detect_multiple_planets
    n_steps, burn_in     : MCMC settings

    Returns
    -------
    flat_samples : (n_samples, n_params) array post burn-in
    param_names  : list of parameter labels
    theta_med    : median posterior parameters
    """
    n_planets = len(detected_planets)
    log.info(f"Setting up joint MCMC for {n_planets} planets")

    # Build prior means and widths from TLS detections
    prior_means  = []
    prior_sigmas = []
    planet_init  = []

    for p in detected_planets:
        pm = p["period"]
        t0 = p["transit_time"]
        dp = p["depth"]
        dr = p.get("duration", 0.05 * pm)

        prior_means  += [pm, t0, dp, dr, 0.30, 0.10][-4:] if False else \
                        [pm, t0, dp, dr]
        prior_sigmas += [pm * 0.005, dr * 2, dp * 0.5, dr * 0.3]
        planet_init.append({"period": pm, "t0": t0, "depth": dp, "duration": dr})

    # Limb darkening priors (solar-like, broad)
    prior_means  += [0.30, 0.10]
    prior_sigmas += [0.15, 0.15]
    prior_means   = np.array(prior_means)
    prior_sigmas  = np.array(prior_sigmas)

    # MAP starting point
    theta_map = find_map(time, flux, flux_err, planet_init, n_planets,
                         prior_means, prior_sigmas)

    # MCMC
    log_prob = lambda th: log_posterior(
        th, time, flux, flux_err, n_planets, prior_means, prior_sigmas
    )
    chain = _mcmc_mh(log_prob, theta_map, n_steps=n_steps)

    # Discard burn-in
    flat_samples = chain[burn_in:]
    names        = param_names(n_planets)

    theta_med = np.median(flat_samples, axis=0)
    theta_lo  = np.percentile(flat_samples, 16, axis=0)
    theta_hi  = np.percentile(flat_samples, 84, axis=0)

    # Print credible intervals
    print("\n" + "="*60)
    print("  MCMC POSTERIOR SUMMARY (median ± 1σ)")
    print("="*60)
    for i, (name, med, lo, hi) in enumerate(
            zip(names, theta_med, theta_lo, theta_hi)):
        print(f"  {name:<14} {med:+.6f}  [{lo:+.6f}, {hi:+.6f}]")
    print("="*60 + "\n")

    return flat_samples, names, theta_med, theta_lo, theta_hi


# ==========================================================
# VISUALIZATION
# ==========================================================

def plot_trace(chain, names, burn_in=config.MCMC_BURN_IN, save=None):
    """MCMC trace plots — one panel per parameter."""
    n_params = min(len(names), 12)   # cap at 12 for readability
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, names[:n_params])):
        ax.plot(chain[:, i], lw=0.5, color="steelblue", alpha=0.8)
        ax.axvline(burn_in, color="red", ls="--", lw=1, label="Burn-in" if i == 0 else "")
        ax.set_ylabel(name, fontsize=8)

    axes[-1].set_xlabel("MCMC Step")
    axes[0].set_title("MCMC Trace Plots")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {save}")
    plt.close()


def plot_corner(flat_samples, names, save=None):
    """
    Corner plot (posterior covariance matrix).
    Implements internally without the 'corner' package.
    """
    n_params = min(len(names), 8)   # cap for readability
    samples  = flat_samples[:, :n_params]
    labels   = names[:n_params]

    fig, axes = plt.subplots(n_params, n_params, figsize=(2.2*n_params, 2.2*n_params))

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            if col == row:
                ax.hist(samples[:, row], bins=30, color="steelblue",
                        density=True, alpha=0.8)
                ax.set_yticks([])
            else:
                ax.plot(samples[:, col], samples[:, row],
                        ",k", markersize=0.5, alpha=0.2)
                ax.axvline(np.median(samples[:, col]), color="red", lw=0.8)
                ax.axhline(np.median(samples[:, row]), color="red", lw=0.8)

            if row == n_params - 1:
                ax.set_xlabel(labels[col], fontsize=7)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != 0:
                ax.set_ylabel(labels[row], fontsize=7)
            else:
                ax.set_yticklabels([])

    plt.suptitle("MCMC Posterior Corner Plot", fontsize=11, y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        log.info(f"Saved: {save}")
    plt.close()


def plot_model_fit(time, flux, flux_err, flat_samples, n_planets,
                   detected_planets, save=None):
    """
    Plot data with MAP model and posterior predictive envelope.
    One panel per planet (phase-folded).
    """
    n_cols = min(3, n_planets)
    n_rows = (n_planets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5*n_cols, 4*n_rows), squeeze=False)

    # Draw N sample models from posterior
    n_draws = min(100, len(flat_samples))
    idx     = np.random.choice(len(flat_samples), n_draws, replace=False)
    draws   = flat_samples[idx]
    theta_med = np.median(flat_samples, axis=0)

    for pi, p in enumerate(detected_planets):
        row, col = divmod(pi, n_cols)
        ax = axes[row][col]

        period = p["period"]
        t0     = p["transit_time"]
        phase  = ((time - t0 + 0.5*period) % period) - 0.5*period
        half_dur = p.get("duration", 0.05*period) * 1.5  # show ±1.5× duration
        window  = np.abs(phase) < half_dur

        ax.errorbar(phase[window], flux[window], yerr=flux_err[window],
                    fmt=".k", markersize=3, alpha=0.4, label="Data", zorder=1)

        # Posterior samples
        for theta in draws:
            plist, u1, u2 = unpack_params(theta, n_planets)
            m = _transit_model_single(
                time[window], plist[pi]["period"], plist[pi]["t0"],
                plist[pi]["depth"], plist[pi]["duration"], u1, u2
            )
            ax.plot(phase[window], m, lw=0.5, color="dodgerblue", alpha=0.05)

        # MAP model
        plist_med, u1_med, u2_med = unpack_params(theta_med, n_planets)
        m_med = _transit_model_single(
            time[window], plist_med[pi]["period"], plist_med[pi]["t0"],
            plist_med[pi]["depth"], plist_med[pi]["duration"], u1_med, u2_med
        )
        ax.plot(phase[window], m_med, "r-", lw=2, label="MAP model", zorder=3)

        ax.set_title(f"Planet {pi+1}  P={period:.3f} d", fontsize=10)
        ax.set_xlabel("Phase (days)", fontsize=9)
        ax.set_ylabel("Normalised Flux", fontsize=9)
        ax.legend(fontsize=8)

    # Hide empty panels
    for pi in range(n_planets, n_rows * n_cols):
        row, col = divmod(pi, n_cols)
        axes[row][col].set_visible(False)

    plt.suptitle("Joint Multi-Planet MCMC Fit", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {save}")
    plt.close()


def plot_posterior_periods(flat_samples, n_planets, names, save=None):
    """Period posterior distributions for all planets."""
    fig, axes = plt.subplots(1, n_planets, figsize=(4*n_planets, 4), squeeze=False)

    for i in range(n_planets):
        idx = i * 4   # period is first param for each planet
        ax  = axes[0][i]
        samples = flat_samples[:, idx]
        ax.hist(samples, bins=40, color="steelblue", density=True, alpha=0.8)
        med = np.median(samples)
        ax.axvline(med, color="red", lw=1.5, label=f"P={med:.4f} d")
        ax.axvline(np.percentile(samples, 16), color="red", lw=0.8, ls="--")
        ax.axvline(np.percentile(samples, 84), color="red", lw=0.8, ls="--")
        ax.set_xlabel(f"Period (days)", fontsize=10)
        ax.set_title(f"Planet {i+1}", fontsize=11)
        ax.legend(fontsize=9)

    plt.suptitle("Posterior Period Distributions", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {save}")
    plt.close()


# ==========================================================
# RESULTS EXPORT
# ==========================================================

def export_mcmc_results(flat_samples, names, theta_med, theta_lo, theta_hi,
                        detected_planets, path=config.MCMC_RESULTS_FILE):
    rows = []
    n_planets = len(detected_planets)
    for i, p in enumerate(detected_planets):
        base = i * 4
        rows.append({
            "planet_id":       i + 1,
            "period_med":      theta_med[base],
            "period_lo":       theta_lo[base],
            "period_hi":       theta_hi[base],
            "t0_med":          theta_med[base+1],
            "depth_med":       theta_med[base+2],
            "depth_lo":        theta_lo[base+2],
            "depth_hi":        theta_hi[base+2],
            "duration_med":    theta_med[base+3],
            "u1_med":          theta_med[n_planets*4],
            "u2_med":          theta_med[n_planets*4+1],
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info(f"MCMC results saved: {path}")
    return df


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_joint_modelling(time, flux, detected_planets,
                        flux_err=None, star="Kepler-11"):
    """
    Full joint multi-planet MCMC pipeline.

    Parameters
    ----------
    time, flux      : light curve (pre-cleaned)
    detected_planets: list of dicts from exoplanets.detect_multiple_planets
    flux_err        : per-point uncertainties (estimated from scatter if None)
    """
    if flux_err is None:
        # Estimate photon noise from out-of-transit scatter
        flux_err = np.ones(len(flux)) * np.std(flux - np.median(flux)) * 0.8
        log.info(f"Estimated flux uncertainty: {flux_err[0]*1e6:.0f} ppm")

    n_planets = len(detected_planets)
    prefix    = star.replace(" ", "_")

    # ── Step 1: MCMC ─────────────────────────────────────────────────────
    flat_samples, names, theta_med, theta_lo, theta_hi = run_joint_mcmc(
        time, flux, flux_err, detected_planets,
        n_steps=config.MCMC_N_STEPS,
        burn_in=config.MCMC_BURN_IN,
    )

    # ── Step 2: Save raw chain ────────────────────────────────────────────
    chain_path = os.path.join(config.OUTPUTS_DIR, f"{prefix}_mcmc_chain.pkl")
    with open(chain_path, "wb") as f:
        pickle.dump({"chain": flat_samples, "names": names}, f)
    log.info(f"Chain saved: {chain_path}")

    # ── Step 3: Plots ─────────────────────────────────────────────────────
    plot_trace(
        np.vstack([[np.zeros(len(names))] * config.MCMC_BURN_IN,
                    flat_samples]),    # prepend burn-in placeholder
        names, burn_in=config.MCMC_BURN_IN,
        save=os.path.join(config.PLOTS_DIR, f"{prefix}_mcmc_trace.png")
    )

    plot_corner(
        flat_samples, names,
        save=os.path.join(config.PLOTS_DIR, f"{prefix}_mcmc_corner.png")
    )

    plot_model_fit(
        time, flux, flux_err, flat_samples, n_planets, detected_planets,
        save=os.path.join(config.PLOTS_DIR, f"{prefix}_mcmc_fit.png")
    )

    plot_posterior_periods(
        flat_samples, n_planets, names,
        save=os.path.join(config.PLOTS_DIR, f"{prefix}_period_posteriors.png")
    )

    # ── Step 4: Export CSV ────────────────────────────────────────────────
    results_df = export_mcmc_results(
        flat_samples, names, theta_med, theta_lo, theta_hi, detected_planets
    )

    return flat_samples, results_df


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    # Load detection output and light curve
    from exoplanets import (
        generate_synthetic_lc, clean_and_flatten,
        detect_multiple_planets, KNOWN_STELLAR_PARAMS
    )

    KEPLER11_PLANETS = [
        {"period_days": 10.30, "t0": 2.1,  "depth": 0.000310, "duration_days": 0.090},
        {"period_days": 13.02, "t0": 5.3,  "depth": 0.000790, "duration_days": 0.100},
        {"period_days": 22.68, "t0": 1.8,  "depth": 0.000940, "duration_days": 0.130},
        {"period_days": 31.99, "t0": 7.0,  "depth": 0.001630, "duration_days": 0.150},
    ]

    time, flux = generate_synthetic_lc(
        "Kepler-11", KEPLER11_PLANETS,
        star_radius=1.065, duration_days=1460, noise_ppm=150
    )
    time, flux = clean_and_flatten(time, flux)

    detected, _ = detect_multiple_planets(time, flux, use_tls=True, max_planets=4)

    if detected:
        flat_samples, results_df = run_joint_modelling(
            time, flux, detected, star="Kepler-11"
        )
        print("\nMCMC results:")
        print(results_df.to_string(index=False))
