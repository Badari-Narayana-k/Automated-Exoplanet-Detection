"""
run_real_mcmc.py — joint MCMC fit on real data, with a fixed proposal
step-size scheme.

BUG FIX (see docs/BUGS_AND_FINDINGS.md #2): the original sampler in
jointMultiplanet.py initialises its Metropolis-Hastings proposal step size
as |theta_0| * 0.005. On the real, full 58,785-point, four-planet dataset
(log-posterior magnitude ~4x10^5, reflecting up to 142 observed transits
per planet), this scale is far larger than the sampler can accept: the
first real-data run returned a 0% acceptance rate across all 2000 steps --
the chain never moved from its starting point.

FIX: a two-stage adaptive scheme -- a tuning phase that adjusts each
parameter's step size individually toward a 25-50% target acceptance band,
followed by production sampling at the tuned scale. This is the standard
fix for a poorly-scaled naive step-size heuristic, not a novel algorithm.
"""
import sys, time as timer, json, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import minimize
from load_real_data import load_kepler11_real
import exoplanets as ex
import jointMultiplanet as jm


def main():
    tls_results = {d["name"]: d for d in json.load(open("results/real_tls_targeted_results.json"))}

    t, f, report = load_kepler11_real()
    t, f = ex.clean_and_flatten(t, f)
    flux_err = np.full(len(f), np.std(f))

    detected_planets = []
    for name in ["Kepler-11b", "Kepler-11c", "Kepler-11d", "Kepler-11e"]:
        r = tls_results[name]
        detected_planets.append({
            "period": r["period_found"], "t0": r.get("T0_found", 0.0),
            "depth": r["depth_found_ppm"] * 1e-6,
            "duration": (r.get("duration_found_hr") or 4.0) / 24.0,
        })

    n_planets = len(detected_planets)
    prior_means, prior_sigmas = [], []
    for p in detected_planets:
        pm, t0v, dp, dr = p["period"], p["t0"], p["depth"], p["duration"]
        prior_means  += [pm, t0v, dp, dr]
        prior_sigmas += [pm * 0.005, dr * 2, dp * 0.5, dr * 0.3]
    prior_means  += [0.30, 0.10]
    prior_sigmas += [0.15, 0.15]
    prior_means, prior_sigmas = np.array(prior_means), np.array(prior_sigmas)

    theta0 = jm.pack_params(detected_planets)

    def neg_log_post(theta):
        val = jm.log_posterior(theta, t, f, flux_err, n_planets, prior_means, prior_sigmas)
        return -val if np.isfinite(val) else 1e10

    bounds = []
    for i in range(n_planets):
        pmi, psi = prior_means[i*4:(i+1)*4], prior_sigmas[i*4:(i+1)*4]
        bounds += [(max(0.1, pmi[0]-5*psi[0]), pmi[0]+5*psi[0]),
                   (pmi[1]-2*psi[1], pmi[1]+2*psi[1]),
                   (max(1e-7, pmi[2]-5*psi[2]), pmi[2]+5*psi[2]),
                   (max(0.001, pmi[3]-5*psi[3]), pmi[3]+5*psi[3])]
    bounds += [(0, 1), (0, 1)]

    result = minimize(neg_log_post, theta0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-9})
    theta_map = result.x
    print(f"MAP: success={result.success}, f={result.fun:.2f}")

    log_prob = lambda th: jm.log_posterior(th, t, f, flux_err, n_planets, prior_means, prior_sigmas)

    # --- Adaptive per-parameter tuning (replaces the broken |theta0|*0.005 heuristic) ---
    rng = np.random.default_rng(7)
    n_params = len(theta_map)
    step = prior_sigmas * 0.25
    current = theta_map.copy()
    lp_cur = log_prob(current)

    tune_steps, check_every = 2500, 40
    n_accept_window = np.zeros(n_params)

    for i in range(tune_steps):
        for j in range(n_params):
            proposal = current.copy()
            proposal[j] += rng.normal(0, step[j])
            lp_prop = log_prob(proposal)
            if np.log(rng.uniform()) < (lp_prop - lp_cur):
                current, lp_cur = proposal, lp_prop
                n_accept_window[j] += 1
        if (i + 1) % check_every == 0:
            rate = n_accept_window / check_every
            step *= np.where(rate < 0.25, 0.8, np.where(rate > 0.5, 1.25, 1.0))
            n_accept_window[:] = 0

    print("Tuning done. Final step sizes:", step)

    # --- Production sampling ---
    n_steps = 1500
    chain = np.zeros((n_steps, n_params))
    n_accept = np.zeros(n_params)
    for i in range(n_steps):
        for j in range(n_params):
            proposal = current.copy()
            proposal[j] += rng.normal(0, step[j])
            lp_prop = log_prob(proposal)
            if np.log(rng.uniform()) < (lp_prop - lp_cur):
                current, lp_cur = proposal, lp_prop
                n_accept[j] += 1
        chain[i] = current

    print("Per-parameter acceptance rate:", n_accept / n_steps)

    burn_in = 500
    flat_samples = chain[burn_in:]
    names = jm.param_names(n_planets)
    theta_med = np.median(flat_samples, axis=0)
    theta_lo = np.percentile(flat_samples, 16, axis=0)
    theta_hi = np.percentile(flat_samples, 84, axis=0)

    os.makedirs("results", exist_ok=True)
    np.save("results/real_mcmc_chain.npy", flat_samples)
    results_df = jm.export_mcmc_results(
        flat_samples, names, theta_med, theta_lo, theta_hi, detected_planets,
        path="results/mcmcResults_real.csv"
    )
    return results_df


if __name__ == "__main__":
    main()
