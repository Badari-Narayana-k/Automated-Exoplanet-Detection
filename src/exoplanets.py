# ==========================================================
# exoplanets.py — EXOPLANET DETECTION PIPELINE (v2)
# BLS + TLS Multi-Planet Transit Search
#
# BUGS FIXED vs v1:
#   [CRITICAL] result.save() → pickle.dump() (TLS has no .save())
#   [CRITICAL] Duplicate-check fired AFTER residual injection
#   [HIGH]     Transit mask padding factor 0.6 inverted → 1+padding
#   [HIGH]     detect_strongest_planet_tls returned inconsistent tuple
#   [MEDIUM]   Bare except: → FileNotFoundError only
#   [MEDIUM]   BLS defined but never wired into pipeline
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
from scipy.signal import savgol_filter

import config

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ==========================================================
# LIGHT CURVE SIMULATION (replaces lightkurve for offline use)
# In production: swap generate_synthetic_lc() with download_target_data()
# ==========================================================

def generate_synthetic_lc(target, planets_params, star_radius=1.0,
                           duration_days=1460, cadence_minutes=30, noise_ppm=150):
    """
    Generate a synthetic Kepler-style multi-planet light curve.

    Parameters
    ----------
    planets_params : list of dict with keys:
        period_days, t0, depth (Rp/R*)^2, duration_days
    noise_ppm : instrumental + stellar noise in parts per million
    """
    log.info(f"Generating synthetic light curve for {target}")
    rng = np.random.default_rng(42)

    cadence = cadence_minutes / (60 * 24)   # days
    time = np.arange(0, duration_days, cadence)
    flux = np.ones(len(time))

    for p in planets_params:
        phase = ((time - p["t0"]) % p["period_days"]) / p["period_days"]
        # Map to [-0.5, 0.5]
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        half_dur_frac = (p["duration_days"] / p["period_days"]) / 2.0
        in_transit = np.abs(phase) < half_dur_frac

        # Trapezoidal ingress/egress (15% of duration)
        ingress_frac = 0.15 * half_dur_frac
        t_dist       = np.abs(phase)
        flat_bottom  = t_dist < (half_dur_frac - ingress_frac)
        ingress_zone = (~flat_bottom) & in_transit

        trap = np.zeros(len(time))
        trap[flat_bottom] = p["depth"]
        if ingress_zone.any():
            frac = (half_dur_frac - t_dist[ingress_zone]) / (ingress_frac + 1e-12)
            trap[ingress_zone] = p["depth"] * np.clip(frac, 0, 1)
        flux -= trap

    # Gaussian noise
    sigma = noise_ppm * 1e-6
    flux += rng.normal(0, sigma, len(time))

    # Mild sinusoidal stellar variability
    flux += 200e-6 * np.sin(2 * np.pi * time / 25.0)   # 25-day rotation period

    return time, flux


def download_target_data(target, mission="kepler"):
    """
    Download light curve from local cache or Lightkurve.
    Requires lightkurve package in production environment.
    """
    try:
        import lightkurve as lk
        import glob

        data_path = os.path.join(config.CACHE_DIR, "tpf")
        fits_files = glob.glob(os.path.join(data_path, "*.fits"))

        if fits_files:
            log.info(f"Loading {len(fits_files)} cached FITS files for {target}")
            lc_list = []
            for f in fits_files:
                try:
                    lc_list.append(lk.read(f))
                except Exception as e:
                    log.warning(f"Skipping corrupt file {f}: {e}")
            if not lc_list:
                raise RuntimeError("No valid light curves loaded from cache")
            lc = lk.LightCurveCollection(lc_list).stitch().remove_nans().normalize()
            lc = lc.bin(time_bin_size=config.BIN_SIZE_DAYS)
            return lc.time.value, lc.flux.value
        else:
            log.info(f"Downloading {target} from MAST")
            search = lk.search_lightcurve(target, mission=mission, author="Kepler")
            if len(search) == 0:
                raise RuntimeError("No data found on MAST")
            lc = search.download_all().stitch().remove_nans().normalize()
            lc = lc.bin(time_bin_size=config.BIN_SIZE_DAYS)
            return lc.time.value, lc.flux.value

    except ImportError:
        raise ImportError(
            "lightkurve not installed. Use generate_synthetic_lc() for testing, "
            "or install lightkurve: pip install lightkurve"
        )


# ==========================================================
# PREPROCESSING
# ==========================================================

def clean_and_flatten(time, flux, window_length=config.WINDOW_LENGTH,
                      sigma=config.OUTLIER_SIGMA):
    """Remove outliers and flatten long-term trends."""
    # Sigma-clip
    med = np.median(flux)
    std = np.std(flux)
    mask = np.abs(flux - med) < sigma * std
    time, flux = time[mask], flux[mask]

    # Savitzky-Golay detrending
    wl = min(window_length, len(flux) - 1)
    if wl % 2 == 0:
        wl -= 1
    trend = savgol_filter(flux, wl, 3)
    flat_flux = flux / trend

    log.info(f"Preprocessing: {mask.sum()} / {len(mask)} points retained")
    return time, flat_flux


def smooth_flux(flux, window=51, poly=3):
    return savgol_filter(flux, min(window, len(flux) - 1), poly)


# ==========================================================
# BLS PERIOD SEARCH
# ==========================================================

def run_bls(time, flux, min_period=config.MIN_PERIOD,
            max_period=config.MAX_PERIOD, n_periods=5000):
    """
    Box Least Squares period search using pure numpy.
    Returns dict: {period, t0, depth, duration, snr, power_array, periods}
    """
    log.info("Running BLS period search...")

    periods = np.linspace(min_period, min(max_period, (time[-1] - time[0]) / 2), n_periods)
    power = np.zeros(n_periods)

    # BLS core: fold and compute in-transit vs out-of-transit flux
    duration_fracs = [0.02, 0.05, 0.10, 0.15]   # try multiple durations

    best = {"power": -np.inf}

    for pi, period in enumerate(periods):
        phase = ((time - time[0]) % period) / period   # [0, 1)
        for dfrac in duration_fracs:
            in_t = phase < dfrac
            out_t = ~in_t
            if in_t.sum() < 3 or out_t.sum() < 3:
                continue
            s = np.mean(flux[in_t]) - np.mean(flux[out_t])   # depth signal
            sigma_in = np.std(flux[in_t]) / np.sqrt(in_t.sum())
            sigma_out = np.std(flux[out_t]) / np.sqrt(out_t.sum())
            snr = abs(s) / (sigma_in + sigma_out + 1e-12)

            if snr > power[pi]:
                power[pi] = snr
            if snr > best.get("power", 0):
                t0 = time[0] + np.argmin(np.convolve(
                    flux, np.ones(max(1, int(dfrac * len(time) / (period / (time[1]-time[0]))))),
                    mode='same')) * (time[1] - time[0]) % period
                best = {
                    "power": snr, "period": period, "t0": t0,
                    "depth": abs(s), "duration_frac": dfrac, "snr": snr
                }

    best["duration_days"] = best["period"] * best.get("duration_frac", 0.05)
    best["power_array"]   = power
    best["periods"]       = periods

    log.info(f"BLS best period: {best['period']:.4f} days  SNR: {best['snr']:.2f}")
    return best


# ==========================================================
# TLS PERIOD SEARCH (pure-numpy reimplementation for offline use)
# In production: use transitleastsquares package directly
# ==========================================================

def run_tls_native(time, flux, min_period=config.MIN_PERIOD,
                   max_period=config.MAX_PERIOD, n_periods=3000,
                   oversampling=config.TLS_OVERSAMPLING):
    """
    Transit Least Squares using the matched-filter principle.
    Unlike BLS (box), TLS uses a physically motivated transit shape.

    Key difference from BLS:
    - BLS fits a flat-bottomed box → overestimates depth for grazing transits
    - TLS convolves with a realistic trapezoidal template including ingress/egress
    - TLS SDE is more statistically meaningful than BLS SNR

    Returns dict with same interface as the 'transitleastsquares' package.
    """
    log.info("Running TLS period search (native implementation)...")

    dt = np.median(np.diff(time))
    periods = np.exp(np.linspace(
        np.log(min_period),
        np.log(min(max_period, (time[-1] - time[0]) / 2)),
        n_periods
    ))   # Log-uniform grid (finer at short periods)

    sde_array = np.zeros(n_periods)
    best = {"sde": -np.inf}

    for pi, period in enumerate(periods):
        phase = ((time - time[0]) % period)
        phase_norm = phase / period   # [0,1)

        # Try several duration fractions
        trial_depths = []
        for dfrac in [0.02, 0.04, 0.07, 0.12]:
            ingress = 0.15 * dfrac
            # Build transit template
            template = np.zeros(len(time))
            in_flat   = phase_norm < (dfrac - ingress)
            in_ingress = (phase_norm >= (dfrac - ingress)) & (phase_norm < dfrac)
            template[in_flat]    = 1.0
            if in_ingress.any():
                template[in_ingress] = (dfrac - phase_norm[in_ingress]) / ingress

            r = np.dot(template, flux - 1.0)    # matched filter
            n_pts = template.sum()
            if n_pts < 2:
                continue
            depth_est = -r / n_pts
            trial_depths.append(depth_est)

        if not trial_depths:
            continue

        # SDE proxy: signal / local noise
        depth_signal = max(trial_depths)
        local_std    = np.std(flux[phase_norm < 0.2]) if (phase_norm < 0.2).sum() > 5 else np.std(flux)
        sde_proxy    = depth_signal / (local_std / np.sqrt(max(1, int(0.05 * len(time) * period / (time[-1]-time[0]) ))))

        sde_array[pi] = sde_proxy

        if sde_proxy > best["sde"]:
            # Find best t0 for this period
            t0_candidates = np.arange(time[0], time[0] + period, dt * 5)
            best_depth, best_t0 = 0, time[0]
            for t0c in t0_candidates:
                ph = ((time - t0c) % period) / period
                ph = np.where(ph > 0.5, ph - 1.0, ph)
                in_t = np.abs(ph) < 0.05
                if in_t.sum() < 2:
                    continue
                d = 1.0 - np.mean(flux[in_t])
                if d > best_depth:
                    best_depth = d
                    best_t0 = t0c

            dur_est = 0.05 * period  # rough estimate
            best = {
                "sde": sde_proxy,
                "period": period,
                "T0": best_t0,
                "depth": 1.0 - best_depth,   # depth convention: 1-min_flux
                "duration": dur_est,
                "power": sde_array,
                "periods": periods,
            }

    # Normalise SDE to zero mean, unit std (matches TLS package convention)
    mu, sig = np.mean(sde_array), np.std(sde_array)
    sde_norm = (sde_array - mu) / (sig + 1e-12)
    best["sde"] = (best["sde"] - mu) / (sig + 1e-12)
    best["power"] = sde_norm
    best["depth"] = max(0.0, min(1.0, 1.0 - best["depth"]))  # TLS convention

    log.info(f"TLS best period: {best['period']:.4f} days  SDE: {best['sde']:.2f}")
    return best


# ==========================================================
# MULTI-PLANET ITERATIVE DETECTION
# ==========================================================

def detect_multiple_planets(time, flux, use_tls=True,
                            max_planets=config.MAX_PLANETS,
                            sde_threshold=config.TLS_SDE_THRESHOLD):
    """
    Iterative 'peel-off' multi-planet detection.

    Algorithm:
      1. Search for strongest periodic signal (BLS or TLS)
      2. If significant, record planet parameters
      3. Mask in-transit points (replace with 1.0) and repeat

    BUG FIX: duplicate check now runs BEFORE residual injection
    BUG FIX: transit mask uses 1+padding factor (was wrongly 0.6)
    """
    planets = []
    residual = flux.copy()

    for i in range(max_planets):
        log.info(f"--- Searching for planet {i+1} ---")

        if use_tls:
            result = run_tls_native(time, residual)
            period = result["period"]
            t0     = result["T0"]
            depth  = 1.0 - result["depth"]    # fractional depth
            dur    = result["duration"]
            sde    = result["sde"]
        else:
            result = run_bls(time, residual)
            period = result["period"]
            t0     = result["t0"]
            depth  = result["depth"]
            dur    = result["duration_days"]
            sde    = result["snr"]

        if sde < sde_threshold:
            log.info(f"SDE {sde:.2f} below threshold {sde_threshold} — stopping")
            break

        # ── DUPLICATE CHECK (must happen BEFORE residual injection) ──────
        is_dup = any(
            abs(period - p["period"]) / p["period"] < config.PERIOD_TOLERANCE
            for p in planets
        )
        if is_dup:
            log.warning(f"Period {period:.4f} d is a duplicate — skipping")
            # Mask anyway to prevent infinite loop
            _inject_ones(time, residual, t0, period, dur)
            continue

        planet = {
            "planet_id":  i + 1,
            "period":     period,
            "transit_time": t0,
            "depth":      depth,
            "duration":   dur,
            "SDE":        sde,
        }
        planets.append(planet)
        log.info(f"Planet {i+1}: P={period:.4f}d  depth={depth:.5f}  SDE={sde:.2f}")

        # ── RESIDUAL INJECTION (after duplicate check) ───────────────────
        _inject_ones(time, residual, t0, period, dur)

    log.info(f"Detection complete: {len(planets)} planets found")
    return planets, result   # return last TLS/BLS result for plotting


def _inject_ones(time, flux, t0, period, duration):
    """Set in-transit flux to 1.0. BUG FIX: padding = 1+factor, not 0.6."""
    phase = ((time - t0 + 0.5*period) % period) - 0.5*period
    mask  = np.abs(phase) < (duration / 2 * (1 + config.TRANSIT_PADDING))
    flux[mask] = 1.0


# ==========================================================
# STELLAR PARAMETERS & PLANET PHYSICS
# ==========================================================

def query_star_data_archive(star):
    """Query NASA Exoplanet Archive via astroquery (requires network)."""
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        table = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="hostname,st_mass,st_rad,st_lum",
            where=f"hostname like '{star}%'"
        )
        return {
            "mass":       float(np.asarray(table["st_mass"][0]).item()),
            "radius":     float(np.asarray(table["st_rad"][0]).item()),
            "luminosity": float(10 ** np.asarray(table["st_lum"][0]).item()),
        }
    except Exception as e:
        log.warning(f"Archive query failed ({e}), using solar defaults")
        return {"mass": 1.0, "radius": 1.0, "luminosity": 1.0}


# Known stellar params for common targets (fallback without network)
KNOWN_STELLAR_PARAMS = {
    "Kepler-11": {"mass": 0.961, "radius": 1.065, "luminosity": 0.952, "teff": 5680},
}


def get_star_data(star):
    if star in KNOWN_STELLAR_PARAMS:
        log.info(f"Using catalogued stellar parameters for {star}")
        return KNOWN_STELLAR_PARAMS[star]
    return query_star_data_archive(star)


def compute_planet_radius_re(star_radius_rsun, depth):
    """Planet radius in Earth radii. depth = (Rp/R*)^2."""
    R_SUN_TO_EARTH = 109.076
    return star_radius_rsun * np.sqrt(depth) * R_SUN_TO_EARTH


def compute_orbital_distance(period_days, star_mass_msun):
    """Semi-major axis via Kepler's third law (AU)."""
    period_yr = period_days / 365.25
    return (period_yr ** (2/3)) * (star_mass_msun ** (1/3))


def compute_habitable_zone(luminosity):
    """Conservative HZ limits (Kopparapu+2013 approximation)."""
    inner = np.sqrt(luminosity / config.HZ_INNER_COEFF)
    outer = np.sqrt(luminosity / config.HZ_OUTER_COEFF)
    return inner, outer


def compute_planet_parameters(star, planets):
    sd = get_star_data(star)
    records = []
    for p in planets:
        orbit   = compute_orbital_distance(p["period"], sd["mass"])
        rp_re   = compute_planet_radius_re(sd["radius"], p["depth"])
        hz_in, hz_out = compute_habitable_zone(sd["luminosity"])
        records.append({
            "star":         star,
            "planet_id":    int(p["planet_id"]),
            "period_days":  float(p["period"]),
            "transit_time": float(p["transit_time"]),
            "transit_depth":float(p["depth"]),
            "duration_hrs": float(p["duration"] * 24),
            "SDE":          float(p["SDE"]),
            "radius_RE":    float(rp_re),
            "orbit_AU":     float(orbit),
            "hz_inner_AU":  float(hz_in),
            "hz_outer_AU":  float(hz_out),
            "in_HZ":        bool(hz_in <= orbit <= hz_out),
            "stellar_mass": float(sd["mass"]),
            "stellar_radius": float(sd["radius"]),
            "stellar_lum":  float(sd["luminosity"]),
        })
    return pd.DataFrame(records)


# ==========================================================
# VISUALIZATION
# ==========================================================

def plot_lightcurve(time, flux, title="Cleaned Light Curve", save=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, flux, ".k", markersize=1.2, alpha=0.6)
    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Normalised Flux", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {save}")
    plt.close()


def plot_periodogram(result, method="TLS", save=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result["periods"], result["power"], lw=0.8, color="steelblue")
    ax.axvline(result["period"], color="red", lw=1.5,
               label=f"Best P = {result['period']:.4f} d")
    ax.set_xlabel("Period (days)", fontsize=12)
    ax.set_ylabel("SDE" if method == "TLS" else "SNR", fontsize=12)
    ax.set_xscale("log")
    ax.set_title(f"{method} Periodogram", fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {save}")
    plt.close()


def plot_phase_folded(time, flux, period, t0, depth=None,
                      title="Phase-Folded Transit", save=None, bins=80):
    phase = ((time - t0 + 0.5*period) % period) - 0.5*period
    idx   = np.argsort(phase)
    ph_s, fl_s = phase[idx], flux[idx]

    # Binned curve
    bin_edges  = np.linspace(ph_s[0], ph_s[-1], bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_flux    = np.array([
        np.median(fl_s[(ph_s >= bin_edges[j]) & (ph_s < bin_edges[j+1])])
        if ((ph_s >= bin_edges[j]) & (ph_s < bin_edges[j+1])).any() else np.nan
        for j in range(bins)
    ])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ph_s, fl_s, ".k", markersize=1.5, alpha=0.3, label="Data")
    ax.plot(bin_centres, bin_flux, "r-", lw=2, label="Binned")
    if depth:
        ax.axhline(1 - depth, color="dodgerblue", lw=1.5, ls="--",
                   label=f"Depth {depth*1e6:.0f} ppm")
    ax.set_xlabel("Phase (days)", fontsize=12)
    ax.set_ylabel("Normalised Flux", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {save}")
    plt.close()


def plot_residual_diagnostics(time, flux, title="Residuals", save=None):
    residuals = flux - 1.0
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))

    axs[0, 0].plot(time, residuals, ".k", markersize=1.2, alpha=0.4)
    axs[0, 0].axhline(0, color="red", lw=0.8)
    axs[0, 0].set_title("Residual vs Time")

    axs[0, 1].hist(residuals, bins=60, density=True, color="steelblue", alpha=0.8)
    x = np.linspace(residuals.min(), residuals.max(), 200)
    from scipy.stats import norm
    axs[0, 1].plot(x, norm.pdf(x, residuals.mean(), residuals.std()), "r-", lw=1.5)
    axs[0, 1].set_title("Residual Distribution")

    acf = np.correlate(residuals - residuals.mean(), residuals - residuals.mean(), "full")
    acf = acf[len(acf)//2:] / acf[len(acf)//2]
    axs[1, 0].plot(acf[:200], color="steelblue")
    axs[1, 0].axhline(0, color="k", lw=0.8)
    axs[1, 0].set_title("Autocorrelation")

    chunk = max(10, len(residuals)//20)
    rms = [np.std(residuals[i:i+chunk]) for i in range(0, len(residuals)-chunk, chunk)]
    axs[1, 1].plot(rms, "o-", color="steelblue", markersize=4)
    axs[1, 1].set_title("RMS Stability")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {save}")
    plt.close()


def plot_bls_vs_tls(time, flux, bls_result, tls_result, save=None):
    """Side-by-side comparison showing WHY TLS outperforms BLS."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, result, method, color in zip(
        axes,
        [bls_result, tls_result],
        ["BLS (Box Least Squares)", "TLS (Transit Least Squares)"],
        ["darkorange", "steelblue"]
    ):
        key = "periods" if "periods" in result else "periods"
        ax.plot(result["periods"], result["power"], lw=0.8, color=color)
        ax.axvline(result["period"], color="red", lw=1.5, ls="--",
                   label=f"Best: {result['period']:.3f} d")
        ax.set_xscale("log")
        ax.set_xlabel("Period (days)", fontsize=11)
        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("SNR (signal/noise)")
    axes[1].set_ylabel("SDE (signal detection efficiency)")
    plt.suptitle("BLS vs TLS Periodogram Comparison", fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()


# ==========================================================
# CATALOG I/O
# ==========================================================

def save_catalog(df, path=config.CATALOG_FILE):
    try:
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass
    except Exception as e:
        log.error(f"Could not read existing catalog: {e}")

    df.to_csv(path, index=False)
    log.info(f"Catalog saved: {path}  ({len(df)} rows)")


def save_tls_result(result, path):
    """Safely pickle TLS/search result object."""
    with open(path, "wb") as f:
        pickle.dump(result, f)
    log.info(f"TLS result pickled: {path}")


# ==========================================================
# STAR ANALYSIS PIPELINE
# ==========================================================

def analyze_star(target, time=None, flux=None, use_tls=True):
    """
    Full analysis pipeline for one star.
    Pass (time, flux) directly for synthetic/pre-loaded data,
    or leave None to trigger download from cache/MAST.
    """
    if time is None or flux is None:
        time, flux = download_target_data(target)

    log.info(f"=== Analysing {target}: {len(time)} data points ===")
    time, flux = clean_and_flatten(time, flux)

    plot_lightcurve(
        time, flux,
        title=f"{target} — Cleaned Light Curve",
        save=os.path.join(config.PLOTS_DIR, f"{target.replace(' ','_')}_lightcurve.png")
    )

    # Run BLS first (fast), then TLS
    bls_result = run_bls(time, flux)
    tls_result = run_tls_native(time, flux)

    plot_bls_vs_tls(
        time, flux, bls_result, tls_result,
        save=os.path.join(config.PLOTS_DIR, f"{target.replace(' ','_')}_bls_vs_tls.png")
    )

    # Iterative multi-planet detection with TLS
    planets, last_result = detect_multiple_planets(time, flux, use_tls=use_tls)

    if not planets:
        log.warning("No planets detected")
        return None

    # Phase-fold each planet
    for p in planets:
        plot_phase_folded(
            time, flux, p["period"], p["transit_time"], depth=p["depth"],
            title=f"{target} — Planet {p['planet_id']} (P={p['period']:.3f} d)",
            save=os.path.join(config.PLOTS_DIR,
                f"{target.replace(' ','_')}_planet{p['planet_id']}_phase.png")
        )

    plot_residual_diagnostics(
        time, flux,
        title=f"{target} — Post-Detection Residuals",
        save=os.path.join(config.PLOTS_DIR, f"{target.replace(' ','_')}_residuals.png")
    )

    df = compute_planet_parameters(target, planets)
    log.info(f"\n{df[['planet_id','period_days','transit_depth','radius_RE','orbit_AU','in_HZ']].to_string()}")
    return df


def scan_star_list(star_list, synthetic_data=None):
    all_results = []
    for i, star in enumerate(star_list):
        try:
            time_i, flux_i = (synthetic_data[i] if synthetic_data else (None, None))
            df = analyze_star(star, time=time_i, flux=flux_i)
            if df is not None:
                all_results.append(df)
        except Exception as e:
            log.error(f"Error with {star}: {e}", exc_info=True)

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        save_catalog(final)
        return final
    return pd.DataFrame()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    # Kepler-11 known planets for synthetic data
    KEPLER11_PLANETS = [
        {"period_days": 10.30, "t0": 2.1,  "depth": 0.000310, "duration_days": 0.090},
        {"period_days": 13.02, "t0": 5.3,  "depth": 0.000790, "duration_days": 0.100},
        {"period_days": 22.68, "t0": 1.8,  "depth": 0.000940, "duration_days": 0.130},
        {"period_days": 31.99, "t0": 7.0,  "depth": 0.001630, "duration_days": 0.150},
        {"period_days": 46.69, "t0": 3.5,  "depth": 0.000540, "duration_days": 0.120},
        {"period_days":118.38, "t0": 12.0, "depth": 0.000890, "duration_days": 0.200},
    ]

    time, flux = generate_synthetic_lc(
        "Kepler-11", KEPLER11_PLANETS,
        star_radius=1.065, duration_days=1460, noise_ppm=150
    )

    df = analyze_star("Kepler-11", time=time, flux=flux)
    if df is not None:
        save_catalog(df)
        print("\n=== Planet Catalog ===")
        print(df.to_string(index=False))
