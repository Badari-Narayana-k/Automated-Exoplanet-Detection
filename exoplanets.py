# ==========================================================
# EXOPLANET DETECTION PIPELINE (UPDATED)
# Multi-Planet Transit Search + TLS + Joint Modeling Ready
# ==========================================================
import glob
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import lightkurve as lk
from transitleastsquares import transitleastsquares

from dash import Dash, dcc, html
import plotly.express as px

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

warnings.filterwarnings("ignore")

# ==========================================================
# CACHE AND CONFIGURATION
# ==========================================================
CACHE_DIR = "cache"
TPF_CACHE = os.path.join(CACHE_DIR, "tpf")
STAR_CACHE = os.path.join(CACHE_DIR, "stars")
os.makedirs(TPF_CACHE, exist_ok=True)
os.makedirs(STAR_CACHE, exist_ok=True)

MAX_PLANETS_PER_STAR = 6
MIN_PERIOD = 0.5  # days
MAX_PERIOD = 300  # days
WINDOW_LENGTH = 401
OUTLIER_SIGMA = 5
CATALOG_FILE = "/content/drive/MyDrive/Research_Project/outputs/detectedPlanets.csv"

# ==========================================================
# LIGHT CURVE DOWNLOAD AND CLEANING
# ==========================================================
def download_target_data(target, mission="kepler"):
    print(f"Loading LOCAL FITS data for {target}")
    data_path = "/content/drive/MyDrive/Research_project/datasets/cache/tpf/"
    print("Checking path:", data_path)
    print("Files inside:", os.listdir(data_path))

    # Get all FITS files
    fits_files = glob.glob(os.path.join(data_path, "*.fits"))

    if len(fits_files) == 0:
        raise Exception("No FITS files found in local dataset")

    lc_list = []

    for file in fits_files:
        try:
            lc = lk.read(file)
            lc_list.append(lc)
        except Exception as e:
            print(f"Skipping corrupt file: {file}")

    if len(lc_list) == 0:
        raise Exception("No valid light curves loaded")

    # Stitch all light curves
    lc_collection = lk.LightCurveCollection(lc_list)
    lc = lc_collection.stitch().remove_nans().normalize()
    # Downsample (binning)
    lc = lc.bin(time_bin_size=0.02)  # ~30 min bins

    return lc

def clean_lightcurve(lc):
    lc = lc.remove_nans().remove_outliers(sigma=OUTLIER_SIGMA).normalize()
    flat_lc = lc.flatten(window_length=WINDOW_LENGTH)
    return flat_lc


def smooth_lightcurve(lc):
    flux = lc.flux.value
    smoothed_flux = savgol_filter(flux, 51, 3)
    lc = lc.copy()
    lc.flux = smoothed_flux * lc.flux.unit
    return lc

# ==========================================================
# VISUALIZATION
# ==========================================================
def plot_residual_diagnostics(time, flux, model_flux=None, title="Residual Analysis"):
    residuals = flux if model_flux is None else flux - model_flux

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Residual vs Time
    axs[0, 0].plot(time, residuals, ".k", markersize=2)
    axs[0, 0].set_title("Residual vs Time")

    # Histogram
    axs[0, 1].hist(residuals, bins=50, density=True, alpha=0.7)
    axs[0, 1].set_title("Residual Distribution")

    # Autocorrelation
    autocorr = np.correlate(residuals - np.mean(residuals),
                            residuals - np.mean(residuals),
                            mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    axs[1, 0].plot(autocorr[:200])
    axs[1, 0].set_title("Autocorrelation")

    # RMS stability
    chunk_size = max(10, len(residuals)//20)
    rms_vals = [
        np.std(residuals[i:i+chunk_size])
        for i in range(0, len(residuals)-chunk_size, chunk_size)
    ]
    axs[1, 1].plot(rms_vals, marker='o')
    axs[1, 1].set_title("RMS Stability")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_phase_folded(time, flux, period, t0, title="Phase Folded"):
    phase = ((time - t0 + 0.5*period) % period) - 0.5*period

    plt.figure(figsize=(6,4))
    plt.plot(phase, flux, ".k", markersize=2)
    plt.xlabel("Phase")
    plt.ylabel("Flux")
    plt.title(title)
    plt.show()


def odd_even_transit_test(time, flux, period, t0):
    phase = ((time - t0) % period) / period

    odd = flux[(phase > 0.0) & (phase < 0.5)]
    even = flux[(phase > 0.5) & (phase < 1.0)]

    if len(odd) > 0 and len(even) > 0:
        diff = abs(np.mean(odd) - np.mean(even))
        print(f"Odd-Even Depth Difference: {diff:.6f}")

# ===========================================================
# BLS Period Search
# ===========================================================

from astropy.timeseries import BoxLeastSquares

def run_bls(time, flux):
    model = BoxLeastSquares(time, flux)
    periods = np.linspace(0.5, 200, 10000)
    results = model.power(periods, 0.1)
    best_period = periods[np.argmax(results.power)]
    print("BLS Best Period:", best_period)
    return best_period



# ==========================================================
# TLS PERIOD SEARCH Basic
# ==========================================================
def detect_strongest_planet_tls(lc):
    time = lc.time.value
    flux = lc.flux.value

    model = transitleastsquares(time, flux)
    results = model.power(minimum_period=MIN_PERIOD, maximum_period=MAX_PERIOD)

    if results.SDE < 8:  # TLS detection threshold, adjust if needed
        return None

    true_depth=1-results.depth
    planet = {
        "period": results.period,
        "transit_time": results.T0,
        "depth": true_depth,
        "duration": results.duration,
        "SDE": results.SDE
    }

    print(f"Detected planet: Period={planet['period']:.4f} days, SDE={planet['SDE']:.2f}")
    return planet, results

# ==========================================================
# MASK OR SUBTRACT TRANSITS
# ==========================================================
def mask_transit(lc, planet, padding=0.1):
    lc = lc.copy()
    period = planet["period"]
    t0 = planet["transit_time"]
    duration = planet["duration"]

    phase = ((lc.time.value - t0 + 0.5*period) % period) - 0.5*period
    transit_window = (duration / 2) * (1 + padding)
    mask = np.abs(phase) > transit_window

    return lc[mask]

# ==========================================================
# MULTI-PLANET DETECTION PIPELINE Optimised
# ==========================================================
from transitleastsquares import transitleastsquares

def detect_multiple_planets_tls(lc, max_planets=5, downsample_factor=20,
                                min_period=0.5, max_period=150, n_threads=2,
                                min_duration=0.01, max_duration=0.3):
    """
    Detect multiple planets in a light curve using TLS with speed optimizations.
    lc: Lightkurve LightCurve object
    max_planets: maximum number of planets to search for
    downsample_factor: take every Nth point
    n_threads: number of CPU threads
    min_duration/max_duration: limits for transit durations (fraction of period)
    """
    # Downsample the light curve
    lc_ds = lc[::downsample_factor]
    time = lc_ds.time.value
    flux = lc_ds.flux.value

    planets = []
    residual_flux = flux.copy()

    for i in range(max_planets):
        print(f"Searching for planet {i+1}...")

        model = transitleastsquares(time, residual_flux)
        result = model.power(
                                minimum_period=min_period,
                                maximum_period=max_period,
                                min_duration=min_duration,
                                max_duration=max_duration,
                                oversampling_factor=3,
                                n_threads=n_threads
                            )

        # Stop if no significant detection
        if result.SDE < 6:  # threshold can be tuned
            print("No more significant planets detected")
            break
        true_depth=1-result.depth
        planet = {
            "planet_id": i+1,
            "period": result.period,
            "transit_time": result.T0,
            "depth": true_depth,
            "duration": result.duration,
            "SDE": result.SDE
        }
        if any(abs(result.period - p["period"]) / p["period"] < 0.01 for p in planets):
            print("Duplicate period detected, skipping...")
            continue
        planets.append(planet)

        # Mask detected transit from residuals
        phase = ((time - result.T0 + 0.5*result.period) % result.period) - 0.5*result.period
        mask = np.abs(phase) > (result.duration/2 * 0.6)  # 10% padding
        # DO NOT remove time points (important fix)
        residual_flux[~mask] = 1.0

        # Diagnostics
        plot_residual_diagnostics(
            time,
            residual_flux,
            title=f"Residual Diagnostics After Planet {i+1}"
        )

        plot_phase_folded(
            time,
            residual_flux,
            result.period,
            result.T0,
            title=f"Phase Folded Residuals (Planet {i+1})"
        )

        odd_even_transit_test(
            time,
            residual_flux,
            result.period,
            result.T0
        )
    plt.show()
    result.save("tls_results.pkl")

    return planets

# ==========================================================
# STELLAR PARAMETERS AND HABITABLE ZONE
# ==========================================================
def query_star_data(star):
    try:
        table = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="hostname,st_mass,st_rad,st_lum",
            where=f"hostname like '{star}%'"
        )
        star_mass = float(np.asarray(table['st_mass'][0]).item())
        star_radius = float(np.asarray(table['st_rad'][0]).item())
        star_lum = float(10 ** np.asarray(table['st_lum'][0]).item())

        return {"mass": star_mass, "radius": star_radius, "luminosity": star_lum}

    except Exception as e:
        print(f"Stellar data problem: {e}, using solar defaults.")
        return {"mass":1.0, "radius":1.0, "luminosity":1.0}


def compute_planet_radius(star_radius, depth):
    return star_radius * np.sqrt(depth)


def compute_orbital_distance(period_days, star_mass):
    period_years = period_days / 365
    return (period_years ** (2/3)) * (star_mass ** (1/3))


def compute_habitable_zone(luminosity):
    inner = np.sqrt(luminosity / 1.1)
    outer = np.sqrt(luminosity / 0.53)
    return inner, outer


def check_habitability(distance, luminosity):
    inner, outer = compute_habitable_zone(luminosity)
    return inner <= distance <= outer

# ==========================================================
# PLANET PARAMETER CALCULATION
# ==========================================================
def compute_planet_parameters(star, planets):
    star_data = query_star_data(star)
    results = []

    for planet in planets:
        radius = compute_planet_radius(star_data["radius"], planet["depth"])
        orbit = compute_orbital_distance(planet["period"], star_data["mass"])
        habitable = check_habitability(orbit, star_data["luminosity"])

        record = {
            "star": star,
            "planet_id": int(planet["planet_id"]),
            "period_days": float(planet["period"]),
            "transit_time": float(planet["transit_time"]),  # ✅ ADD THIS
            "transit_depth": float(planet["depth"]),        # also standardize
            "radius": float(radius),
            "orbit_AU": float(orbit),
            "habitable": bool(habitable)
        }
        results.append(record)

    return pd.DataFrame(results)

# ==========================================================
# SAVE PLANET CATALOG
# ==========================================================
def save_catalog(df):
    try:
        old = pd.read_csv(CATALOG_FILE)
        df = pd.concat([old, df])
    except:
        pass

    # df = df.drop_duplicates(subset=["star","planet_id"])
    df.to_csv(CATALOG_FILE, index=False)

# ==========================================================
# VISUALIZATION
# ==========================================================
def plot_lightcurve(lc):
    plt.figure(figsize=(10,4))
    plt.plot(lc.time.value, lc.flux.value, ".k")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.title("Cleaned Light Curve")
    plt.show()


def launch_dashboard():
    df = pd.read_csv(CATALOG_FILE)
    fig = px.scatter(df, x="orbit_AU", y="radius", color="habitable", size="radius",
                     hover_data=["star","planet_id","period_days"], log_x=True,
                     title="Detected Exoplanets")
    fig.update_layout(template="plotly_dark", xaxis_title="Orbital Distance (AU)", yaxis_title="Planet Radius (Solar Radii)")

    app = Dash(__name__)
    app.layout = html.Div([html.H1("Exoplanet Detection Dashboard"), dcc.Graph(figure=fig)])
    app.run(debug=True)

# ==========================================================
# STAR ANALYSIS PIPELINE
# ==========================================================
def analyze_star(target):
    lc = download_target_data(target)
    lc = clean_lightcurve(lc)
    lc = smooth_lightcurve(lc)
    plot_lightcurve(lc)

     # ======================
    # DOWNsampling for TLS
    # ======================
    # downsample_factor = 20   # take every 5th point
    # lc = lc[::downsample_factor]

    planets =  detect_multiple_planets_tls(lc,
                                      max_planets=MAX_PLANETS_PER_STAR,
                                      downsample_factor=20,
                                      min_period=MIN_PERIOD,
                                      max_period=MAX_PERIOD,
                                      n_threads=2)
    if len(planets) == 0:
        print("No planets detected")
        return None

    df = compute_planet_parameters(target, planets)
    return df


def scan_star_list(star_list):
    all_results = []
    for star in star_list:
        try:
            df = analyze_star(star)
            if df is not None:
                all_results.append(df)
        except Exception as e:
            print(f"Error with {star}: {e}")

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        save_catalog(final)

# ==========================================================
# MAIN PROGRAM
# ==========================================================
if __name__ == "__main__":
    stars = ["Kepler-11"]
    scan_star_list(stars)
    print("\nPlanet catalog saved")
