# ==========================================================
# JOINT MULTI-PLANET TRANSIT FITTING WITH MCMC (STABLE)
# ==========================================================

import os
import pandas as pd
import numpy as np
import lightkurve as lk
import exoplanet as xo
import pymc as pm
import pytensor.tensor as tt
import matplotlib.pyplot as plt
import corner
from exoplanets import plot_residual_diagnostics

# -------------------------
# Thread control (Windows fix)
# -------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ==========================================================
# CONFIG
# ==========================================================
print("Using catalog:", CATALOG_FILE)
CATALOG_FILE = "/content/drive/MyDrive/Research_Project/outputs/validated_candidates.csv"
TARGET_KIC = "6541920"  # Kepler-11
N_DRAWS = 200
N_TUNE = 200
DOWNSAMPLE_FACTOR = 20

# ==========================================================
# LOAD LIGHT CURVE
# ==========================================================
print("Loading light curve...")
lc = lk.search_lightcurve(f"KIC {TARGET_KIC}", mission="Kepler").download_all()
lc = lc.stitch().remove_nans().normalize()

lc = lc[::DOWNSAMPLE_FACTOR]

time = np.asarray(lc.time.value, dtype=float)
flux = np.asarray(lc.flux.value, dtype=float)

mask = np.isfinite(time) & np.isfinite(flux)
time = time[mask]
flux = flux[mask]

print(f"Light curve loaded. Points: {len(time)}")

# ==========================================================
# LOAD PLANET GUESSES
# ==========================================================
print("Loading candidate planets...")
df = pd.read_csv(CATALOG_FILE)
print(df.columns)
df_star = df[df["star"].astype(str) == TARGET_KIC].sort_values("period_days")

planet_guesses = [
    {
        "period": row["period_days"],
        "t0": row["transit_time"],
        "depth": row["transit_depth"]
    }
    for _, row in df_star.iterrows()
]

print(f"Planets detected: {len(planet_guesses)}")

# ==========================================================
# BUILD MODEL
# ==========================================================
print("Building joint model...")
assert len(time) == len(flux), "Time and flux length mismatch!"

with pm.Model() as model:

    # Convert time to tensor ONCE
    t_tensor = tt.as_tensor_variable(time)
    t_tensor = tt.reshape(t_tensor, (-1,))

    # Baseline
    mean_flux = pm.Normal("mean_flux", mu=1.0, sigma=0.01)

    # Initialize model ONCE
    flux_model = mean_flux + tt.zeros_like(t_tensor)

    # Planets
    for i, p in enumerate(planet_guesses):

        period = pm.Normal(f"period_{i}", mu=p["period"], sigma=0.01)
        t0 = pm.Normal(f"t0_{i}", mu=p["t0"], sigma=0.05)
        r = pm.HalfNormal(f"r_{i}", sigma=0.1)
        b = pm.Uniform(f"b_{i}", lower=0, upper=0.99)

        orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)

        ld = xo.LimbDarkLightCurve(u1=0.3, u2=0.2)

        lc_model = ld.get_light_curve(
            orbit=orbit,
            r=r,
            t=t_tensor
        )

        # IMPORTANT: keep shape consistent
        lc_model = lc_model[:, 0]
        lc_model = tt.reshape(lc_model, (-1,))

        # NaN protection
        lc_model = tt.switch(tt.isnan(lc_model), 1.0, lc_model)

        # Add planet signal
        flux_model += (lc_model - 1)

    # Noise
    sigma = pm.HalfNormal("sigma", sigma=0.001)

    pm.Normal("obs", mu=flux_model, sigma=sigma, observed=flux)

    # HARD stability guard
    pm.Potential(
        "finite_guard",
        tt.switch(tt.any(tt.isnan(flux_model)), -np.inf, 0)
    )

    # Sampling
    print("Running MCMC...")
    trace = pm.sample(
        draws=N_DRAWS,
        tune=N_TUNE,
        cores=1,
        target_accept=0.98,
        max_treedepth=14,
        init="adapt_diag",
        return_inferencedata=False
    )

    pm.Deterministic("final_model_flux", flux_model)

print("Sampling complete")

# ==========================================================
# EXTRACT MODEL + COMPARE WITH DATA
# ==========================================================

print("Extracting model flux...")

# Extract deterministic variable from trace
final_model = np.mean(trace["final_model_flux"], axis=0)

# Plot model vs data
plt.figure(figsize=(10,4))
plt.plot(time, flux, ".k", alpha=0.5, label="Data")
plt.plot(time, final_model, "r", label="Model")
plt.legend()
plt.title("Model vs Data")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.show()

# Residual diagnostics
plot_residual_diagnostics(
    time,
    flux,
    model_flux=final_model,
    title="Post-MCMC Residuals"
)


# ==========================================================
# RESULTS
# ==========================================================
print("\nPosterior Summary:\n")

for i in range(len(planet_guesses)):
    print(f"Planet {i+1}")
    print(f"Period: {trace[f'period_{i}'].mean():.5f}")
    print(f"Rp/R*: {trace[f'r_{i}'].mean():.5f}")
    print(f"Impact param: {trace[f'b_{i}'].mean():.3f}\n")

# ==========================================================
# CORNER PLOT
# ==========================================================
print("Generating corner plot...")

samples = []
labels = []

for i in range(len(planet_guesses)):
    samples += [
        trace[f"period_{i}"],
        trace[f"r_{i}"],
        trace[f"b_{i}"]
    ]
    labels += [f"P{i+1}", f"R{i+1}", f"b{i+1}"]

samples = np.column_stack(samples)

corner.corner(samples, labels=labels, show_titles=True)

plt.show()

print("Done.")
