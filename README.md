# 🌌 Exoplanet Detection Pipeline (TLS + Multi-Planet Modeling)

A research-oriented pipeline for detecting and analyzing exoplanets using transit photometry from Kepler data. This project implements a full workflow including light curve preprocessing, transit detection using Transit Least Squares (TLS), multi-planet identification, and habitability estimation.

> 📦 Datasets are not included. See [Datasets & Cache](#-datasets--cache) for setup instructions.

---

## 🚀 Features

* 📡 Load and process Kepler light curve FITS data (offline support)
* 🧹 Advanced light curve cleaning and smoothing
* 🔍 Transit detection using Transit Least Squares (TLS)
* 🪐 Multi-planet detection with iterative masking
* 📊 Residual diagnostics and phase-folded visualization
* 🌍 Habitability analysis using stellar parameters
* 📁 Export detected planets to CSV catalog
* 📈 Interactive dashboard for visualization (Plotly + Dash)

---

## 🧠 Pipeline Overview

1. **Data Loading**

   * Reads `.fits` light curve files from local storage
   * Supports offline datasets to avoid server dependency

2. **Preprocessing**

   * Removes NaNs and outliers
   * Normalizes and flattens the light curve
   * Applies Savitzky-Golay smoothing

3. **Transit Detection**

   * Uses Transit Least Squares (TLS)
   * Detects strongest periodic transit signals

4. **Multi-Planet Detection**

   * Iteratively masks detected transits
   * Searches for additional planets in residuals

5. **Post-processing**

   * Computes planetary radius, orbital distance
   * Evaluates habitability zone

6. **Visualization**

   * Residual diagnostics
   * Phase-folded transit plots
   * Interactive dashboard

---

## ⚙️ Tech Stack

* Python 3.10+
* `lightkurve`
* `transitleastsquares`
* `numpy`, `pandas`, `matplotlib`
* `scipy`
* `astropy`
* `astroquery`
* `plotly`, `dash`

---

## 📂 Project Structure

```
Research_project/
│
├── code/
│   ├── exoplanets.py          # Main pipeline
│   ├── validation.py          # Validation logic
│   ├── jointMultiplanet.py    # Joint modeling (MCMC)
│
├── datasets/
│   └── cache/tpf/             # Local FITS files
│
├── outputs/
│   └── detectedPlanets.csv    # Output catalog
│
└── README.md
```

---

## 📦 Datasets & Cache

> ⚠️ **Note:** This repository does **not include datasets or cache files**.

The Kepler light curve FITS files and intermediate cache data are intentionally excluded because:

* 📁 They are **very large (GB-scale)**
* 🔁 They can be **easily re-downloaded from public archives**
* 🚫 Including them would make the repository unnecessarily heavy

---

### 📥 How to Obtain Data

You can download the required data from:

* **MAST (Mikulski Archive for Space Telescopes)**
* NASA Exoplanet Archive

Or directly via Python using:

```python
import lightkurve as lk

lc = lk.search_lightcurve("Kepler-11", mission="Kepler").download()
```

---

### 📁 Expected Directory Structure

After downloading, place files like this:

```id="k8x2qf"
datasets/
└── cache/
    └── tpf/
        ├── Kepler-11.fits
        ├── Kepler-90.fits
```

---

### ⚡ Tip (Recommended)

For faster and reproducible runs:

* Use **offline FITS files** (as done in this project)
* Avoid repeated downloads from MAST (can be slow or unstable)

---

### 🚫 Ignored Files

Make sure your `.gitignore` includes:

```id="z8h3af"
datasets/
*.fits
*.csv
.cache/
```

---

This ensures your repo stays clean and lightweight while still being fully reproducible.


---

## 🛠️ Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/exoplanet-pipeline.git
cd exoplanet-pipeline
```

### 2. Install dependencies

```bash
pip install lightkurve transitleastsquares astropy astroquery scipy plotly dash
```

---

## ▶️ Usage

### Run the pipeline

```bash
python exoplanets.py
```

### Modify target stars

Edit inside `exoplanets.py`:

```python
stars = ["Kepler-11"]
```

---

## ⚡ Performance Optimizations

* Downsampling of light curves (~20x speedup)
* Restricted period search range
* Multi-threaded TLS execution
* Offline FITS usage (no network dependency)

---

## 📊 Output

The pipeline generates:

* `detectedPlanets.csv` containing:

  * Period
  * Transit depth
  * Planet radius
  * Orbital distance
  * Habitability status

---

## 🌍 Habitability Criteria

* Based on stellar luminosity
* Uses conservative habitable zone estimates
* Checks if planet orbit lies within zone

---

## ⚠️ Known Issues

* TLS can be slow for very large datasets
* FITS format variations may require preprocessing tweaks
* MAST server instability (handled via offline mode)

---

## 🔬 Future Work

* GPU acceleration for transit search
* Improved MCMC-based joint modeling
* Automated dataset ingestion
* Integration with TESS data
* Web-based dashboard deployment

---

## 📸 Sample Output

* Phase-folded transit plots
* Residual diagnostics
* Multi-planet detection logs

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📜 License

This project is open-source under the MIT License.

---

## 👨‍💻 Author

**Badari Narayana K**
Research Focus: Exoplanet Detection, Computational Astrophysics

---

## ⭐ Acknowledgements

* NASA Kepler Mission
* Lightkurve Team
* Transit Least Squares (TLS) developers

---

## 💡 Note

This project is designed as a **research-grade pipeline**, not just a demo. It reflects real-world challenges like noisy data, incomplete observations, and computational constraints.
