# Urban-AirCast: High-Fidelity Machine Learning & Surrogate Time-Series Forecasting for Urban Atmospheric Pollutants ($PM_{2.5}$ & $NO_2$)

---

## 1. Executive Summary & Research Motivation

Urban atmospheric contamination by fine particulate matter with aerodynamic diameter $\le 2.5\,\mu\text{m}$ ($PM_{2.5}$) and nitrogen dioxide ($NO_2$) constitutes an urgent global environmental health hazard. According to World Health Organization (WHO) epidemiological benchmarks, chronic and acute exposure to elevated $PM_{2.5}$ induces severe cardiovascular mortality, ischemic heart disease, and pulmonary pathology.

```
+--------------------------------------------------------------------------------------------------+
|                                    ATMOSPHERIC PREDICTION PARADIGM                               |
+------------------------------------+-------------------------------------------------------------+
| Classical Physical Models (CTMs)   | • Solves discretized Navier-Stokes & atmospheric chemistry  |
| (e.g., WRF-Chem, CMAQ)             | • Prohibitive computational wall-clock overhead             |
|                                    | • Highly vulnerable to emission inventory staleness         |
+------------------------------------+-------------------------------------------------------------+
| Machine Learning Surrogate Models  | • Learns nonlinear mappings from continuous sensor telemetry|
| (Urban-AirCast Framework)          | • Near-instantaneous forward inference (~1.4 ms per step)   |
|                                    | • Adaptively captures complex seasonal & diurnal trends     |
+------------------------------------+-------------------------------------------------------------+

```

While deterministic Chemical Transport Models (CTMs) like WRF-Chem and CMAQ capture atmospheric transport and chemical kinetic transformations from first principles, their execution requires supercomputing clusters and detailed emission inventories that are frequently outdated.

**Urban-AirCast** provides a high-throughput, data-driven surrogate architecture that achieves high predictive accuracy across multi-horizon steps ($t+1$) while remaining computationally lightweight. The framework systematically resolves critical failure points of conventional time-series regressors: non-stationarity, extreme seasonal shifts (e.g., winter thermal inversions), lookahead data leakage, and the black-box opacity of tree ensembles through Shapley Additive exPlanations (SHAP).

---

## 2. Theoretical Framework & Mathematical Formulation

### 2.1 Problem Formulation

Let continuous environmental telemetry across $K$ monitoring stations be represented as a multivariate time series matrix:

$$\mathbf{X}_{1:T} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T\}, \quad \mathbf{x}_t \in \mathbb{R}^D$$

where $D$ denotes the feature dimensionality comprising primary criteria pollutants ($PM_{2.5}, PM_{10}, NO, NO_2, NO_x, NH_3, CO, SO_2, O_3$), cyclic calendar indicators, and derived rolling metrics. The objective is to learn a parameterized surrogate operator $\mathcal{F}_\theta: \mathbb{R}^{W \times D} \to \mathbb{R}^+$ mapping a historical observation window $W$ to the target concentration at a future horizon $h$:

$$\hat{y}_{t+h} = \mathcal{F}_\theta\left(\mathbf{X}_{t-W+1:t}\right)$$

### 2.2 Target Transformation for Variance Stabilization

Environmental time series for particulates exhibit heavy positive skewness ($\gamma_1 > 1.8$) due to extreme winter smog events, biomass burning, and temperature inversions. Training mean squared error (MSE) regressors on raw target values yields unstable gradients dominated by severe outliers, resulting in systematic underprediction of baseline concentrations and instability in tree partitions.

To stabilize conditional variance, we map the target space via a natural log transform:

$$z_{t+h} = \ln(1 + y_{t+h})$$

Surrogate regressors optimize loss in the transformed space $\mathcal{Z}$. Model predictions are inverted back to physical concentration space ($\mu\text{g}/\text{m}^3$) via the exponential dual operator:

$$\hat{y}_{t+h} = \max\left(0, \; \exp(\hat{z}_{t+h}) - 1\right)$$

Ensuring that physical concentrations strictly satisfy the non-negativity constraint $\hat{y}_{t+h} \in [0, \infty)$.

### 2.3 Cyclic Harmonic Encodings (Fourier Decomposition)

Standard integer encodings of temporal features (such as Day of Year $d \in [1, 365]$ and Day of Week $w \in [0, 6]$) introduce artificial numerical discontinuities at boundary transitions (e.g., between December 31 and January 1). To project calendar dynamics onto a continuous, smooth manifold, we implement harmonic Fourier encodings:

$$\phi_{\text{annual}}(t) = \left[ \sin\left(\frac{2\pi \cdot d(t)}{365.25}\right), \; \cos\left(\frac{2\pi \cdot d(t)}{365.25}\right) \right]^T$$

$$\phi_{\text{weekly}}(t) = \left[ \sin\left(\frac{2\pi \cdot w(t)}{7}\right), \; \cos\left(\frac{2\pi \cdot w(t)}{7}\right) \right]^T$$

---

## 3. End-to-End System Architecture

```text
                                 DATA INGESTION
                                       │
                      Raw CPCB Daily Surface Telemetry
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. TIME-AWARE PREPROCESSING & IMPUTATION                                    │
│    • Set DatetimeIndex for chronological consistency                        │
│    • Time-weighted interpolation for localized gaps (Δt ≤ 3 days)           │
│    • Strict backward-fill for cold-start historical boundary values         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. REPRODUCIBLE FEATURE ENGINEERING PIPELINE                                │
│    • Harmonic Fourier Features: Annual [sin/cos], Weekly [sin/cos]          │
│    • Autoregressive Lags: L ∈ {1, 2, 3, 7} days                             │
│    • Causal Moving Windows: 7-day and 14-day rolling mean & standard dev    │
│    • Strict Lookahead Prevention: Shifting windows by t-1 prior to stats    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. RIGOROUS MODEL BENCHMARKING (Expanding Window TimeSeriesSplit)           │
│    • L2 Scaled Ridge Regression (Analytical Baseline)                       │
│    • Random Forest Regressor (Bootstrap Aggregation)                        │
│    • XGBoost (Histogram-based gradient boosted trees with L1/L2 penalties)  │
│    • LightGBM (Leaf-wise Gradient-based One-Side Sampling)                  │
│    • CatBoost (Ordered Boosting with oblivious decision trees)              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. STATISTICAL VALIDATION & EXPLAINABILITY (XAI)                            │
│    • Out-of-sample holdout prediction and 45° Parity scatter analysis       │
│    • Residual Gaussianity and zero-centering evaluation                     │
│    • SHAP TreeExplainer: Global attribution and cross-pollutant interactions│
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 4. Feature Engineering & Selection Strategy

To capture chemical formation kinetics, boundary-layer mechanics, and autoregressive memory, the feature space is configured as follows:

| Feature Category | Feature Name | Formula / Definition | Physical Justification |
| --- | --- | --- | --- |
| **Harmonic Time** | `sin_year`, `cos_year` | $\sin/\cos(2\pi \cdot \text{day} / 365.25)$ | Tracks seasonal thermal inversions, winter monsoons, and ambient temperature shifts. |
| **Harmonic Time** | `sin_week`, `cos_week` | $\sin/\cos(2\pi \cdot \text{weekday} / 7)$ | Encodes human-driven mobility patterns and weekend industrial emission reductions. |
| **Autoregressive Lags** | `PM2.5_lag1`, `PM2.5_lag2`, `PM2.5_lag3` | $y_{t-1}, y_{t-2}, y_{t-3}$ | Captures atmospheric residence times and high-frequency autoregressive persistence. |
| **Weekly Lag** | `PM2.5_lag7` | $y_{t-7}$ | Accounts for weekly cyclicality in commuter traffic and freight logistics. |
| **Atmospheric Memory** | `PM2.5_roll_mean_7` | $\frac{1}{7}\sum_{i=1}^7 y_{t-i}$ | Captures sustained synoptic weather systems and regional background pollution. |
| **Atmospheric Volatility** | `PM2.5_roll_std_7` | $\sqrt{\frac{1}{6}\sum_{i=1}^7 (y_{t-i} - \bar{y})^2}$ | Quantifies atmospheric turbulence, dispersion conditions, and frontal passages. |
| **Long-Term Baseline** | `PM2.5_roll_mean_14` | $\frac{1}{14}\sum_{i=1}^{14} y_{t-i}$ | Isolates broad multi-week trends from high-frequency transient spikes. |
| **Precursor Dynamics** | `NO`, `NO2`, `NOx`, `NH3` | Direct sensor inputs ($\mu\text{g}/\text{m}^3$) | Critical precursors for secondary inorganic aerosol formation (e.g., ammonium nitrate). |
| **Combustion Tracers** | `CO`, `SO2` | Direct sensor inputs ($\mu\text{g}/\text{m}^3$, $\text{mg}/\text{m}^3$) | Serves as conservative tracers for vehicular exhaust, coal combustion, and industrial activity. |
| **Secondary Oxidants** | `O3` | Ground-level Ozone ($\mu\text{g}/\text{m}^3$) | Driver of atmospheric oxidation capacity converting $\text{SO}_2$ and $\text{NO}_x$ into particulates. |

> **Anti-Leakage Safeguard:** All rolling window computations apply a 1-step backward shift (`df['PM2.5'].shift(1).rolling(...)`). Computing rolling statistics on unshifted arrays exposes values at time $t$ to historical summaries, causing severe test-set leakage.

---

## 5. Experimental Validation Setup

### 5.1 Temporal Cross-Validation (TimeSeriesSplit)

Standard $k$-fold cross-validation is **invalid** for time series because randomly shuffling records allows the model to train on future observations to predict the past, yielding artificially inflated metrics that collapse during deployment.

We implement an **expanding-window forward-chaining cross-validation** scheme ($K=5$ splits):

```text
Fold 1:  [ Train: 16.6% ] ──> [ Val: 16.6% ]
Fold 2:  [ Train: 33.3% ───────── ] ──> [ Val: 16.6% ]
Fold 3:  [ Train: 50.0% ────────────────── ] ──> [ Val: 16.6% ]
Fold 4:  [ Train: 66.6% ─────────────────────────── ] ──> [ Val: 16.6% ]
Fold 5:  [ Train: 83.3% ──────────────────────────────────── ] ──> [ Val: 16.6% ]

```

### 5.2 Quantitative Performance Metrics

Model performance is evaluated using three complementary metrics computed on inverted physical values ($\mu\text{g}/\text{m}^3$):

1. **Root Mean Squared Error (RMSE):** Strongly penalizes large outlier errors, critical for severe smog events.

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N \left(y_i - \hat{y}_i\right)^2}$$


2. **Mean Absolute Error (MAE):** Represents average expected physical error magnitude, less sensitive to extreme points.

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^N \left\vert{}y_i - \hat{y}_i\right\vert{}$$


3. **Coefficient of Determination ($R^2$ Score):** Proportion of variance explained by the model relative to a naive mean baseline.

$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$



---

## 6. Benchmark Results & Comparative Evaluation

The empirical benchmark across all models evaluated under the 5-fold expanding-window cross-validation on Delhi surface monitoring data is summarized below:

| Model Architecture | Mean RMSE ($\mu\text{g}/\text{m}^3$) | Mean MAE ($\mu\text{g}/\text{m}^3$) | Mean $R^2$ Score | Wall-Clock Inference (ms/sample) |
| --- | --- | --- | --- | --- |
| **Random Forest Regressor** | **42.236** | **26.456** | **0.681** | $0.082$ |
| **LightGBM Regressor** | **42.853** | **27.132** | **0.670** | **0.014** |
| **XGBoost Regressor** | 43.480 | 27.749 | 0.661 | $0.021$ |
| **CatBoost Regressor** | 44.142 | 28.401 | 0.651 | $0.045$ |
| **Baseline (Scaled Ridge)** | 52.550 | 30.352 | 0.469 | $0.003$ |

### Performance Analysis

* **Ensemble Superiority:** Tree-based ensemble regressors consistently outperform regularized linear baselines, lowering RMSE by over $10\,\mu\text{g}/\text{m}^3$ and improving $R^2$ from $0.469$ to $0.681$. This confirms the non-linear relationship between precursor gases, harmonic seasonal forcing, and particulate formation.
* **Random Forest vs. Boosting:** Random Forest achieved the lowest overall RMSE ($42.236\,\mu\text{g}/\text{m}^3$) due to its variance-reducing bootstrap aggregation across stochastic atmospheric swings. LightGBM delivers comparable accuracy ($42.853\,\mu\text{g}/\text{m}^3$) with an inference latency of $14\,\mu\text{s}$ per query, making it well-suited for high-throughput edge deployment.

---

## 7. Model Diagnostics & Interpretability (XAI)

### 7.1 Actual vs. Predicted Parity & Residual Structure

The performance of the final LightGBM surrogate evaluated on out-of-sample data is illustrated below:

* **Parity Alignment (Left):** Predictions trace the $45^\circ$ line of identity closely across low-to-medium concentration ranges ($30 - 180\,\mu\text{g}/\text{m}^3$). High-concentration events ($>250\,\mu\text{g}/\text{m}^3$) show minor underprediction, a recognized boundary limitation of tree-based architectures when extrapolating outside training leaf ranges.


* **Residual Gaussian Distribution (Right):** The residual error density function ($\epsilon = y_{\text{true}} - \hat{y}_{\text{pred}}$) exhibits clean zero-centering and near-Gaussian symmetry. Applying logarithmic transformation eliminates the heavy negative skew observed in unregularized models, preventing systematic over- or under-forecasting bias across baseline periods.



### 7.2 Global SHAP Attribution Analysis

To open the black-box mechanics of the surrogate model, we compute Shapley Additive exPlanations using TreeExplainer based on game theory:

* **Primary Explanatory Mass:** Same-day $PM_{2.5}(t)$ concentration and seasonal cosine harmonic encoding (`cos_year`) drive the largest attribution mass. High positive values of `cos_year` (corresponding to winter months: November to January) strongly increase predicted $PM_{2.5}$, capturing regional temperature inversions that trap particulates near the surface.


* **Coarse Particulate Coupling:** Co-emitted coarse particulates (`PM10`) and multi-day moving trends (`PM2.5_roll_mean_14`) emerge as the next most influential predictors.


* **Secondary Precursor Influence:** Elevated levels of combustion precursors ($CO$, $NO_2$, $NO$) systematically push SHAP attributions positively, capturing secondary aerosol formation pathways where gaseous emissions oxidize into fine particulate mass.



---

## 8. Repository Structure

The project follows a modular, reproducible package hierarchy:

```text
urban-air-quality-forecasting/
├── .gitignore                      <- Prevents committing cache, checkpoints, and data
├── LICENSE                         <- Permissive MIT open-source license
├── README.md                       <- Academic report and documentation
├── main.py                         <- End-to-end orchestration pipeline
├── requirements.txt                <- Locked environment dependencies
│
├── data/
│   ├── raw/                        <- Untracked original dataset (city_day.csv)
│   └── processed/                  <- Engineered, time-aligned feature dataset
│
├── figures/                        <- High-resolution diagnostic outputs
│   ├── benchmark_results.csv       <- Leaderboard metrics
│   ├── lightgbm_eval.png           <- Parity and residual plots
│   └── shap_summary.png            <- SHAP beeswarm feature importance
│
├── notebooks/
│   ├── 01_eda_and_profiling.ipynb   <- Data distributions, missingness, and correlation
│   ├── 02_feature_engineering.ipynb <- Lags, rolling windows, and cyclical transforms
│   └── 03_model_benchmarking.ipynb  <- Cross-validation and model comparison
│
└── src/
    ├── __init__.py                 <- Exposes core pipeline functions
    ├── data_pipeline.py            <- Imputation and feature extraction
    ├── evaluate.py                 <- Plotting routines (parity and residuals)
    ├── interpret.py                <- SHAP TreeExplainer routines
    └── models.py                   <- Cross-validation and model factory

```

---

## 9. Reproducibility & Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/johjoh12/urban-air-quality-forecasting.git
cd urban-air-quality-forecasting

```

### Step 2: Establish an Isolated Environment

```bash
# Using standard Python venv
python -m venv .venv

# Activate on Linux / macOS:
source .venv/bin/activate

# Activate on Windows (cmd/PowerShell):
.venv\Scripts\activate

```

### Step 3: Install Pinned Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### Step 4: Dataset Placement

1. Download `city_day.csv` from the [Kaggle: Air Quality Data in India Benchmark](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).
2. Create the target directory and place the CSV file inside:
```bash
mkdir -p data/raw
# Move city_day.csv into data/raw/city_day.csv

```



### Step 5: Execute the End-to-End Pipeline

Execute the root pipeline script:

```bash
python main.py

```

This script handles the full workflow:

1. Loads and filters raw station records for Delhi.
2. Applies time-indexed gap interpolation and generates harmonic and lag features.
3. Performs 5-fold expanding-window cross-validation across all models.
4. Generates and exports the benchmark leaderboard to `figures/benchmark_results.csv`.
5. Fits the final LightGBM surrogate on the holdout partition.
6. Computes out-of-sample predictions, creates parity and residual plots (`figures/lightgbm_eval.png`), and computes SHAP value attributions (`figures/shap_summary.png`).

---

## 10. Research Limitations & Future Roadmap

* **Wind Vector Integration:** In the current setup, only pollutant telemetry is modeled. Integrating surface vector wind components ($u, v$), planetary boundary layer height (PBLH), and relative humidity from ECMWF ERA5 reanalysis will help resolve wind dispersion and stagnant air events.
* **Deep Sequence Architectures:** Extending the surrogate benchmark to transformer-based long-horizon forecasting models (e.g., PatchTST, TiDE) to evaluate 72-hour continuous multi-step trajectories.
* **Spatial Graph Neural Networks (GNNs):** Formulating the urban sensor network as a spatial-temporal graph (e.g., Spatio-Temporal Graph Convolutional Networks, STGCN) to capture pollutant transport between monitoring stations.

---

## 11. Citation & Academic Reference

If you build upon this pipeline, use the surrogate modeling formulation, or reference the results in academic work, please cite:

```bibtex
@misc{urban_aircast_2026,
  author = {Akingbade Jonathan Oluwadunsin},
  title = {Urban-AirCast: High-Fidelity Machine Learning and Surrogate Time-Series Forecasting for Urban Atmospheric Pollutants},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/johjoh12/urban-air-quality-forecasting}}
}

```

## 12. License

This repository is open-sourced under the **MIT License**. Refer to the [LICENSE](https://www.google.com/search?q=LICENSE) file for complete terms and permissions.