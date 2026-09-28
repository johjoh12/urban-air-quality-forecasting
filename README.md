# Urban-AirCast: Machine Learning & Surrogate Time-Series Forecasting for Urban Air Pollutants ($PM_{2.5}$ & $NO_2$)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end applied machine learning and surrogate modeling framework designed to forecast multi-horizon urban air pollutant concentrations ($PM_{2.5}$ and $NO_2$). Developed using historical atmospheric monitoring data from the Central Pollution Control Board (CPCB), the architecture integrates time-aware interpolation, multi-scale cyclic harmonic encoding, lag dynamics, and tree-ensemble surrogates benchmarked against expanding-window cross-validation.

---

## Executive Summary & Abstract

Particulate Matter ($PM_{2.5}$) and Nitrogen Dioxide ($NO_2$) are critical urban atmospheric pollutants directly linked to acute cardiopulmonary morbidity and meteorological degradation. While traditional Numerical Weather Prediction (NWP) and chemical transport models (CTMs) require massive computational resources and extensive emission inventories, **data-driven surrogate models** provide rapid, high-fidelity multi-step forecasts.

This study implements a multi-model comparative analysis across:
* **Linear Baseline:** Regularized Ridge Regression with standard scaling.
* **Nonlinear Ensembles:** Random Forest, Extreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine (LightGBM), and CatBoost.
* **Explainable AI (XAI):** Tree-based Shapley Additive exPlanations (SHAP) to interpret global feature attribution and transient chemical precursor dependencies.

---

## System Architecture & Modeling Pipeline

```text
Raw CPCB Sensor Data (city_day.csv)
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Preprocessing & Feature Engineering                    │
│  - Datetime index re-alignment                         │
│  - Time-weighted gap interpolation (limit = 3 days)    │
│  - Cyclic harmonic Fourier terms (Annual & Weekly)     │
│  - Autoregressive lag operators (t-1, t-2, t-3, t-7)   │
│  - Shifted rolling statistics (7-day & 14-day windows) │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Model Benchmarking (5-Fold TimeSeriesSplit)            │
│  - Ridge (L2 regularized baseline)                     │
│  - Random Forest Regressor                             │
│  - XGBoost Regressor                                   │
│  - LightGBM Regressor                                  │
│  - CatBoost Regressor                                  │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Validation & Explainability                            │
│  - Time-ordered out-of-sample evaluation               │
│  - Actual vs. Predicted Parity Analysis                │
│  - Residual Error Diagnostics                          │
│  - SHAP Global Attribution (TreeExplainer)             │
└────────────────────────────────────────────────────────┘