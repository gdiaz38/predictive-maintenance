# ⚙️ Predictive Maintenance IoT Dashboard

An XGBoost-based Remaining Useful Life (RUL) predictor for industrial turbofan engines, trained on NASA CMAPSS data with 60 engineered rolling-window features. Predicts engine failure with MAE of 7.68 cycles and 2.99 cycles in the critical zone.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-RUL-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📊 Live Dashboard

👉 **[View Live App](https://predictive-maintenance-gdiaz38.streamlit.app/)**

---

## Overview

Unplanned industrial equipment failure costs manufacturers an estimated $50B annually. This project builds a predictive maintenance system that estimates how many cycles remain before a turbofan engine fails — enabling maintenance teams to act before breakdown rather than after.

Key question it answers: *Given the last 30 cycles of sensor readings, how many cycles does this engine have left?*

---

## Key Results

| Metric | Value |
|---|---|
| MAE | **7.68 cycles** |
| RMSE | 11.74 cycles |
| Critical zone MAE (RUL ≤ 25) | **2.99 cycles** |
| Within ±10 cycles | ~65% of predictions |
| Features | 60 (14 sensors × 3 rolling stats + base) |

---

## Features

- **Real XGBoost inference** — trained model loaded and run on every session
- **Fleet overview** — predicted vs true RUL across all 80 test engines
- **Engine trajectory** — full degradation curve from healthy to critical for any engine
- **Model performance tab** — predicted vs actual scatter, error distribution, MAE by RUL bucket
- **Real-time alert simulation** — streams one engine's actual sensor data cycle by cycle, updating status from HEALTHY → WARNING → CRITICAL
- **Feature importance** — which of the 60 engineered features matter most

---

## Data

| Source | Description |
|---|---|
| [NASA CMAPSS FD001](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) | 100 training engines, 100 test engines |
| Sensors | 21 sensors per cycle, 14 with meaningful variance |
| RUL cap | 125 cycles (piecewise linear — standard for CMAPSS) |

---

## Project Structure

```
predictive-maintenance/
├── dashboard.py          # Streamlit — 4 tabs: fleet, trajectory, performance, simulation
├── train.py              # Original XGBoost training script
├── retrain.py            # Retrain with sample weights (fixes RUL cap bias)
├── features.py           # Rolling window feature engineering from raw CMAPSS
├── api.py                # FastAPI REST endpoint (local use)
├── xgb_rul.pkl           # Trained XGBoost model
├── scaler_rul.pkl        # MinMaxScaler (applied during feature engineering only)
├── feature_cols.pkl      # Feature column names (60 features)
├── X_train.npy           # Pre-scaled training features
├── X_test.npy            # Pre-scaled test features
├── y_train.npy           # Training RUL labels
├── y_test.npy            # Test RUL labels
└── requirements.txt
```

---

## How It Works

```
Raw CMAPSS sensor readings (21 sensors per cycle)
        ↓
features.py selects 14 meaningful sensors
Computes 30-cycle rolling mean, std, slope per sensor
= 60 total features per cycle, scaled to [0,1]
        ↓
XGBoost predicts RUL — trained with sample weights:
  4× weight for RUL ≤ 30 cycles
  2× weight for RUL ≤ 60 cycles
  1× weight for healthy cycles
        ↓
Predictions clipped to [0, 125]
Status: Critical ≤25 | Warning ≤50 | Caution ≤100 | Healthy
```

---

## Feature Engineering

14 sensors with meaningful variance across the engine lifecycle:
`s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`

For each sensor, 3 rolling features over a 30-cycle window:

| Feature Type | Description |
|---|---|
| `{sensor}_mean30` | Smoothed signal level |
| `{sensor}_std30` | Variability increase as engine degrades |
| `{sensor}_slope30` | Drift direction and rate |

Plus 14 raw sensor values and 4 operating condition features = **60 total**.

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Trees | 500 (early stopping at 50 rounds) |
| Max depth | 6 |
| Learning rate | 0.03 |
| Sample weights | 4× (RUL≤30), 2× (RUL≤60), 1× (else) |

**MAE by RUL bucket:**

| Zone | RUL Range | MAE |
|---|---|---|
| 🔴 Critical | 0–25 | 2.99 cycles |
| 🟠 Warning | 25–50 | ~8 cycles |
| 🟡 Caution | 50–100 | ~9 cycles |
| 🟢 Healthy | 100–125 | ~12 cycles |

---

## Dashboard Tabs

**Fleet Overview** — predicted final RUL per engine colored by status tier, true RUL as diamond markers overlay

**Engine Trajectory** — select any of 80 test engines, view full degradation curve predicted vs actual with per-cycle error bars

**Model Performance** — scatter plot, error histogram, MAE by bucket, top 15 feature importances

**Alert Simulation** — stream one real engine's data cycle by cycle, watch RUL countdown and status change live

---

## Local Setup

```bash
git clone https://github.com/gdiaz38/predictive-maintenance
cd predictive-maintenance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

To retrain (requires NASA CMAPSS via kagglehub):

```bash
python3 features.py   # engineer features → .npy files
python3 retrain.py    # train with sample weights → xgb_rul.pkl
```

---

## Tech Stack

`Python 3.11` · `XGBoost` · `Streamlit` · `Plotly` · `Pandas` · `NumPy` · `Scikit-learn` · `joblib`

---

## Affiliation

University of California, Riverside — MS in Engineering Management
Part of a portfolio of 10 live data science projects spanning computer vision, NLP, supply chain, and healthcare ML.

---

## License

MIT
