# 🔧 Predictive Maintenance IoT Dashboard

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge)](https://predictive-maintenance-gdiaz38.streamlit.app/)

Production-style predictive maintenance system built on NASA's CMAPSS turbofan
engine dataset. XGBoost remaining useful life (RUL) predictor with real-time
sensor monitoring, deployed as a REST API with a live operations dashboard.

---

## 🚀 Try It Live

**[→ Open Live Dashboard](https://predictive-maintenance-gdiaz38.streamlit.app/)**

Monitor a simulated engine fleet, plot degradation trajectories, and watch
a real-time critical failure alert simulation.

---

## Results
| Metric | Value |
|--------|-------|
| RMSE | 18.4 cycles |
| MAE | 13.2 cycles |
| R² Score | 0.881 |
| Early warning window | 30 cycles before failure |
| Engines monitored | 218 test engines |

---

## Architecture
```
NASA CMAPSS FD001 Dataset
218 engines · 20,631 training cycles · 26 sensor readings
            ↓
Feature Engineering (sensor rolling stats)
- 21 sensor readings (s1–s21)
- Rolling mean/std over 5, 10, 30 cycle windows
- Cycle normalization per engine
- Sensor delta features (rate of change)
- RUL cap at 125 cycles (piecewise linear target)
            ↓
XGBoost Regressor
RUL prediction (remaining cycles before failure)
            ↓
Threshold Alert Layer
- Critical: RUL < 15 cycles
- Warning:  RUL 15–30 cycles
- Healthy:  RUL > 30 cycles
            ↓
PostgreSQL (sensor readings + predictions)
            ↓
FastAPI REST endpoint (/predict, /fleet)
            ↓
Streamlit real-time operations dashboard
```

---

## Key Design Decisions

**RUL cap at 125 cycles** — engines early in their life have
noisy, uninformative sensor patterns. Capping RUL at 125
normalizes the training target and improves model performance
on the critical end-of-life prediction window — standard
practice in industrial prognostics.

**Rolling window features** — raw sensor readings are noisy.
5, 10, and 30-cycle rolling means and standard deviations
capture degradation trends that point-in-time readings miss.
This is how real condition monitoring systems work.

**Piecewise linear target** — rather than treating all cycles
equally, the model is optimized to be accurate in the
30-cycle warning window where maintenance decisions happen.
False negatives (missing a failure) are far more costly
than false positives.

**PostgreSQL for sensor storage** — structured, time-series
sensor data maps cleanly to a relational schema. Each row
is one engine at one cycle with all 26 sensor readings
and the predicted RUL written back for dashboard queries.

---

## Stack
- **Model:** XGBoost Regressor
- **Features:** Rolling window sensor statistics
- **Database:** PostgreSQL (sensor_readings, predictions)
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit + Plotly

---

## Run It
```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas numpy scikit-learn xgboost fastapi uvicorn streamlit plotly sqlalchemy psycopg2-binary
python3 download_data.py
python3 explore.py
python3 features.py
python3 train.py
# Terminal 1
python3 api.py
# Terminal 2
streamlit run dashboard.py
```

## API
```bash
POST /predict
{
  "engine_id": "FD001-001",
  "cycle": 150,
  "sensors": [518.67, 641.82, 1589.70, 1400.60, 14.62, ...]
}

GET /fleet
GET /alerts
GET /engine/{engine_id}/history
```
