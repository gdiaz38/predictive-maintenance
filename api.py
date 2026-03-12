import numpy as np
import joblib
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
model       = joblib.load("xgb_rul.pkl")
scaler      = joblib.load("scaler_rul.pkl")
feature_cols = joblib.load("feature_cols.pkl")
print(f"Model loaded. Expects {len(feature_cols)} features.")

USEFUL_SENSORS = ['s2','s3','s4','s7','s8','s9','s11','s12','s13',
                  's14','s15','s17','s20','s21']

app = FastAPI(
    title="Predictive Maintenance RUL API",
    description="XGBoost-based Remaining Useful Life predictor for industrial IoT sensors.",
    version="1.0.0"
)

class SensorReading(BaseModel):
    engine_id:  str
    cycle:      int
    op1: float; op2: float; op3: float
    s2:  float; s3:  float; s4:  float
    s7:  float; s8:  float; s9:  float
    s11: float; s12: float; s13: float
    s14: float; s15: float; s17: float
    s20: float; s21: float
    # Rolling window features (client computes or use /predict_batch)
    rolling_means: List[float]   # 14 values — one per sensor
    rolling_stds:  List[float]   # 14 values
    rolling_slopes: List[float]  # 14 values

class RULResponse(BaseModel):
    record_id:    str
    timestamp:    str
    engine_id:    str
    cycle:        int
    rul_predicted: float
    status:       str
    alert_level:  str
    recommendation: str

class SimpleSensorReading(BaseModel):
    """Simplified input — just raw sensors, API computes everything"""
    engine_id: str
    cycle:     int
    op1: float = 0.0
    op2: float = 0.0
    op3: float = 100.0
    s2:  float = 642.0
    s3:  float = 1590.0
    s4:  float = 1400.0
    s7:  float = 554.0
    s8:  float = 2388.0
    s9:  float = 9065.0
    s11: float = 47.0
    s12: float = 521.0
    s13: float = 2388.0
    s14: float = 8138.0
    s15: float = 8.4
    s17: float = 392.0
    s20: float = 39.0
    s21: float = 23.0

# In-memory engine history for rolling features
engine_history = {}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "XGBoostRegressor",
        "features": len(feature_cols),
        "version": "1.0.0"
    }

@app.post("/predict", response_model=RULResponse)
def predict(req: SimpleSensorReading):
    eid = req.engine_id

    # Store reading in history
    reading = {s: getattr(req, s) for s in USEFUL_SENSORS}
    reading.update({'op1': req.op1, 'op2': req.op2,
                    'op3': req.op3, 'cycle': req.cycle})

    if eid not in engine_history:
        engine_history[eid] = []
    engine_history[eid].append(reading)

    # Keep last 30 readings for rolling features
    history = engine_history[eid][-30:]

    # Build feature vector
    base = [reading[s] for s in USEFUL_SENSORS]
    base += [req.op1, req.op2, req.op3, req.cycle]

    # Rolling features
    rolling = []
    for s in USEFUL_SENSORS:
        vals = np.array([h[s] for h in history])
        rolling.append(float(np.mean(vals)))
        rolling.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        if len(vals) > 1:
            slope = float(np.polyfit(range(len(vals)), vals, 1)[0])
        else:
            slope = 0.0
        rolling.append(slope)

    features = np.array(base + rolling, dtype=np.float32).reshape(1, -1)
    features_scaled = scaler.transform(features)

    rul = float(np.clip(model.predict(features_scaled)[0], 0, 125))

    # Status logic
    if rul <= 25:
        status       = "CRITICAL"
        alert_level  = "🔴"
        recommendation = "Immediate maintenance required — failure imminent"
    elif rul <= 50:
        status       = "WARNING"
        alert_level  = "🟠"
        recommendation = "Schedule maintenance within next 25 cycles"
    elif rul <= 100:
        status       = "CAUTION"
        alert_level  = "🟡"
        recommendation = "Monitor closely — plan maintenance soon"
    else:
        status       = "HEALTHY"
        alert_level  = "🟢"
        recommendation = "Operating normally"

    return RULResponse(
        record_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        engine_id=eid,
        cycle=req.cycle,
        rul_predicted=round(rul, 1),
        status=status,
        alert_level=alert_level,
        recommendation=recommendation
    )

@app.get("/engine/{engine_id}/history")
def engine_history_endpoint(engine_id: str):
    if engine_id not in engine_history:
        raise HTTPException(status_code=404, detail="Engine not found")
    return {
        "engine_id": engine_id,
        "readings": len(engine_history[engine_id]),
        "last_cycle": engine_history[engine_id][-1]["cycle"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
