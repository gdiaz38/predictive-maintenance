import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

print("Loading data...")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── XGBoost RUL Predictor ─────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric='rmse'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

joblib.dump(model, "xgb_rul.pkl")
print("✅ Saved xgb_rul.pkl")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 125)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

print(f"\n{'='*45}")
print(f"FINAL RESULTS")
print(f"{'='*45}")
print(f"MAE  : {mae:.2f} cycles")
print(f"RMSE : {rmse:.2f} cycles")
print(f"MAPE : {mape:.2f}%")

# Score by RUL bucket — most important for maintenance scheduling
print("\n=== ACCURACY BY RUL BUCKET ===")
buckets = [(0,25,'Critical (0-25)'),(25,50,'Warning (25-50)'),
           (50,100,'Caution (50-100)'),(100,126,'Healthy (100+)')]
for lo, hi, label in buckets:
    mask = (y_test >= lo) & (y_test < hi)
    if mask.sum() > 0:
        bucket_mae = mean_absolute_error(y_test[mask], y_pred[mask])
        print(f"  {label:<22} n={mask.sum():>5} | MAE={bucket_mae:.2f} cycles")

# ── Feature importance ────────────────────────────────────────────────────────
feature_cols = joblib.load("feature_cols.pkl")
importance   = model.feature_importances_
top_idx      = np.argsort(importance)[::-1][:15]

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Predicted vs actual
axes[0].scatter(y_test, y_pred, alpha=0.3, color='#00d4aa', s=5)
axes[0].plot([0,125],[0,125], 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('Actual RUL')
axes[0].set_ylabel('Predicted RUL')
axes[0].set_title(f'Predicted vs Actual RUL\nMAE={mae:.1f} cycles')
axes[0].legend()

# 2. Feature importance
axes[1].barh(
    [feature_cols[i] for i in top_idx],
    importance[top_idx],
    color='#00d4aa'
)
axes[1].set_title('Top 15 Feature Importances')
axes[1].set_xlabel('Importance Score')
axes[1].invert_yaxis()

# 3. Error distribution
errors = y_pred - y_test
axes[2].hist(errors, bins=60, color='#00d4aa', edgecolor='black', alpha=0.8)
axes[2].axvline(0, color='red', linestyle='--', label='Zero error')
axes[2].set_title(f'Prediction Error Distribution\nRMSE={rmse:.1f} cycles')
axes[2].set_xlabel('Predicted RUL - Actual RUL')
axes[2].set_ylabel('Count')
axes[2].legend()

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("✅ Saved training_results.png")
