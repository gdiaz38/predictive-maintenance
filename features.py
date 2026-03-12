import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "/Users/gabrieldiaz/.cache/kagglehub/datasets/behrad3d/nasa-cmaps/versions/1/CMaps"

COLS = (
    ['engine_id', 'cycle', 'op1', 'op2', 'op3'] +
    [f's{i}' for i in range(1, 22)]
)

# Only sensors with meaningful variance
USEFUL_SENSORS = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']

def load_data(subset='FD001'):
    train = pd.read_csv(f"{DATA_PATH}/train_{subset}.txt",
                        sep=r'\s+', header=None, names=COLS)
    test  = pd.read_csv(f"{DATA_PATH}/test_{subset}.txt",
                        sep=r'\s+', header=None, names=COLS)
    rul   = pd.read_csv(f"{DATA_PATH}/RUL_{subset}.txt",
                        header=None, names=['RUL'])
    return train, test, rul

def add_rul(df):
    """Add Remaining Useful Life column — counts down to 0 at failure"""
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

def add_rolling_features(df, window=30):
    """
    Rolling window features — captures degradation trend, not just current value.
    This is what makes the model robust to sensor noise.
    """
    df = df.sort_values(['engine_id', 'cycle'])
    for sensor in USEFUL_SENSORS:
        grp = df.groupby('engine_id')[sensor]
        df[f'{sensor}_mean{window}'] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{sensor}_std{window}']  = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f'{sensor}_slope{window}'] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0,
                raw=True))
    return df

def clip_rul(df, max_rul=125):
    """
    Cap RUL at 125 cycles — engines are 'healthy' until ~125 cycles before failure.
    This is standard practice in predictive maintenance literature.
    It forces the model to focus on the degradation window, not healthy operation.
    """
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    return df

print("Loading and processing FD001...")
train, test, rul_test = load_data('FD001')

# Add RUL to training set
train = add_rul(train)
train = clip_rul(train)

# Add rolling features
print("Engineering rolling window features...")
train = add_rolling_features(train, window=30)

# For test set: RUL at the last cycle of each engine is given in RUL_FD001.txt
# We reconstruct full RUL for test set
test = add_rul(test)
# Adjust test RUL: last cycle RUL = given value, count up from there
test = test.sort_values(['engine_id','cycle'])
for eng_id in test['engine_id'].unique():
    mask     = test['engine_id'] == eng_id
    true_rul = rul_test.loc[eng_id - 1, 'RUL']
    max_c    = test.loc[mask, 'cycle'].max()
    test.loc[mask, 'RUL'] = true_rul + (max_c - test.loc[mask, 'cycle'])

test = clip_rul(test)
test = add_rolling_features(test, window=30)

# Feature columns for model
base_features   = USEFUL_SENSORS + ['op1','op2','op3','cycle']
rolled_features = [c for c in train.columns if any(
    s in c for s in ['_mean30','_std30','_slope30'])]
FEATURE_COLS = base_features + rolled_features

print(f"Total features: {len(FEATURE_COLS)}")

# Drop rows with NaN (from rolling window at start of each engine)
train = train.dropna(subset=FEATURE_COLS)
test  = test.dropna(subset=FEATURE_COLS)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train[FEATURE_COLS])
y_train = train['RUL'].values

X_test  = scaler.transform(test[FEATURE_COLS])
y_test  = test['RUL'].values

# Save
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy",  X_test)
np.save("y_test.npy",  y_test)
joblib.dump(scaler, "scaler_rul.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Plot RUL distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=50, color='#00d4aa', edgecolor='black')
axes[0].set_title('Training RUL Distribution (clipped at 125)')
axes[0].set_xlabel('RUL (cycles)')
axes[0].set_ylabel('Count')

# Plot one engine's RUL countdown
eng1 = train[train['engine_id'] == 1].sort_values('cycle')
axes[1].plot(eng1['cycle'], eng1['RUL'], color='#00d4aa', linewidth=2)
axes[1].set_title('Engine 1 — RUL Countdown')
axes[1].set_xlabel('Cycle')
axes[1].set_ylabel('Remaining Useful Life')
axes[1].axhline(0, color='red', linestyle='--', label='Failure')
axes[1].legend()

plt.tight_layout()
plt.savefig("rul_distribution.png", dpi=150)
print("✅ Saved rul_distribution.png")
print("✅ Features saved")
