import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "/Users/gabrieldiaz/.cache/kagglehub/datasets/behrad3d/nasa-cmaps/versions/1/CMaps"

# Column names from the dataset readme
COLS = (
    ['engine_id', 'cycle', 'op1', 'op2', 'op3'] +
    [f's{i}' for i in range(1, 22)]
)

print("Loading FD001 training data...")
train = pd.read_csv(f"{DATA_PATH}/train_FD001.txt", sep='\s+', header=None, names=COLS)
test  = pd.read_csv(f"{DATA_PATH}/test_FD001.txt",  sep='\s+', header=None, names=COLS)
rul   = pd.read_csv(f"{DATA_PATH}/RUL_FD001.txt",   header=None, names=['RUL'])

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"RUL entries: {len(rul)}")

print(f"\nNumber of unique engines (train): {train['engine_id'].nunique()}")
print(f"Max cycles per engine: {train.groupby('engine_id')['cycle'].max().max()}")
print(f"Min cycles per engine: {train.groupby('engine_id')['cycle'].max().min()}")

print("\n=== FIRST 5 ROWS ===")
print(train[['engine_id','cycle','op1','op2','op3','s1','s2','s3','s4']].head())

# Plot degradation for 5 engines — sensor 2 (fan speed, most informative)
print("\nPlotting sensor degradation curves...")
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()

for i, eng_id in enumerate(range(1, 6)):
    eng_data = train[train['engine_id'] == eng_id]
    axes[i].plot(eng_data['cycle'], eng_data['s2'],
                 color='#00d4aa', linewidth=1, label='Sensor 2 (Fan Speed)')
    axes[i].plot(eng_data['cycle'], eng_data['s11'],
                 color='#ffa500', linewidth=1, label='Sensor 11 (Static Pressure)')
    axes[i].set_title(f'Engine {eng_id} — {eng_data["cycle"].max()} cycles to failure')
    axes[i].set_xlabel('Cycle')
    axes[i].set_ylabel('Sensor Value')
    axes[i].legend(fontsize=7)

# Cycle length distribution
axes[5].hist(train.groupby('engine_id')['cycle'].max(),
             bins=20, color='#00d4aa', edgecolor='black')
axes[5].set_title('Distribution of Engine Lifetimes')
axes[5].set_xlabel('Total Cycles Before Failure')
axes[5].set_ylabel('Count')

plt.suptitle('NASA CMAPSS — Turbofan Engine Sensor Degradation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("sensor_degradation.png", dpi=150)
print("✅ Saved sensor_degradation.png")

# Which sensors actually change over time? (constant sensors are useless)
print("\n=== SENSOR VARIANCE (low variance = useless for prediction) ===")
sensor_cols = [f's{i}' for i in range(1, 22)]
variances = train[sensor_cols].var().sort_values(ascending=False)
print(variances.round(4))
print(f"\nSensors with near-zero variance (drop these): {list(variances[variances < 0.01].index)}")
