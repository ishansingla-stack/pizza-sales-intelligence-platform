"""
Pizza Intelligence - Demand Forecasting Data Preparation
Predicts TOTAL pizza demand by hour/day, not individual order quantities
Target: Total pizzas sold per hour
Features: Temporal patterns (date, time, day of week, seasonality)
Output: data/processed/demand_forecasting/
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import mlflow
from src.config_loader import ConfigLoader

print("=" * 80)
print("DEMAND FORECASTING - DATA PREPARATION")
print("=" * 80)

# Load configuration
print("\n[*] Loading configuration...")
config = ConfigLoader()
config.setup_databricks_env()
config.create_output_dirs()

print(f"[OK] Databricks environment configured")
print(f"   Host: {os.getenv('DATABRICKS_HOST')}")
print(f"   MLflow: {mlflow.get_tracking_uri()}")

#%% Load Raw Data
print("\n" + "=" * 80)
print("STEP 1: LOAD RAW DATA")
print("=" * 80)

data_path = config.get_local_data_path()

if not os.path.exists(data_path):
    print(f"[ERROR] Data file not found: {data_path}")
    sys.exit(1)

df = pd.read_excel(data_path, sheet_name='pizza_sales')
print(f"[OK] Loaded {len(df):,} order line items")
print(f"    Date range: {df['order_date'].min()} to {df['order_date'].max()}")

#%% Aggregate by Time Period
print("\n" + "=" * 80)
print("STEP 2: AGGREGATE DEMAND BY HOUR")
print("=" * 80)

# Convert to datetime
df['order_date'] = pd.to_datetime(df['order_date'])
df['order_time'] = pd.to_datetime(df['order_time'], format='%H:%M:%S', errors='coerce')

# Create datetime column
df['order_datetime'] = pd.to_datetime(
    df['order_date'].astype(str) + ' ' +
    df['order_time'].dt.strftime('%H:%M:%S')
)

# Extract hour
df['hour'] = df['order_time'].dt.hour

# Aggregate by date + hour
print("\n[*] Aggregating orders by date and hour...")
demand_df = df.groupby([df['order_date'].dt.date, 'hour']).agg({
    'quantity': 'sum',  # Total pizzas sold in that hour
    'order_id': 'nunique',  # Number of unique orders
    'total_price': 'sum'  # Total revenue
}).reset_index()

demand_df.columns = ['date', 'hour', 'total_pizzas', 'num_orders', 'total_revenue']

# Convert date back to datetime
demand_df['date'] = pd.to_datetime(demand_df['date'])

print(f"[OK] Aggregated {len(df):,} line items into {len(demand_df):,} hourly demand records")
print(f"    Average pizzas per hour: {demand_df['total_pizzas'].mean():.1f}")
print(f"    Max pizzas in an hour: {demand_df['total_pizzas'].max()}")
print(f"    Min pizzas in an hour: {demand_df['total_pizzas'].min()}")

#%% Feature Engineering for Time Series
print("\n" + "=" * 80)
print("STEP 3: TIME SERIES FEATURE ENGINEERING")
print("=" * 80)

# Extract temporal features
demand_df['day_of_week'] = demand_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
demand_df['month'] = demand_df['date'].dt.month
demand_df['day_of_month'] = demand_df['date'].dt.day
demand_df['week_of_year'] = demand_df['date'].dt.isocalendar().week
demand_df['is_weekend'] = demand_df['day_of_week'].isin([5, 6]).astype(int)

# Cyclic encoding for temporal features
print("\n[*] Applying cyclic encoding...")

# Hour (0-23 cycle)
demand_df['hour_sin'] = np.sin(2 * np.pi * demand_df['hour'] / 24)
demand_df['hour_cos'] = np.cos(2 * np.pi * demand_df['hour'] / 24)

# Day of Week (0-6 cycle)
demand_df['day_sin'] = np.sin(2 * np.pi * demand_df['day_of_week'] / 7)
demand_df['day_cos'] = np.cos(2 * np.pi * demand_df['day_of_week'] / 7)

# Month (1-12 cycle)
demand_df['month_sin'] = np.sin(2 * np.pi * demand_df['month'] / 12)
demand_df['month_cos'] = np.cos(2 * np.pi * demand_df['month'] / 12)

# Day of month (1-31 cycle)
demand_df['day_of_month_sin'] = np.sin(2 * np.pi * demand_df['day_of_month'] / 31)
demand_df['day_of_month_cos'] = np.cos(2 * np.pi * demand_df['day_of_month'] / 31)

print("[OK] Cyclic encoding applied:")
print("    - Hour (0-23)")
print("    - Day of Week (0-6)")
print("    - Month (1-12)")
print("    - Day of Month (1-31)")

# Lag features (previous hour, previous day same hour)
print("\n[*] Creating lag features...")
demand_df = demand_df.sort_values(['date', 'hour'])
demand_df['prev_hour_pizzas'] = demand_df['total_pizzas'].shift(1)
demand_df['same_hour_yesterday'] = demand_df.groupby('hour')['total_pizzas'].shift(1)

# Rolling averages
demand_df['rolling_3h_avg'] = demand_df['total_pizzas'].rolling(window=3, min_periods=1).mean()
demand_df['rolling_24h_avg'] = demand_df['total_pizzas'].rolling(window=24, min_periods=1).mean()

print("[OK] Lag features created:")
print("    - Previous hour demand")
print("    - Same hour yesterday")
print("    - 3-hour rolling average")
print("    - 24-hour rolling average")

#%% Define Features and Target
print("\n" + "=" * 80)
print("STEP 4: DEFINE FEATURES AND TARGET")
print("=" * 80)

# Target variable
target_col = 'total_pizzas'

# Feature columns (temporal features only, NO product-specific features)
feature_columns = [
    # Original temporal features
    'hour', 'day_of_week', 'month', 'day_of_month', 'week_of_year', 'is_weekend',

    # Cyclic encoded features
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos',

    # Lag features
    'prev_hour_pizzas', 'same_hour_yesterday',
    'rolling_3h_avg', 'rolling_24h_avg'
]

# Drop rows with NaN (from lag features)
print(f"\n[*] Original records: {len(demand_df)}")
demand_df_clean = demand_df.dropna(subset=feature_columns)
print(f"[*] After dropping NaN: {len(demand_df_clean)}")

X = demand_df_clean[feature_columns].copy()
y = demand_df_clean[target_col].copy()

print(f"\n[OK] Dataset prepared:")
print(f"    Features: {len(feature_columns)}")
print(f"    Samples: {len(X):,}")
print(f"    Target range: {y.min():.0f} - {y.max():.0f} pizzas")
print(f"    Target mean: {y.mean():.1f} pizzas per hour")
print(f"    Target std: {y.std():.1f}")

#%% Train-Val-Test Split (Chronological)
print("\n" + "=" * 80)
print("STEP 5: CHRONOLOGICAL TRAIN-VAL-TEST SPLIT")
print("=" * 80)

# For time series, use chronological split (NOT random shuffle)
n_total = len(X)
n_train = int(0.70 * n_total)  # 70% train
n_val = int(0.15 * n_total)    # 15% validation
# Remaining is test

X_train = X.iloc[:n_train]
y_train = y.iloc[:n_train]

X_val = X.iloc[n_train:n_train+n_val]
y_val = y.iloc[n_train:n_train+n_val]

X_test = X.iloc[n_train+n_val:]
y_test = y.iloc[n_train+n_val:]

print(f"\n[OK] Chronological split (no shuffle):")
print(f"    Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"    Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"    Test:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

print(f"\n    Target distribution:")
print(f"      Train - Mean: {y_train.mean():.1f}, Std: {y_train.std():.1f}")
print(f"      Val   - Mean: {y_val.mean():.1f}, Std: {y_val.std():.1f}")
print(f"      Test  - Mean: {y_test.mean():.1f}, Std: {y_test.std():.1f}")

#%% Save Processed Data
print("\n" + "=" * 80)
print("STEP 6: SAVE PROCESSED DATA")
print("=" * 80)

output_dir = Path(project_root) / "data" / "processed" / "demand_forecasting"
output_dir.mkdir(parents=True, exist_ok=True)

# Save splits
X_train.to_parquet(output_dir / "X_train.parquet")
X_val.to_parquet(output_dir / "X_val.parquet")
X_test.to_parquet(output_dir / "X_test.parquet")
y_train.to_frame('total_pizzas').to_parquet(output_dir / "y_train.parquet")
y_val.to_frame('total_pizzas').to_parquet(output_dir / "y_val.parquet")
y_test.to_frame('total_pizzas').to_parquet(output_dir / "y_test.parquet")

# Save feature metadata
import json
feature_info = {
    'feature_columns': feature_columns,
    'n_features': len(feature_columns),
    'target': target_col,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'description': 'Demand forecasting - predicts total hourly pizza demand',
    'note': 'Chronological split for time series'
}

with open(output_dir / "feature_info.json", 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f"\n[OK] Saved processed data to: {output_dir}")
print(f"    - X_train.parquet ({len(X_train):,} samples)")
print(f"    - X_val.parquet ({len(X_val):,} samples)")
print(f"    - X_test.parquet ({len(X_test):,} samples)")
print(f"    - y_*.parquet files")
print(f"    - feature_info.json")

#%% Summary
print("\n" + "=" * 80)
print("DEMAND FORECASTING DATA PREPARATION COMPLETE")
print("=" * 80)

print(f"""
Summary:
  - Aggregation Level: Hourly demand
  - Total Time Periods: {len(demand_df_clean):,}
  - Target: Total pizzas per hour (range: {y.min():.0f}-{y.max():.0f}, mean: {y.mean():.1f})
  - Features: {len(feature_columns)} temporal features
  - Split: 70/15/15 (chronological, no shuffle)

Feature Categories:
  - Temporal: hour, day_of_week, month, day_of_month, week_of_year, is_weekend
  - Cyclic: hour_sin/cos, day_sin/cos, month_sin/cos, day_of_month_sin/cos
  - Lag: prev_hour, same_hour_yesterday
  - Rolling: 3h average, 24h average

Use Case:
  "Predict how many pizzas will be sold on Friday at 7 PM"

Next Steps:
  1. Train time series models on this data
  2. Evaluate forecasting accuracy
  3. Use for inventory planning and staffing
""")

print("=" * 80)
