"""
Pizza Intelligence - Clean Data Preparation for Quantity Prediction
NO DATA LEAKAGE - Excludes 'Quantity' and 'total_price' from features
Target: quantity (sales volume)
Output: data/processed/quantity_target/
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import mlflow
from src.config_loader import ConfigLoader

print("=" * 80)
print("CLEAN DATA PREPARATION - QUANTITY PREDICTION (NO LEAKAGE)")
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
    print(f" Data file not found: {data_path}")
    sys.exit(1)

df = pd.read_excel(data_path, sheet_name='pizza_sales')
print(f" Loaded {len(df):,} records")
print(f" Date range: {df['order_date'].min()} to {df['order_date'].max()}")

#%% Feature Engineering
print("\n" + "=" * 80)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 80)

df_ml = df.copy()

# =============================================================================
# TIME-BASED FEATURES
# =============================================================================
print("\n[TIME] Extracting time-based features...")

# Convert to datetime
df_ml['order_date'] = pd.to_datetime(df_ml['order_date'])
df_ml['order_time'] = pd.to_datetime(df_ml['order_time'], format='%H:%M:%S', errors='coerce')

# Extract time components
df_ml['Hour'] = df_ml['order_time'].dt.hour
df_ml['Day_of_Week'] = df_ml['order_date'].dt.dayofweek  # 0=Monday, 6=Sunday
df_ml['Month'] = df_ml['order_date'].dt.month
df_ml['Is_Weekend'] = df_ml['Day_of_Week'].isin([5, 6]).astype(int)

print(f"    Hour: {df_ml['Hour'].min()}-{df_ml['Hour'].max()}")
print(f"    Day_of_Week: {df_ml['Day_of_Week'].min()}-{df_ml['Day_of_Week'].max()}")
print(f"    Month: {df_ml['Month'].min()}-{df_ml['Month'].max()}")
print(f"    Is_Weekend: {df_ml['Is_Weekend'].sum():,} weekend orders")

# =============================================================================
# CYCLIC ENCODING FOR TIME FEATURES
# =============================================================================
print("\n Applying cyclic encoding (so 23:00 is close to 00:00)...")

# Hour (0-23 cycle)
df_ml['Hour_Sin'] = np.sin(2 * np.pi * df_ml['Hour'] / 24)
df_ml['Hour_Cos'] = np.cos(2 * np.pi * df_ml['Hour'] / 24)

# Day of Week (0-6 cycle)
df_ml['Day_Sin'] = np.sin(2 * np.pi * df_ml['Day_of_Week'] / 7)
df_ml['Day_Cos'] = np.cos(2 * np.pi * df_ml['Day_of_Week'] / 7)

# Month (1-12 cycle)
df_ml['Month_Sin'] = np.sin(2 * np.pi * df_ml['Month'] / 12)
df_ml['Month_Cos'] = np.cos(2 * np.pi * df_ml['Month'] / 12)

print("    Cyclic encoding applied")
print("      - Hour -> Hour_Sin, Hour_Cos")
print("      - Day_of_Week -> Day_Sin, Day_Cos")
print("      - Month -> Month_Sin, Month_Cos")

# =============================================================================
# ONE-HOT ENCODING FOR CATEGORICAL VARIABLES
# =============================================================================
print("\n  Applying One-Hot Encoding...")

# pizza_category
if 'pizza_category' in df_ml.columns:
    category_dummies = pd.get_dummies(df_ml['pizza_category'], prefix='Category')
    df_ml = pd.concat([df_ml, category_dummies], axis=1)
    print(f"    pizza_category -> {len(category_dummies.columns)} columns: {list(category_dummies.columns)}")

# pizza_size
if 'pizza_size' in df_ml.columns:
    size_dummies = pd.get_dummies(df_ml['pizza_size'], prefix='Size')
    df_ml = pd.concat([df_ml, size_dummies], axis=1)
    print(f"    pizza_size -> {len(size_dummies.columns)} columns: {list(size_dummies.columns)}")

# pizza_type (if column exists)
if 'pizza_type' in df_ml.columns:
    type_dummies = pd.get_dummies(df_ml['pizza_type'], prefix='Type')
    df_ml = pd.concat([df_ml, type_dummies], axis=1)
    print(f"    pizza_type -> {len(type_dummies.columns)} columns")

# pizza_name (one-hot encode individual pizzas)
name_dummies = pd.get_dummies(df_ml['pizza_name'], prefix='Pizza')
df_ml = pd.concat([df_ml, name_dummies], axis=1)
print(f"    pizza_name -> {len(name_dummies.columns)} columns (32 unique pizzas)")

# =============================================================================
# ADDITIONAL FEATURES
# =============================================================================
print("\n Creating additional features...")

# Ingredient count
df_ml['Ingredient_Count'] = df_ml['pizza_ingredients'].str.split(',').str.len()

# Price features - ONLY Unit_Price, NO Quantity or total_price to avoid leakage
df_ml['Unit_Price'] = df_ml['unit_price']

print(f"    Ingredient_Count: range {df_ml['Ingredient_Count'].min()}-{df_ml['Ingredient_Count'].max()}")
print(f"    Unit_Price: range ${df_ml['Unit_Price'].min():.2f}-${df_ml['Unit_Price'].max():.2f}")

#%% Define Target and Features
print("\n" + "=" * 80)
print("STEP 3: DEFINE TARGET AND FEATURES (NO LEAKAGE)")
print("=" * 80)

# =============================================================================
# TARGET VARIABLE: quantity (SALES VOLUME - REGRESSION)
# =============================================================================
target_col = 'quantity'
df_ml['Target'] = df_ml[target_col]

print(f"\n Target Variable: {target_col} (Sales Volume)")
print(f"   Mean: {df_ml['Target'].mean():.2f}")
print(f"   Median: {df_ml['Target'].median():.2f}")
print(f"   Std: {df_ml['Target'].std():.2f}")
print(f"   Range: {df_ml['Target'].min():.0f} - {df_ml['Target'].max():.0f}")

# =============================================================================
# FEATURE SELECTION - EXCLUDE LEAKING FEATURES
# =============================================================================
print("\n Selecting features (EXCLUDING Quantity and total_price)...")

# Get all one-hot encoded columns
category_cols = [col for col in df_ml.columns if col.startswith('Category_')]
size_cols = [col for col in df_ml.columns if col.startswith('Size_')]
type_cols = [col for col in df_ml.columns if col.startswith('Type_')]
pizza_cols = [col for col in df_ml.columns if col.startswith('Pizza_')]

# Define feature list - NO QUANTITY OR TOTAL_PRICE
feature_columns = (
    # Time features (original)
    ['Hour', 'Day_of_Week', 'Month', 'Is_Weekend'] +

    # Time features (cyclic)
    ['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos'] +

    # Product features - ONLY Unit_Price and Ingredient_Count
    ['Unit_Price', 'Ingredient_Count'] +

    # One-hot encoded categoricals
    category_cols + size_cols + type_cols + pizza_cols
)

# Remove duplicates
feature_columns = list(dict.fromkeys(feature_columns))

# Verify all features exist
feature_columns = [col for col in feature_columns if col in df_ml.columns]

print(f"\n Selected {len(feature_columns)} features:")
print(f"   - Time features: {sum(1 for c in feature_columns if 'Hour' in c or 'Day' in c or 'Month' in c or 'Weekend' in c)}")
print(f"   - Product features: 2 (Unit_Price, Ingredient_Count) - NO LEAKAGE")
print(f"   - Category encoding: {len(category_cols)} columns")
print(f"   - Size encoding: {len(size_cols)} columns")
print(f"   - Pizza name encoding: {len(pizza_cols)} columns")

# Create X and y
X = df_ml[feature_columns].copy()
y = df_ml['Target'].copy()

# Handle missing values
print(f"\n Checking for missing values...")
missing_counts = X.isnull().sum()
if missing_counts.sum() > 0:
    print("     Found missing values:")
    print(missing_counts[missing_counts > 0])
    print("   Filling with 0...")
    X = X.fillna(0)
else:
    print("    No missing values")

print(f"\n Final dataset shape: X={X.shape}, y={y.shape}")

#%% Data Splitting: 80-10-10
print("\n" + "=" * 80)
print("STEP 4: TRAIN-VALIDATION-TEST SPLIT (80-10-10)")
print("=" * 80)

# First split: 80% train, 20% temp (which will become 10% val, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

# Second split: Split temp into 50% validation, 50% test (each 10% of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
)

print(f"\n Data split complete:")
print(f"    Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"    Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"    Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"    Total:          {len(X):,} samples")

# Verify split ratios
print(f"\n Actual split ratios:")
print(f"   Train: {len(X_train)/len(X):.3f}")
print(f"   Val:   {len(X_val)/len(X):.3f}")
print(f"   Test:  {len(X_test)/len(X):.3f}")
print(f"   Sum:   {(len(X_train)+len(X_val)+len(X_test))/len(X):.3f}")

# Target distribution
print(f"\n Target distribution:")
print(f"   Train - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
print(f"   Val   - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}")
print(f"   Test  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")

#%% Save Processed Data
print("\n" + "=" * 80)
print("STEP 5: SAVE PROCESSED DATA")
print("=" * 80)

output_dir = Path(project_root) / "data" / "processed" / "quantity_target"
output_dir.mkdir(parents=True, exist_ok=True)

# Save splits
X_train.to_parquet(output_dir / "X_train.parquet")
X_val.to_parquet(output_dir / "X_val.parquet")
X_test.to_parquet(output_dir / "X_test.parquet")
y_train.to_frame('quantity').to_parquet(output_dir / "y_train.parquet")
y_val.to_frame('quantity').to_parquet(output_dir / "y_val.parquet")
y_test.to_frame('quantity').to_parquet(output_dir / "y_test.parquet")

# Save feature names
feature_info = {
    'feature_columns': feature_columns,
    'n_features': len(feature_columns),
    'target': target_col,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'note': 'NO DATA LEAKAGE - Quantity and total_price excluded from features'
}

import json
with open(output_dir / "feature_info.json", 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f"\n Saved processed data to: {output_dir}")
print(f"   - X_train.parquet ({len(X_train):,} samples)")
print(f"   - X_val.parquet ({len(X_val):,} samples)")
print(f"   - X_test.parquet ({len(X_test):,} samples)")
print(f"   - y_train.parquet")
print(f"   - y_val.parquet")
print(f"   - y_test.parquet")
print(f"   - feature_info.json ({len(feature_columns)} features)")

#%% Summary Report
print("\n" + "=" * 80)
print(" DATA PREPARATION COMPLETE - NO LEAKAGE")
print("=" * 80)

print(f"""
Summary:
  - Total Records: {len(df):,}
  - Target Variable: {target_col} (Regression)
  - Features: {len(feature_columns)} (NO LEAKAGE)
  - Train/Val/Test Split: {len(X_train)}/{len(X_val)}/{len(X_test)} (80/10/10)

Feature Categories:
  - Time Features: Hour, Day_of_Week, Month, Is_Weekend (+ cyclic encoding)
  - Product Features: Unit_Price, Ingredient_Count (EXCLUDES Quantity, total_price)
  - Category One-Hot: {len(category_cols)} columns
  - Size One-Hot: {len(size_cols)} columns
  - Pizza Name One-Hot: {len(pizza_cols)} columns

Data Leakage Prevention:
  - EXCLUDED: Quantity (the target itself)
  - EXCLUDED: total_price (can derive quantity from it)
  - INCLUDED: Unit_Price (known pizza attribute)

Next Steps:
  1. Re-run: python scripts/03_quantity_volume_training.py
  2. Models will learn from time patterns, pizza attributes, NOT leaked targets
""")

print("=" * 80)
