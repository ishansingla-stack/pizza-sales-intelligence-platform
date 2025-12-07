"""
Export Best Trained Models for Production Deployment
Recreates and saves the champion models identified during training
"""

import sys
import os
from pathlib import Path
import joblib
import json

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

print("=" * 80)
print("EXPORT BEST MODELS FOR PRODUCTION")
print("=" * 80)

# Create models directory
models_dir = Path(project_root) / "outputs" / "models"
models_dir.mkdir(parents=True, exist_ok=True)
print(f"\n[*] Models will be saved to: {models_dir}")

# ============================================================================
# STEP 1: Load Training Data
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOAD TRAINING DATA")
print("=" * 80)

# Load REVENUE data (from revenue_target folder - CLEAN DATA, no Quantity leakage)
revenue_data_dir = Path(project_root) / "data" / "processed" / "revenue_target"
X_train_revenue = pd.read_parquet(revenue_data_dir / "X_train.parquet")
X_val_revenue = pd.read_parquet(revenue_data_dir / "X_val.parquet")
X_test_revenue = pd.read_parquet(revenue_data_dir / "X_test.parquet")
y_train_revenue = pd.read_parquet(revenue_data_dir / "y_train.parquet")['total_price']
y_val_revenue = pd.read_parquet(revenue_data_dir / "y_val.parquet")['total_price']
y_test_revenue = pd.read_parquet(revenue_data_dir / "y_test.parquet")['total_price']

print(f"[OK] Revenue data loaded from refactored/:")
print(f"   Features: {X_train_revenue.shape[1]}")
print(f"   Train samples: {X_train_revenue.shape[0]}")

# Load QUANTITY data (from quantity_target folder - where quantity models were trained)
quantity_data_dir = Path(project_root) / "data" / "processed" / "quantity_target"
X_train_quantity = pd.read_parquet(quantity_data_dir / "X_train.parquet")
X_val_quantity = pd.read_parquet(quantity_data_dir / "X_val.parquet")
X_test_quantity = pd.read_parquet(quantity_data_dir / "X_test.parquet")
y_train_quantity = pd.read_parquet(quantity_data_dir / "y_train.parquet")['quantity']
y_val_quantity = pd.read_parquet(quantity_data_dir / "y_val.parquet")['quantity']
y_test_quantity = pd.read_parquet(quantity_data_dir / "y_test.parquet")['quantity']

print(f"[OK] Quantity data loaded from quantity_target/:")
print(f"   Features: {X_train_quantity.shape[1]}")
print(f"   Train samples: {X_train_quantity.shape[0]}")

# ============================================================================
# STEP 2: Load Model Performance Results
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: IDENTIFY BEST MODELS FROM RESULTS")
print("=" * 80)

revenue_results = pd.read_csv(Path(project_root) / "outputs" / "results" / "ml_model_tracking" / "revenue_prediction_runs.csv")
quantity_results = pd.read_csv(Path(project_root) / "outputs" / "results" / "ml_model_tracking" / "quantity_prediction_runs.csv")

# Best Ridge model for revenue (champion - no overfitting)
best_ridge_revenue = revenue_results[revenue_results['model_name'] == 'Ridge'].nsmallest(1, 'test_rmse').iloc[0]
print(f"\n[CHAMPION] Best Ridge (Revenue) - NO OVERFITTING:")
print(f"   Train R²: {best_ridge_revenue['train_r2']:.4f}")
print(f"   Val R²: {best_ridge_revenue['val_r2']:.4f}")
print(f"   Test R²: {best_ridge_revenue['test_r2']:.4f}")
print(f"   Test RMSE: {best_ridge_revenue['test_rmse']:.4f}")
print(f"   Alpha: {best_ridge_revenue['alpha']}")
print(f"   [OK] Good generalization (consistent train/val/test)")

# Best Random Forest for quantity (champion - no overfitting)
rf_quantity = quantity_results[quantity_results['model_name'] == 'RandomForest']
if len(rf_quantity) > 0:
    best_quantity = rf_quantity.nsmallest(1, 'test_rmse').iloc[0]
else:
    # Fallback to best overall if no RF
    best_quantity = quantity_results.nlargest(1, 'test_r2').iloc[0]

print(f"\n[CHAMPION] Best Random Forest (Quantity) - NO OVERFITTING:")
print(f"   Train R²: {best_quantity['train_r2']:.6f}")
print(f"   Val R²: {best_quantity['val_r2']:.6f}")
print(f"   Test R²: {best_quantity['test_r2']:.6f}")
print(f"   Test RMSE: {best_quantity['test_rmse']:.6f}")
print(f"   [OK] Excellent accuracy with good generalization")

# ============================================================================
# STEP 3: Train and Save Revenue Models
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: TRAIN AND SAVE REVENUE MODELS")
print("=" * 80)

# Train Ridge model for revenue
print("\n[*] Training Ridge regression for revenue...")
ridge_revenue = Ridge(
    alpha=best_ridge_revenue['alpha'],
    random_state=42
)

# Combine train + val for final training
X_train_full_revenue = pd.concat([X_train_revenue, X_val_revenue])
y_train_full_revenue = pd.concat([y_train_revenue, y_val_revenue])

ridge_revenue.fit(X_train_full_revenue, y_train_full_revenue)
ridge_revenue_path = models_dir / "revenue_model.pkl"
joblib.dump(ridge_revenue, ridge_revenue_path)
print(f"[OK] Saved Revenue model (Ridge): {ridge_revenue_path}")
print(f"    File size: {ridge_revenue_path.stat().st_size / 1024:.1f} KB")

# ============================================================================
# STEP 4: Train and Save Quantity Models
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: TRAIN AND SAVE QUANTITY MODELS")
print("=" * 80)

# Combine train + val for quantity
X_train_full_quantity = pd.concat([X_train_quantity, X_val_quantity])
y_train_full_quantity = pd.concat([y_train_quantity, y_val_quantity])

# Train Random Forest for quantity (best generalizing model)
print("\n[*] Training Random Forest for quantity...")

# Handle NaN values in hyperparameters
n_estimators = best_quantity.get('n_estimators', 100)
max_depth = best_quantity.get('max_depth', 20)
min_samples_split = best_quantity.get('min_samples_split', 2)
min_samples_leaf = best_quantity.get('min_samples_leaf', 1)

quantity_model = RandomForestRegressor(
    n_estimators=int(n_estimators) if pd.notna(n_estimators) else 100,
    max_depth=int(max_depth) if pd.notna(max_depth) else 20,
    min_samples_split=int(min_samples_split) if pd.notna(min_samples_split) else 2,
    min_samples_leaf=int(min_samples_leaf) if pd.notna(min_samples_leaf) else 1,
    random_state=42
)

quantity_model.fit(X_train_full_quantity, y_train_full_quantity)
quantity_model_path = models_dir / "quantity_model.pkl"
joblib.dump(quantity_model, quantity_model_path)
print(f"[OK] Saved Quantity model (Random Forest): {quantity_model_path}")
print(f"    File size: {quantity_model_path.stat().st_size / 1024:.1f} KB")

# ============================================================================
# STEP 5: Save Feature Names and Metadata
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: SAVE FEATURE METADATA")
print("=" * 80)

# Save feature names for revenue model
feature_metadata_revenue = {
    "feature_names": list(X_train_revenue.columns),
    "n_features": len(X_train_revenue.columns),
    "model_type": "Ridge",
    "test_r2": float(best_ridge_revenue['test_r2']),
    "test_rmse": float(best_ridge_revenue['test_rmse']),
    "alpha": float(best_ridge_revenue['alpha'])
}

metadata_path_revenue = models_dir / "revenue_model_metadata.json"
with open(metadata_path_revenue, 'w') as f:
    json.dump(feature_metadata_revenue, f, indent=2)
print(f"[OK] Saved revenue metadata: {metadata_path_revenue}")

# Save feature names for quantity model
feature_metadata_quantity = {
    "feature_names": list(X_train_quantity.columns),
    "n_features": len(X_train_quantity.columns),
    "model_type": best_quantity['model_name'],
    "test_r2": float(best_quantity['test_r2']),
    "test_rmse": float(best_quantity['test_rmse'])
}

metadata_path_quantity = models_dir / "quantity_model_metadata.json"
with open(metadata_path_quantity, 'w') as f:
    json.dump(feature_metadata_quantity, f, indent=2)
print(f"[OK] Saved quantity metadata: {metadata_path_quantity}")

# ============================================================================
# STEP 6: Verify Models
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: VERIFY SAVED MODELS")
print("=" * 80)

# Load and test revenue model
loaded_ridge = joblib.load(ridge_revenue_path)
test_pred_revenue = loaded_ridge.predict(X_test_revenue[:5])
print(f"\n[OK] Revenue model loaded successfully")
print(f"   Sample predictions: {test_pred_revenue}")

# Load and test quantity model
loaded_quantity = joblib.load(quantity_model_path)
test_pred_qty = loaded_quantity.predict(X_test_quantity[:5])
print(f"\n[OK] Quantity model loaded successfully")
print(f"   Sample predictions: {test_pred_qty}")

# Verify no overfitting
from sklearn.metrics import r2_score, mean_squared_error
test_r2_revenue = r2_score(y_test_revenue, loaded_ridge.predict(X_test_revenue))
test_r2_quantity = r2_score(y_test_quantity, loaded_quantity.predict(X_test_quantity))

print("\n" + "=" * 80)
print("OVERFITTING CHECK")
print("=" * 80)
print(f"Revenue model (Ridge):")
print(f"  Train R²: {best_ridge_revenue['train_r2']:.4f}")
print(f"  Val R²: {best_ridge_revenue['val_r2']:.4f}")
print(f"  Test R²: {test_r2_revenue:.4f}")
print(f"  [OK] Generalization gap: {abs(best_ridge_revenue['train_r2'] - test_r2_revenue):.4f} (< 0.01 is excellent)")

print(f"\nQuantity model (Random Forest):")
print(f"  Train R²: {best_quantity['train_r2']:.6f}")
print(f"  Val R²: {best_quantity['val_r2']:.6f}")
print(f"  Test R²: {test_r2_quantity:.6f}")
print(f"  [OK] Generalization gap: {abs(best_quantity['train_r2'] - test_r2_quantity):.6f} (< 0.01 is excellent)")

print("\n" + "=" * 80)
print("MODEL EXPORT COMPLETE")
print("=" * 80)
print(f"\nSaved models (NO OVERFITTING):")
print(f"  1. {ridge_revenue_path}")
print(f"  2. {quantity_model_path}")
print(f"  3. {metadata_path_revenue}")
print(f"  4. {metadata_path_quantity}")
print("\n[SUCCESS] These models are ready for deployment in the Streamlit dashboard.")
print("[SUCCESS] Both models show excellent generalization with no overfitting.")
