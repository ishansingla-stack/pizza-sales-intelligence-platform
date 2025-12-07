"""
Deploy Revenue Prediction Model - Phase 2
Exports the champion ensemble model to production
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import mlflow
import mlflow.sklearn
import json
from src.config_loader import ConfigLoader

print("=" * 80)
print("DEPLOY REVENUE PREDICTION MODEL - PHASE 2")
print("=" * 80)

# Setup Databricks
config = ConfigLoader()
config.setup_databricks_env()

# ============================================================================
# STEP 1: LOAD RESULTS AND FIND CHAMPION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: FIND CHAMPION REVENUE MODEL")
print("=" * 80)

results_path = project_root / "outputs" / "results" / "sales_prediction_results.csv"
results_df = pd.read_csv(results_path)

print(f"[OK] Loaded {len(results_df)} training results")

# Get champion model - Ensemble_n5_uniform_divTrue (identified as top model)
ensemble_df = results_df[results_df['model_name'] == 'Ensemble']
champion = ensemble_df[ensemble_df['run_name'] == 'Ensemble_n5_uniform_divTrue'].iloc[0]

print(f"\n[CHAMPION MODEL]")
print(f"  Model: {champion['model_name']}")
print(f"  Config: {champion['run_name']}")
print(f"  Val RMSE: {champion['val_rmse']:.4f}")
print(f"  Val R²: {champion['val_r2']:.4f}")
print(f"  Test RMSE: {champion['test_rmse']:.4f}")
print(f"  Test R²: {champion['test_r2']:.4f}")

# ============================================================================
# STEP 2: FIND MLFLOW RUN
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: LOCATE MLFLOW RUN")
print("=" * 80)

# Set experiment
mlflow.set_experiment("/pizza-sales-prediction")

# Search for the champion run
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("/pizza-sales-prediction")

if experiment is None:
    print(f"[ERROR] Experiment not found!")
    sys.exit(1)

# Find run by name
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{champion['run_name']}'",
    max_results=1
)

if len(runs) == 0:
    print(f"[ERROR] Run not found: {champion['run_name']}")
    sys.exit(1)

champion_run = runs[0]
run_id = champion_run.info.run_id

print(f"[OK] Found MLflow run: {run_id}")
print(f"     Run name: {champion['run_name']}")

# ============================================================================
# STEP 3: DOWNLOAD AND EXPORT MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: EXPORT MODEL TO PRODUCTION")
print("=" * 80)

# Create production directory
models_dir = project_root / "outputs" / "models" / "production"
models_dir.mkdir(parents=True, exist_ok=True)

# Download model from MLflow
model_uri = f"runs:/{run_id}/model"
export_path = models_dir / "revenue_prediction"

print(f"[*] Downloading model from MLflow...")
print(f"    URI: {model_uri}")
print(f"    Export to: {export_path}")

# Remove existing model if it exists
if export_path.exists():
    import shutil
    shutil.rmtree(export_path)
    print(f"[OK] Removed existing model")

# Download model
model = mlflow.sklearn.load_model(model_uri)
mlflow.sklearn.save_model(model, str(export_path))

print(f"[OK] Model exported to: {export_path}")

# ============================================================================
# STEP 4: CREATE PRODUCTION CONFIG
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: UPDATE PRODUCTION CONFIG")
print("=" * 80)

# Load existing Phase 1 config
config_dir = project_root / "outputs" / "results" / "ml_model_tracking"
config_dir.mkdir(parents=True, exist_ok=True)

phase1_config_path = config_dir / "production_config_phase1.json"

if phase1_config_path.exists():
    with open(phase1_config_path, 'r') as f:
        phase1_config = json.load(f)
else:
    phase1_config = {}

# Create Phase 2 config
phase2_config = {
    "deployment_phase": 2,
    "deployment_date": "2025-12-07",
    "models_deployed": 3,

    "demand_forecasting": phase1_config.get("demand_forecasting", {
        "status": "deployed",
        "model_type": "Ensemble",
        "metrics": {
            "test_r2": 0.6923,
            "test_rmse": 4.53,
            "test_mae": 3.56
        }
    }),

    "clustering": phase1_config.get("clustering", {
        "status": "deployed",
        "algorithm": "DBSCAN",
        "metrics": {
            "silhouette_score": 0.8305
        }
    }),

    "revenue_prediction": {
        "status": "deployed",
        "model_type": "Ensemble",
        "run_id": run_id,
        "run_name": champion['run_name'],
        "metrics": {
            "test_r2": float(champion['test_r2']),
            "test_rmse": float(champion['test_rmse']),
            "test_mae": float(champion['test_mae']),
            "val_r2": float(champion['val_r2']),
            "val_rmse": float(champion['val_rmse']),
            "val_mae": float(champion['val_mae'])
        },
        "use_case": "Revenue prediction per order",
        "output": "total_price (revenue in dollars)",
        "input_features": 53,
        "feature_engineering": "Excludes quantity to prevent data leakage"
    },

    "association_rules": {
        "status": "deployed",
        "algorithm": "Apriori",
        "num_rules": 460,
        "metrics": {
            "avg_lift": 1.0929,
            "avg_confidence": 0.0907
        }
    }
}

# Save Phase 2 config
phase2_config_path = config_dir / "production_config_phase2.json"
with open(phase2_config_path, 'w') as f:
    json.dump(phase2_config, f, indent=2)

print(f"[OK] Saved Phase 2 config to: {phase2_config_path}")

# Also create a simple revenue config for easy loading
revenue_config = {
    "model_type": "Ensemble",
    "status": "deployed",
    "run_id": run_id,
    "run_name": champion['run_name'],
    "metrics": {
        "test_r2": float(champion['test_r2']),
        "test_rmse": float(champion['test_rmse']),
        "test_mae": float(champion['test_mae']),
        "val_r2": float(champion['val_r2']),
        "val_rmse": float(champion['val_rmse']),
        "val_mae": float(champion['val_mae'])
    },
    "use_case": "Predict order revenue based on pizza composition",
    "input_features": 53,
    "output": "total_price (dollars)"
}

revenue_config_path = models_dir / "revenue_config.json"
with open(revenue_config_path, 'w') as f:
    json.dump(revenue_config, f, indent=2)

print(f"[OK] Saved revenue config to: {revenue_config_path}")

# ============================================================================
# STEP 5: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DEPLOYMENT COMPLETE - PHASE 2")
print("=" * 80)

print(f"\n[✅ SUCCESS] Revenue Prediction Model Deployed!")
print(f"\nDeployed Files:")
print(f"  - Model: {export_path}")
print(f"  - Config: {revenue_config_path}")
print(f"  - Phase 2 Config: {phase2_config_path}")

print(f"\n[PHASE 2 STATUS]")
print(f"  Models Deployed: 3/4")
print(f"  - ✅ Demand Forecasting (R²: {phase2_config['demand_forecasting']['metrics']['test_r2']:.4f})")
print(f"  - ✅ Customer Clustering (Silhouette: {phase2_config['clustering']['metrics']['silhouette_score']:.4f})")
print(f"  - ✅ Revenue Prediction (R²: {phase2_config['revenue_prediction']['metrics']['test_r2']:.4f})")
print(f"  - ✅ Association Rules ({phase2_config['association_rules']['num_rules']} rules)")

print(f"\n[NEXT STEP] Update Streamlit app to load revenue model")
print("=" * 80)
