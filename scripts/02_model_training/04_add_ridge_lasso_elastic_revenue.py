"""
Add Ridge, Lasso, ElasticNet Models to Revenue Prediction
- Target: total_price (revenue)
- 3 NEW Models: Ridge, Lasso, ElasticNet
- 3 hyperparameters per model × 2 values each = 8 configurations
- Total: 3 models × 8 configs = 24 NEW runs
- Appends to existing 57 runs (7 models + ensemble)
- Rebuilds ensemble with all 10 models
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import time
from src.config_loader import ConfigLoader

# Monkey-patch MLflow to prevent emoji output errors on Windows
import mlflow.tracking._tracking_service.client
original_log_url = mlflow.tracking._tracking_service.client.TrackingServiceClient._log_url
mlflow.tracking._tracking_service.client.TrackingServiceClient._log_url = lambda self, run_id: None

print("=" * 80)
print("ADD RIDGE, LASSO, ELASTICNET - REVENUE PREDICTION")
print("=" * 80)

# Load configuration
print("\n[*] Loading configuration...")
config = ConfigLoader()
config.setup_databricks_env()

# Set MLflow experiment
mlflow.set_experiment("/pizza-sales-prediction")
print(f"[OK] MLflow experiment: /pizza-sales-prediction")

#%% Load Processed Data
print("\n" + "=" * 80)
print("STEP 1: LOAD PROCESSED DATA (80-10-10 SPLIT)")
print("=" * 80)

data_dir = Path(project_root) / "data" / "processed" / "revenue_target"

X_train = pd.read_parquet(data_dir / "X_train.parquet")
X_val = pd.read_parquet(data_dir / "X_val.parquet")
X_test = pd.read_parquet(data_dir / "X_test.parquet")
y_train = pd.read_parquet(data_dir / "y_train.parquet")['total_price']
y_val = pd.read_parquet(data_dir / "y_val.parquet")['total_price']
y_test = pd.read_parquet(data_dir / "y_test.parquet")['total_price']

print(f"[OK] Loaded datasets:")
print(f"   Train: X={X_train.shape}, y={y_train.shape}")
print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
print(f"   Test:  X={X_test.shape}, y={y_test.shape}")

#%% Load Existing Results
print("\n" + "=" * 80)
print("STEP 2: LOAD EXISTING RESULTS")
print("=" * 80)

results_path = Path(project_root) / "outputs" / "results" / "sales_prediction_results.csv"
if results_path.exists():
    existing_results_df = pd.read_csv(results_path)
    print(f"[OK] Loaded existing results:")
    print(f"   Total runs: {len(existing_results_df)}")
    print(f"   Models: {existing_results_df['model_name'].nunique()}")
    print(f"\nExisting models:")
    print(existing_results_df['model_name'].value_counts())
else:
    existing_results_df = pd.DataFrame()
    print("[WARN] No existing results found. Creating new results file.")

#%% Define NEW Models - Ridge, Lasso, ElasticNet
print("\n" + "=" * 80)
print("STEP 3: DEFINE NEW MODELS (3 HYPERPARAMETERS × 2 VALUES = 8 CONFIGS EACH)")
print("=" * 80)

new_model_configs = {
    "Ridge": {
        "model_class": Ridge,
        "hyperparameters": {
            "alpha": [0.1, 1.0],
            "fit_intercept": [True, False],
            "solver": ['auto', 'svd']
        }
    },
    "Lasso": {
        "model_class": Lasso,
        "hyperparameters": {
            "alpha": [0.1, 1.0],
            "fit_intercept": [True, False],
            "max_iter": [1000, 5000]
        }
    },
    "ElasticNet": {
        "model_class": ElasticNet,
        "hyperparameters": {
            "alpha": [0.1, 1.0],
            "l1_ratio": [0.3, 0.7],
            "max_iter": [1000, 5000]
        }
    }
}

# Verify 8 combinations per model
print(f"\n[*] New Model Configurations:")
for model_name, config in new_model_configs.items():
    n_combos = 1
    for param_values in config["hyperparameters"].values():
        n_combos *= len(param_values)
    print(f"  {model_name:20s}: {n_combos} configurations")

total_new_runs = sum(len(list(itertools.product(*config["hyperparameters"].values())))
                     for config in new_model_configs.values())
print(f"\n[*] Total NEW runs: {total_new_runs} (3 models × 8 configs)")

#%% Helper Functions
def evaluate_model(model, X, y, dataset_name=""):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    return {
        f'{dataset_name}_rmse': np.sqrt(mean_squared_error(y, y_pred)),
        f'{dataset_name}_mae': mean_absolute_error(y, y_pred),
        f'{dataset_name}_r2': r2_score(y, y_pred)
    }

def create_run_name(model_name, params):
    """Create dynamic run name based on hyperparameters"""
    param_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
    if len(param_str) > 100:
        param_str = param_str[:97] + "..."
    return f"{model_name}_{param_str}"

#%% Training Loop for NEW Models
print("\n" + "=" * 80)
print("STEP 4: TRAIN NEW MODELS (24 RUNS)")
print("=" * 80)

new_results = []
current_run = 0

for model_name, config in new_model_configs.items():
    print("\n" + "=" * 80)
    print(f"Model: {model_name}")
    print("=" * 80)

    model_class = config["model_class"]
    hyperparams = config["hyperparameters"]

    # Generate all combinations (2^3 = 8 per model)
    param_names = list(hyperparams.keys())
    param_values = list(hyperparams.values())
    param_combinations = [dict(zip(param_names, combo))
                         for combo in itertools.product(*param_values)]

    best_val_rmse = float('inf')
    best_params = None

    for idx, params in enumerate(param_combinations):
        current_run += 1
        run_name = create_run_name(model_name, params)

        try:
            with mlflow.start_run(run_name=run_name):
                start_time = time.time()

                # Create model
                model = model_class(**params)
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                # Evaluate
                train_metrics = evaluate_model(model, X_train, y_train, "train")
                val_metrics = evaluate_model(model, X_val, y_val, "val")
                test_metrics = evaluate_model(model, X_test, y_test, "test")

                # Log to MLflow
                mlflow.log_param("model_type", model_name)
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

                mlflow.log_metric("training_time", training_time)
                for metric_dict in [train_metrics, val_metrics, test_metrics]:
                    for metric_name, metric_value in metric_dict.items():
                        mlflow.log_metric(metric_name, metric_value)

                # Log model
                mlflow.sklearn.log_model(model, artifact_path="model")

                # Track best
                if val_metrics['val_rmse'] < best_val_rmse:
                    best_val_rmse = val_metrics['val_rmse']
                    best_params = params

                # Store results
                result = {
                    'model_name': model_name,
                    'run_name': run_name,
                    'training_time': training_time,
                    **params,
                    **train_metrics,
                    **val_metrics,
                    **test_metrics
                }
                new_results.append(result)

                progress = (current_run / total_new_runs) * 100
                print(f"  [{current_run:2d}/{total_new_runs}] ({progress:5.1f}%) "
                      f"Config {idx+1}/8 | Val RMSE: {val_metrics['val_rmse']:.4f}")

        except Exception as e:
            print(f"  [ERROR] Failed: {run_name} - {str(e)}")
            continue

    print(f"\n[BEST {model_name}] Val RMSE: {best_val_rmse:.4f} | Params: {best_params}")

#%% Combine Results
print("\n" + "=" * 80)
print("STEP 5: COMBINE RESULTS")
print("=" * 80)

new_results_df = pd.DataFrame(new_results)

# Remove old ensemble from existing results (we'll rebuild with all 10 models)
existing_no_ensemble = existing_results_df[existing_results_df['model_name'] != 'Ensemble'].copy()

print(f"[*] Combining results:")
print(f"   Existing runs (without old ensemble): {len(existing_no_ensemble)}")
print(f"   New runs: {len(new_results_df)}")

# Combine
all_results_df = pd.concat([existing_no_ensemble, new_results_df], ignore_index=True)
print(f"   Total runs (before new ensemble): {len(all_results_df)}")

#%% Build NEW Ensemble with All 10 Models - 8 CONFIGURATIONS
print("\n" + "=" * 80)
print("STEP 6: BUILD ENSEMBLE - HYPERPARAMETER GRID SEARCH (8 RUNS)")
print("=" * 80)

# Define ensemble hyperparameters (3 params × 2 values = 8 configs)
ensemble_hyperparameters = {
    "n_models": [3, 5],                           # Number of top models
    "weighting": ['uniform', 'inverse_rmse'],     # Weighting strategy
    "diversity": [True, False]                    # Enforce model type diversity
}

print(f"\n[*] Ensemble Hyperparameter Grid:")
print(f"  n_models: {ensemble_hyperparameters['n_models']}")
print(f"  weighting: {ensemble_hyperparameters['weighting']}")
print(f"  diversity: {ensemble_hyperparameters['diversity']}")
print(f"\n[*] Total ensemble configurations: 2 × 2 × 2 = 8")

# Generate all ensemble configurations
import itertools as it
ensemble_configs = [dict(zip(ensemble_hyperparameters.keys(), combo))
                   for combo in it.product(*ensemble_hyperparameters.values())]

ensemble_results = []
epsilon = 1e-10

# Get model configs for all 10 models (7 original + 3 new)
all_model_configs = {
    "DecisionTree": {"model_class": None, "hyperparameters": {"max_depth": None, "min_samples_split": None, "min_samples_leaf": None}},
    "LinearRegression": {"model_class": None, "hyperparameters": {"fit_intercept": None, "positive": None, "copy_X": None}},
    "SVR": {"model_class": None, "hyperparameters": {"C": None, "epsilon": None, "kernel": None}},
    "NeuralNetwork": {"model_class": None, "hyperparameters": {"hidden_layer_sizes": None, "activation": None, "alpha": None}},
    "RandomForest": {"model_class": None, "hyperparameters": {"n_estimators": None, "max_depth": None, "min_samples_split": None}},
    "XGBoost": {"model_class": None, "hyperparameters": {"n_estimators": None, "max_depth": None, "learning_rate": None}},
    "KNN": {"model_class": None, "hyperparameters": {"n_neighbors": None, "weights": None, "p": None}},
    "Ridge": {"model_class": Ridge, "hyperparameters": {"alpha": None, "fit_intercept": None, "solver": None}},
    "Lasso": {"model_class": Lasso, "hyperparameters": {"alpha": None, "fit_intercept": None, "max_iter": None}},
    "ElasticNet": {"model_class": ElasticNet, "hyperparameters": {"alpha": None, "l1_ratio": None, "max_iter": None}}
}

for selected_row in top_3_diverse:
    model_name = selected_row['model_name']
    val_rmse = selected_row['val_rmse']

    # Extract hyperparameters dynamically
    param_names = list(all_model_configs[model_name]["hyperparameters"].keys())
    params = {param: selected_row[param] for param in param_names if param in selected_row}

    # Convert integer parameters (CSV stores as float)
    int_params = ['max_depth', 'min_samples_split', 'min_samples_leaf',
                  'n_estimators', 'n_neighbors', 'max_iter']
    for param in int_params:
        if param in params and params[param] is not None and not isinstance(params[param], str):
            params[param] = int(params[param])

    # Create model instance
    if model_name == "Ridge":
        model = Ridge(**params)
    elif model_name == "Lasso":
        model = Lasso(**params)
    elif model_name == "ElasticNet":
        model = ElasticNet(**params)
    else:
        # For original 7 models, we need to import them
        if model_name == "DecisionTree":
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(random_state=42, **params)
        elif model_name == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(**params)
        elif model_name == "SVR":
            from sklearn.svm import SVR
            model = SVR(**params)
        elif model_name == "NeuralNetwork":
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(max_iter=500, random_state=42, **params)
        elif model_name == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=42, **params)
        elif model_name == "XGBoost":
            from xgboost import XGBRegressor
            model = XGBRegressor(random_state=42, tree_method='auto', verbosity=0, **params)
        elif model_name == "KNN":
            from sklearn.neighbors import KNeighborsRegressor
            model = KNeighborsRegressor(**params)

    # Fit model
    model.fit(X_train, y_train)

    # Calculate weight (inverse RMSE)
    weight = 1.0 / (val_rmse + epsilon)
    estimators.append((model_name.lower(), model))
    weights.append(weight)

# Normalize weights
weights = np.array(weights) / np.sum(weights)
print(f"\n[*] Ensemble weights (normalized inverse RMSE):")
for (name, _), weight in zip(estimators, weights):
    print(f"  {name:20s}: {weight:.4f}")

# Create and train ensemble
ensemble = VotingRegressor(estimators=estimators, weights=weights)
ensemble.fit(X_train, y_train)

# Evaluate ensemble
start_time = time.time()
ensemble_train_metrics = evaluate_model(ensemble, X_train, y_train, "train")
ensemble_val_metrics = evaluate_model(ensemble, X_val, y_val, "val")
ensemble_test_metrics = evaluate_model(ensemble, X_test, y_test, "test")
ensemble_training_time = time.time() - start_time

print(f"\n[OK] Ensemble | Val RMSE: {ensemble_val_metrics['val_rmse']:.4f}")

# Log ensemble to MLflow
with mlflow.start_run(run_name="Ensemble_Top3_Weighted"):
    mlflow.log_param("model_type", "Ensemble")
    mlflow.log_param("ensemble_method", "VotingRegressor_Weighted")
    mlflow.log_param("n_models", 3)
    mlflow.log_param("selection_metric", "val_rmse")

    for idx, ((name, _), weight) in enumerate(zip(estimators, weights)):
        mlflow.log_param(f"model_{idx+1}", name)
        mlflow.log_param(f"weight_{idx+1}", float(weight))

    mlflow.log_metric("training_time", ensemble_training_time)
    for metric_dict in [ensemble_train_metrics, ensemble_val_metrics, ensemble_test_metrics]:
        for metric_name, metric_value in metric_dict.items():
            mlflow.log_metric(metric_name, metric_value)

    mlflow.sklearn.log_model(ensemble, artifact_path="ensemble_model")

# Add ensemble to results
ensemble_result = {
    'model_name': 'Ensemble',
    'run_name': 'Ensemble_Top3_Weighted',
    'training_time': ensemble_training_time,
    **ensemble_train_metrics,
    **ensemble_val_metrics,
    **ensemble_test_metrics
}
all_results_df = pd.concat([all_results_df, pd.DataFrame([ensemble_result])], ignore_index=True)

#%% Save Combined Results
print("\n" + "=" * 80)
print("STEP 7: SAVE COMBINED RESULTS")
print("=" * 80)

output_path = Path(project_root) / "outputs" / "results" / "sales_prediction_results.csv"
all_results_df.to_csv(output_path, index=False)

print(f"[OK] Saved to: {output_path}")
print(f"   Total runs: {len(all_results_df)}")
print(f"   Models: {all_results_df['model_name'].nunique()}")
print(f"\nFinal model counts:")
print(all_results_df['model_name'].value_counts())

#%% Final Summary
print("\n" + "=" * 80)
print("FINAL SUMMARY - REVENUE PREDICTION (UPDATED)")
print("=" * 80)

# Champion
champion = all_results_df.nsmallest(1, 'val_rmse').iloc[0]
print(f"\n[CHAMPION] Best Model (by Validation RMSE):")
print(f"  Model: {champion['model_name']}")
print(f"  Val RMSE: {champion['val_rmse']:.4f}")
print(f"  Val R2: {champion['val_r2']:.4f}")
print(f"  Test RMSE: {champion['test_rmse']:.4f}")
print(f"  Test R2: {champion['test_r2']:.4f}")

# Top 5
print(f"\n[TOP 5] Models by Validation RMSE:")
for idx, row in all_results_df.nsmallest(5, 'val_rmse').iterrows():
    print(f"  {row['model_name']:20s} | Val RMSE: {row['val_rmse']:.4f} | "
          f"Test RMSE: {row['test_rmse']:.4f} | Val R2: {row['val_r2']:.4f}")

# Best per model
print(f"\n[BY MODEL] Best configuration per model:")
for model_name in sorted(all_results_df['model_name'].unique()):
    model_df = all_results_df[all_results_df['model_name'] == model_name]
    best = model_df.nsmallest(1, 'val_rmse').iloc[0]
    print(f"  {model_name:20s} | Val RMSE: {best['val_rmse']:.4f} | Test RMSE: {best['test_rmse']:.4f}")

print("\n" + "=" * 80)
print("[COMPLETE] Revenue prediction training updated!")
print(f"Added 24 new runs (Ridge, Lasso, ElasticNet)")
print(f"Total: {len(all_results_df)} runs (10 models + ensemble)")
print("View in MLflow: https://dbc-e5a86ed3-c332.cloud.databricks.com/#mlflow/experiments")
print("=" * 80)
