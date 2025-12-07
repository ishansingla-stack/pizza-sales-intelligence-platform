"""
Build Ensemble Configurations for Revenue Prediction
- Loads existing 80 model results (NO RE-TRAINING)
- Builds 8 different ensemble configurations
- 3 hyperparameters × 2 values = 8 configs
- Total: 80 models + 8 ensembles = 88 runs
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent  # scripts/04_model_selection -> scripts -> pizza-intelligence
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import time
from src.config_loader import ConfigLoader

# Monkey-patch MLflow to prevent emoji output errors on Windows
import mlflow.tracking._tracking_service.client
original_log_url = mlflow.tracking._tracking_service.client.TrackingServiceClient._log_url
mlflow.tracking._tracking_service.client.TrackingServiceClient._log_url = lambda self, run_id: None

print("=" * 80)
print("BUILD ENSEMBLE CONFIGURATIONS - REVENUE PREDICTION")
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
print("STEP 2: LOAD EXISTING MODEL RESULTS")
print("=" * 80)

results_path = Path(project_root) / "outputs" / "results" / "ml_model_tracking" / "revenue_prediction_runs.csv"
existing_results = pd.read_csv(results_path)

print(f"[OK] Loaded existing results: {len(existing_results)} runs")
print(f"   Models: {existing_results['model_name'].nunique()}")

# Remove old ensemble entries
model_results = existing_results[existing_results['model_name'] != 'Ensemble'].copy()
print(f"[*] Removed old ensembles: {len(model_results)} model runs remaining")

#%% Define Ensemble Hyperparameters
print("\n" + "=" * 80)
print("STEP 3: DEFINE ENSEMBLE HYPERPARAMETERS (3 × 2 = 8 CONFIGS)")
print("=" * 80)

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
ensemble_configs = [dict(zip(ensemble_hyperparameters.keys(), combo))
                   for combo in itertools.product(*ensemble_hyperparameters.values())]

#%% Model Configs
all_model_configs = {
    "DecisionTree": {"model_class": DecisionTreeRegressor, "hyperparameters": {"max_depth": None, "min_samples_split": None, "min_samples_leaf": None}},
    "LinearRegression": {"model_class": LinearRegression, "hyperparameters": {"fit_intercept": None, "positive": None, "copy_X": None}},
    "SVR": {"model_class": SVR, "hyperparameters": {"C": None, "epsilon": None, "kernel": None}},
    "NeuralNetwork": {"model_class": MLPRegressor, "hyperparameters": {"hidden_layer_sizes": None, "activation": None, "alpha": None}},
    "RandomForest": {"model_class": RandomForestRegressor, "hyperparameters": {"n_estimators": None, "max_depth": None, "min_samples_split": None}},
    "XGBoost": {"model_class": XGBRegressor, "hyperparameters": {"n_estimators": None, "max_depth": None, "learning_rate": None}},
    "KNN": {"model_class": KNeighborsRegressor, "hyperparameters": {"n_neighbors": None, "weights": None, "p": None}},
    "Ridge": {"model_class": Ridge, "hyperparameters": {"alpha": None, "fit_intercept": None, "solver": None}},
    "Lasso": {"model_class": Lasso, "hyperparameters": {"alpha": None, "fit_intercept": None, "max_iter": None}},
    "ElasticNet": {"model_class": ElasticNet, "hyperparameters": {"alpha": None, "l1_ratio": None, "max_iter": None}}
}

#%% Helper Functions
def evaluate_model(model, X, y, dataset_name=""):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    return {
        f'{dataset_name}_rmse': np.sqrt(mean_squared_error(y, y_pred)),
        f'{dataset_name}_mae': mean_absolute_error(y, y_pred),
        f'{dataset_name}_r2': r2_score(y, y_pred)
    }

def create_model_from_row(row):
    """Recreate a trained model from results row"""
    model_name = row['model_name']
    model_class = all_model_configs[model_name]["model_class"]

    # Extract hyperparameters
    param_names = list(all_model_configs[model_name]["hyperparameters"].keys())
    params = {param: row[param] for param in param_names if param in row}

    # Convert integer parameters (CSV stores as float)
    int_params = ['max_depth', 'min_samples_split', 'min_samples_leaf',
                  'n_estimators', 'n_neighbors', 'max_iter']
    for param in int_params:
        if param in params and params[param] is not None and not isinstance(params[param], str):
            params[param] = int(params[param])

    # Convert tuple parameters (CSV stores as string)
    tuple_params = ['hidden_layer_sizes']
    for param in tuple_params:
        if param in params and params[param] is not None:
            if isinstance(params[param], str):
                # Remove quotes and whitespace
                clean_str = params[param].strip().strip("'\"")
                try:
                    # Use ast.literal_eval for safe evaluation
                    import ast
                    params[param] = ast.literal_eval(clean_str)
                except:
                    # If that fails, try manual parsing
                    if clean_str.startswith('(') and clean_str.endswith(')'):
                        values = clean_str[1:-1].split(',')
                        params[param] = tuple(int(v.strip()) for v in values if v.strip())
                    else:
                        # Single value - convert to tuple
                        try:
                            params[param] = (int(float(clean_str)),)
                        except:
                            pass
            elif not isinstance(params[param], tuple):
                # Convert single number to tuple
                try:
                    params[param] = (int(float(params[param])),)
                except:
                    pass

    # Create model instance
    if model_name in ['DecisionTree', 'RandomForest']:
        return model_class(random_state=42, **params)
    elif model_name == 'NeuralNetwork':
        return MLPRegressor(max_iter=500, random_state=42, **params)
    elif model_name == 'XGBoost':
        return XGBRegressor(random_state=42, tree_method='auto', verbosity=0, **params)
    else:
        return model_class(**params)

#%% Build Ensembles
print("\n" + "=" * 80)
print("STEP 4: BUILD 8 ENSEMBLE CONFIGURATIONS")
print("=" * 80)

epsilon = 1e-10
ensemble_results = []

for config_idx, config in enumerate(ensemble_configs):
    print(f"\n[{config_idx+1}/8] Building Ensemble: {config}")

    n_models = config['n_models']
    weighting = config['weighting']
    diversity = config['diversity']

    # Select top N models
    if diversity:
        # Enforce different model types
        selected_models = []
        seen_model_types = set()
        for idx, row in model_results.nsmallest(len(model_results), 'val_rmse').iterrows():
            model_type = row['model_name']
            if model_type not in seen_model_types:
                selected_models.append(row)
                seen_model_types.add(model_type)
                if len(selected_models) == n_models:
                    break
    else:
        # Select top N regardless of type
        selected_models = [row for _, row in model_results.nsmallest(n_models, 'val_rmse').iterrows()]

    print(f"  Selected models ({n_models}, diversity={diversity}):")
    for model_row in selected_models:
        print(f"    {model_row['model_name']:20s} | Val RMSE: {model_row['val_rmse']:.4f}")

    # Rebuild each model and fit
    start_time = time.time()
    estimators = []
    val_rmses = []

    for model_row in selected_models:
        model = create_model_from_row(model_row)
        model.fit(X_train, y_train)
        estimators.append((model_row['model_name'].lower() + f"_{len(estimators)}", model))
        val_rmses.append(model_row['val_rmse'])

    # Calculate weights
    if weighting == 'uniform':
        weights = None
        print(f"  Weighting: uniform (equal weights)")
    else:  # inverse_rmse
        weights = 1.0 / (np.array(val_rmses) + epsilon)
        weights = weights / weights.sum()
        print(f"  Weighting: inverse_rmse")
        for (name, _), weight in zip(estimators, weights):
            print(f"    {name:25s}: {weight:.4f}")

    # Build and train ensemble
    ensemble = VotingRegressor(estimators=estimators, weights=weights)
    ensemble.fit(X_train, y_train)

    # Evaluate
    train_metrics = evaluate_model(ensemble, X_train, y_train, "train")
    val_metrics = evaluate_model(ensemble, X_val, y_val, "val")
    test_metrics = evaluate_model(ensemble, X_test, y_test, "test")
    training_time = time.time() - start_time

    print(f"  Val RMSE: {val_metrics['val_rmse']:.4f} | Test RMSE: {test_metrics['test_rmse']:.4f}")

    # Log to MLflow
    run_name = f"Ensemble_n{n_models}_{weighting}_div{diversity}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "Ensemble")
        mlflow.log_param("n_models", n_models)
        mlflow.log_param("weighting", weighting)
        mlflow.log_param("diversity", diversity)

        for idx, ((name, _), val_rmse) in enumerate(zip(estimators, val_rmses)):
            mlflow.log_param(f"model_{idx+1}", selected_models[idx]['model_name'])
            mlflow.log_param(f"model_{idx+1}_val_rmse", val_rmse)
            if weights is not None:
                mlflow.log_param(f"weight_{idx+1}", float(weights[idx]))

        mlflow.log_metric("training_time", training_time)
        for metric_dict in [train_metrics, val_metrics, test_metrics]:
            for metric_name, metric_value in metric_dict.items():
                mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(ensemble, artifact_path="ensemble_model")

    # Store results
    ensemble_result = {
        'model_name': 'Ensemble',
        'run_name': run_name,
        'training_time': training_time,
        'n_models': n_models,
        'weighting': weighting,
        'diversity': diversity,
        **train_metrics,
        **val_metrics,
        **test_metrics
    }
    ensemble_results.append(ensemble_result)

#%% Save Results
print("\n" + "=" * 80)
print("STEP 5: SAVE UPDATED RESULTS")
print("=" * 80)

# Combine model results with new ensembles
ensemble_df = pd.DataFrame(ensemble_results)
final_results = pd.concat([model_results, ensemble_df], ignore_index=True)

# Save
output_path = Path(project_root) / "outputs" / "results" / "ml_model_tracking" / "revenue_prediction_runs.csv"
final_results.to_csv(output_path, index=False)

print(f"[OK] Saved to: {output_path}")
print(f"   Total runs: {len(final_results)}")
print(f"   Models: {len(model_results)}")
print(f"   Ensembles: {len(ensemble_df)}")

#%% Final Summary
print("\n" + "=" * 80)
print("FINAL SUMMARY - ENSEMBLE BUILDING")
print("=" * 80)

print(f"\n[BEST ENSEMBLE] By Validation RMSE:")
best_ensemble = ensemble_df.nsmallest(1, 'val_rmse').iloc[0]
print(f"  Config: n_models={best_ensemble['n_models']}, weighting={best_ensemble['weighting']}, diversity={best_ensemble['diversity']}")
print(f"  Val RMSE: {best_ensemble['val_rmse']:.4f}")
print(f"  Test RMSE: {best_ensemble['test_rmse']:.4f}")
print(f"  Val R2: {best_ensemble['val_r2']:.4f}")

print(f"\n[ALL ENSEMBLES] Ranked by Validation RMSE:")
for idx, row in ensemble_df.nsmallest(8, 'val_rmse').iterrows():
    print(f"  [{idx+1}] n={row['n_models']}, {row['weighting']:12s}, div={row['diversity']} | "
          f"Val RMSE: {row['val_rmse']:.4f} | Test RMSE: {row['test_rmse']:.4f}")

print("\n" + "=" * 80)
print("[COMPLETE] Ensemble building complete!")
print(f"Built 8 ensemble configurations")
print(f"Total: {len(final_results)} runs (80 models + 8 ensembles)")
print("View in MLflow: https://dbc-e5a86ed3-c332.cloud.databricks.com/#mlflow/experiments")
print("=" * 80)
