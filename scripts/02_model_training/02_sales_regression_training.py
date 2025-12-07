"""
Sales Prediction - Regression Training Pipeline
- Target: total_price (revenue)
- 6 Regression Models: DecisionTree, LinearRegression, NeuralNetwork, RandomForest, XGBoost, KNN
- 3 hyperparameters per model × 2 values each = 8 configurations
- Total: 6 models × 8 configs + 8 ensembles = 56 runs
- Validation Set for Hyperparameter Selection
- Dynamic MLflow Run Naming
"""

import sys
import os
from pathlib import Path

# Get project root (when running with -m, use cwd)
if __name__ == "__main__":
    project_root = Path(os.getcwd())
else:
    project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
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
print("SALES PREDICTION - REGRESSION TRAINING")
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

#%% Model Configurations - 3 Params × 2 Values = 8 Configs Each
print("\n" + "=" * 80)
print("STEP 2: DEFINE MODELS - 3 HYPERPARAMETERS × 2 VALUES = 8 CONFIGS")
print("=" * 80)

model_configs = {
    "DecisionTree": {
        "model_class": DecisionTreeRegressor,
        "hyperparameters": {
            "max_depth": [10, 20],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 4]
        }
    },
    "LinearRegression": {
        "model_class": LinearRegression,
        "hyperparameters": {
            "fit_intercept": [True, False],
            "positive": [True, False],
            "copy_X": [True, False]  # 3rd parameter
        }
    },
    "NeuralNetwork": {
        "model_class": MLPRegressor,
        "hyperparameters": {
            "hidden_layer_sizes": [(50,), (100,)],
            "activation": ['relu', 'tanh'],
            "alpha": [0.0001, 0.001]
        }
    },
    "RandomForest": {
        "model_class": RandomForestRegressor,
        "hyperparameters": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "min_samples_split": [2, 5]
        }
    },
    "XGBoost": {
        "model_class": XGBRegressor,
        "hyperparameters": {
            "n_estimators": [100, 200],
            "max_depth": [3, 7],
            "learning_rate": [0.1, 0.2]
        }
    },
    "KNN": {
        "model_class": KNeighborsRegressor,
        "hyperparameters": {
            "n_neighbors": [3, 7],
            "weights": ['uniform', 'distance'],
            "p": [1, 2]
        }
    }
}

# Verify 8 combinations per model
print(f"\n[*] Hyperparameter Configurations:")
for model_name, config in model_configs.items():
    n_combos = 1
    for param_values in config["hyperparameters"].values():
        n_combos *= len(param_values)
    print(f"  {model_name:20s}: {n_combos} configurations")

total_runs = sum(len(list(itertools.product(*config["hyperparameters"].values())))
                 for config in model_configs.values())
print(f"\n[*] Total runs: {total_runs} base models + 8 ensembles = {total_runs + 8} total")

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

#%% Training Loop
print("\n" + "=" * 80)
print("STEP 3: HYPERPARAMETER TUNING (80 RUNS)")
print("=" * 80)

all_results = []
current_run = 0

for model_name, config in model_configs.items():
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
                if model_name == "NeuralNetwork":
                    model = model_class(max_iter=500, random_state=42, **params)
                elif model_name == "XGBoost":
                    model = model_class(random_state=42, tree_method='auto', verbosity=0, **params)
                elif model_name == "RandomForest":
                    model = model_class(random_state=42, **params)
                elif model_name == "DecisionTree":
                    model = model_class(random_state=42, **params)
                else:
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

                mlflow.sklearn.log_model(model, "model")

                # Track best
                if val_metrics['val_rmse'] < best_val_rmse:
                    best_val_rmse = val_metrics['val_rmse']
                    best_params = params

                # Store results
                result = {
                    'model_name': model_name,
                    'config_num': idx + 1,
                    'run_name': run_name,
                    'training_time': training_time,
                    **params,
                    **train_metrics,
                    **val_metrics,
                    **test_metrics
                }
                all_results.append(result)

                progress = (current_run / total_runs) * 100
                print(f"  [{current_run:2d}/{total_runs}] ({progress:5.1f}%) "
                      f"Config {idx+1}/8 | Val RMSE: {val_metrics['val_rmse']:.4f}")

        except Exception as e:
            print(f"  [ERROR] Failed: {run_name} - {str(e)}")
            continue

    print(f"\n[BEST {model_name}] Val RMSE: {best_val_rmse:.4f} | Params: {best_params}")

#%% Ensemble Models - 8 Configurations
print("\n" + "=" * 80)
print("ENSEMBLE: 8 Configurations (Top 3 or 5 Models, Weighted/Uniform)")
print("=" * 80)

results_df = pd.DataFrame(all_results)

# Get top 5 models by validation R²
top_models = results_df.nlargest(5, 'val_r2')

print(f"\n[*] Top 5 models for ensemble:")
for idx, row in top_models.iterrows():
    print(f"  {row['model_name']:20s} | Val R²: {row['val_r2']:.4f}, Val RMSE: {row['val_rmse']:.4f}")

# Ensemble configurations - 8 ensembles with different combinations
ensemble_configs = {
    "n_estimators": [3, 5],  # Use top 3 or top 5 models
    "weights": ['uniform', 'inverse_rmse'],  # Equal vs weighted by performance
    "div_by_std": [False, True]  # Whether to divide weights by std
}

ensemble_param_combos = list(itertools.product(*ensemble_configs.values()))

for ens_idx, (n_est, weight_type, div_std) in enumerate(ensemble_param_combos, 1):
    print(f"\n[{ens_idx}/8] Ensemble #{ens_idx} - n={n_est}, weights={weight_type}, div_std={div_std}")

    # Select top N models
    selected = top_models.head(n_est)

    # Create base estimators (retrain with same hyperparams)
    estimators = []
    for idx, row in selected.iterrows():
        model_name = row['model_name']
        model_class = model_configs[model_name]["model_class"]

        # Extract hyperparameters from the row
        param_names = list(model_configs[model_name]["hyperparameters"].keys())
        params_dict = {param: row[param] for param in param_names if param in row}

        # Convert float params to int for models that need it
        if model_name == "XGBoost":
            if 'n_estimators' in params_dict:
                params_dict['n_estimators'] = int(params_dict['n_estimators'])
            if 'max_depth' in params_dict:
                params_dict['max_depth'] = int(params_dict['max_depth'])
            params_dict['random_state'] = 42
            params_dict['verbosity'] = 0
        elif model_name in ["RandomForest", "DecisionTree"]:
            if 'n_estimators' in params_dict:
                params_dict['n_estimators'] = int(params_dict['n_estimators'])
            if 'max_depth' in params_dict:
                params_dict['max_depth'] = int(params_dict['max_depth'])
            if 'min_samples_split' in params_dict:
                params_dict['min_samples_split'] = int(params_dict['min_samples_split'])
            if 'min_samples_leaf' in params_dict:
                params_dict['min_samples_leaf'] = int(params_dict['min_samples_leaf'])
            params_dict['random_state'] = 42
        elif model_name == "NeuralNetwork":
            params_dict['random_state'] = 42

        base_model = model_class(**params_dict)
        estimators.append((f"{model_name}_{idx}", base_model))

    # Calculate weights
    if weight_type == 'uniform':
        weights = None
    elif weight_type == 'inverse_rmse':
        rmse_vals = selected['val_rmse'].values
        if div_std:
            std_vals = selected['val_r2'].std()
            weights = (1.0 / rmse_vals) / (std_vals + 1e-6)
        else:
            weights = 1.0 / rmse_vals
        weights = weights / weights.sum()  # Normalize

    # Train ensemble
    with mlflow.start_run(run_name=f"Ensemble_n{n_est}_{weight_type}_div{div_std}"):
        start_time = time.time()

        mlflow.log_params({
            "n_estimators": n_est,
            "weight_type": weight_type,
            "div_by_std": div_std
        })
        mlflow.log_param("model_type", "Ensemble")

        ensemble = VotingRegressor(estimators=estimators, weights=weights)
        ensemble.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Evaluate
        train_metrics = evaluate_model(ensemble, X_train, y_train, "train")
        val_metrics = evaluate_model(ensemble, X_val, y_val, "val")
        test_metrics = evaluate_model(ensemble, X_test, y_test, "test")

        # Log
        mlflow.log_metric("training_time", training_time)
        for metric_dict in [train_metrics, val_metrics, test_metrics]:
            for metric_name, metric_value in metric_dict.items():
                mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(ensemble, "model")

        all_results.append({
            'model_name': 'Ensemble',
            'config_num': ens_idx,
            'run_name': f"Ensemble_n{n_est}_{weight_type}_div{div_std}",
            'n_estimators': n_est,
            'weight_type': weight_type,
            'div_by_std': div_std,
            'training_time': training_time,
            **train_metrics,
            **val_metrics,
            **test_metrics
        })

        print(f"   Test R²: {test_metrics['test_r2']:.4f}, RMSE: {test_metrics['test_rmse']:.2f}")

print(f"\n[OK] Completed 8 ensemble configurations")

#%% Save Results
print("\n" + "=" * 80)
print("STEP 4: SAVE RESULTS")
print("=" * 80)

results_df = pd.DataFrame(all_results)
output_path = Path(project_root) / "outputs" / "results" / "sales_prediction_results.csv"
results_df.to_csv(output_path, index=False)

print(f"[OK] Saved to: {output_path}")
print(f"   Total runs: {len(results_df)}")

#%% Final Summary
print("\n" + "=" * 80)
print("FINAL SUMMARY - SALES PREDICTION")
print("=" * 80)

best_overall = results_df.nsmallest(1, 'val_rmse').iloc[0]

print(f"\n[CHAMPION] Best Model (by Validation RMSE):")
print(f"  Model: {best_overall['model_name']}")
print(f"  Val RMSE: {best_overall['val_rmse']:.4f}")
print(f"  Val R2: {best_overall['val_r2']:.4f}")
print(f"  Test RMSE: {best_overall['test_rmse']:.4f}")
print(f"  Test R2: {best_overall['test_r2']:.4f}")

print(f"\n[TOP 5] Models by Validation RMSE:")
top_5 = results_df.nsmallest(5, 'val_rmse')[['model_name', 'val_rmse', 'val_r2', 'test_rmse', 'test_r2']]
for idx, row in top_5.iterrows():
    print(f"  {row['model_name']:20s} | Val RMSE: {row['val_rmse']:.4f} | Test RMSE: {row['test_rmse']:.4f} | Val R2: {row['val_r2']:.4f}")

print(f"\n[BY MODEL] Best configuration per model:")
best_per_model = results_df.loc[results_df.groupby('model_name')['val_rmse'].idxmin()]
for _, row in best_per_model.iterrows():
    print(f"  {row['model_name']:20s} | Val RMSE: {row['val_rmse']:.4f} | Test RMSE: {row['test_rmse']:.4f}")

print("\n" + "=" * 80)
print("[COMPLETE] Sales prediction training complete!")
print(f"View in MLflow: https://dbc-e5a86ed3-c332.cloud.databricks.com/#mlflow/experiments")
print("=" * 80)
