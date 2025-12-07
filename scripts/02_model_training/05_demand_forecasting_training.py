"""
Demand Forecasting - Model Training
Predicts total hourly pizza demand using time series features
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
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
from src.config_loader import ConfigLoader

# Monkey-patch MLflow to prevent emoji output errors on Windows
import mlflow.tracking._tracking_service.client
original_log_url = mlflow.tracking._tracking_service.client.TrackingServiceClient._log_url
mlflow.tracking._tracking_service.client.TrackingServiceClient._log_url = lambda self, run_id: None

print("=" * 80)
print("DEMAND FORECASTING - MODEL TRAINING")
print("=" * 80)

config = ConfigLoader()
config.setup_databricks_env()

mlflow.set_experiment("/pizza-demand-forecasting")
print(f"[OK] MLflow experiment: /pizza-demand-forecasting")

# Load data
print("\n[*] Loading demand forecasting data...")
data_dir = Path(project_root) / "data" / "processed" / "demand_forecasting"

X_train = pd.read_parquet(data_dir / "X_train.parquet")
X_val = pd.read_parquet(data_dir / "X_val.parquet")
X_test = pd.read_parquet(data_dir / "X_test.parquet")
y_train = pd.read_parquet(data_dir / "y_train.parquet")['total_pizzas']
y_val = pd.read_parquet(data_dir / "y_val.parquet")['total_pizzas']
y_test = pd.read_parquet(data_dir / "y_test.parquet")['total_pizzas']

print(f"[OK] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Model configs - 8 models × 8 configs = 64 runs
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
            "copy_X": [True, False],
            "positive": [False, True]
        }
    },
    "SVR": {
        "model_class": SVR,
        "hyperparameters": {
            "C": [1.0, 10.0],
            "epsilon": [0.1, 0.2],
            "kernel": ['linear', 'rbf']
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

print(f"\n[*] Training {len(model_configs)} model types...")

results = []
total_models = sum([len(list(itertools.product(*config["hyperparameters"].values())))
                    for config in model_configs.values()])
current = 0

for model_name, config in model_configs.items():
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")

    param_names = list(config["hyperparameters"].keys())
    param_values = list(config["hyperparameters"].values())

    for param_combo in itertools.product(*param_values):
        current += 1
        params = dict(zip(param_names, param_combo))

        print(f"\n[{current}/{total_models}] {model_name} - {params}")

        with mlflow.start_run(run_name=f"{model_name}_{'_'.join([f'{k}={v}' for k,v in params.items()])}"):
            # Log params
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)

            # Train
            if model_name == "XGBoost":
                params['random_state'] = 42
                params['verbosity'] = 0
            elif model_name in ["RandomForest", "GradientBoosting"]:
                params['random_state'] = 42

            model = config["model_class"](**params)
            model.fit(X_train, y_train)

            # Predict
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Log metrics
            mlflow.log_metrics({
                "train_r2": train_r2, "val_r2": val_r2, "test_r2": test_r2,
                "train_rmse": train_rmse, "val_rmse": val_rmse, "test_rmse": test_rmse,
                "train_mae": train_mae, "val_mae": val_mae, "test_mae": test_mae
            })

            # Log model
            mlflow.sklearn.log_model(model, "model")

            print(f"   Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")

            results.append({
                "model_name": model_name,
                **params,
                "train_r2": train_r2, "val_r2": val_r2, "test_r2": test_r2,
                "train_rmse": train_rmse, "val_rmse": val_rmse, "test_rmse": test_rmse,
                "train_mae": train_mae, "val_mae": val_mae, "test_mae": test_mae
            })

# Build Ensemble Models (8 configurations)
print(f"\n{'='*80}")
print("BUILDING ENSEMBLE MODELS")
print(f"{'='*80}")

# Get best models from base training
results_df_temp = pd.DataFrame(results)
top_models = results_df_temp.nlargest(5, 'val_r2')

print(f"\nTop 5 base models:")
for i, row in enumerate(top_models.itertuples(), 1):
    print(f"  {i}. {row.model_name} - Val R²: {row.val_r2:.4f}")

# Ensemble configurations - 8 ensembles with different combinations
ensemble_configs = {
    "n_estimators": [3, 5],  # Use top 3 or top 5 models
    "weights": ['uniform', 'inverse_rmse'],  # Equal vs weighted by performance
    "div_by_std": [False, True]  # Whether to divide weights by std
}

ensemble_param_combos = list(itertools.product(*ensemble_configs.values()))

for ens_idx, (n_est, weight_type, div_std) in enumerate(ensemble_param_combos, 1):
    current += 1
    print(f"\n[{current}/{total_models + 8}] Ensemble #{ens_idx} - n={n_est}, weights={weight_type}, div_std={div_std}")

    # Select top N models
    selected = top_models.head(n_est)

    # Create base estimators (retrain with same hyperparams)
    estimators = []
    for idx, row in enumerate(selected.itertuples(), 1):
        # Reconstruct model
        model_config = model_configs[row.model_name]
        params_dict = {k: getattr(row, k) for k in model_config["hyperparameters"].keys() if hasattr(row, k)}

        if row.model_name == "XGBoost":
            # Convert float params to int for XGBoost
            if 'n_estimators' in params_dict:
                params_dict['n_estimators'] = int(params_dict['n_estimators'])
            if 'max_depth' in params_dict:
                params_dict['max_depth'] = int(params_dict['max_depth'])
            params_dict['random_state'] = 42
            params_dict['verbosity'] = 0
        elif row.model_name in ["RandomForest", "DecisionTree"]:
            # Convert float params to int for tree models
            if 'n_estimators' in params_dict:
                params_dict['n_estimators'] = int(params_dict['n_estimators'])
            if 'max_depth' in params_dict:
                params_dict['max_depth'] = int(params_dict['max_depth'])
            if 'min_samples_split' in params_dict:
                params_dict['min_samples_split'] = int(params_dict['min_samples_split'])
            if 'min_samples_leaf' in params_dict:
                params_dict['min_samples_leaf'] = int(params_dict['min_samples_leaf'])
            params_dict['random_state'] = 42
        elif row.model_name == "NeuralNetwork":
            params_dict['random_state'] = 42

        base_model = model_config["model_class"](**params_dict)
        estimators.append((f"{row.model_name}_{idx}", base_model))

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
        mlflow.log_params({
            "n_estimators": n_est,
            "weight_type": weight_type,
            "div_by_std": div_std
        })
        mlflow.log_param("model_type", "Ensemble")

        ensemble = VotingRegressor(estimators=estimators, weights=weights)
        ensemble.fit(X_train, y_train)

        # Predict
        y_train_pred = ensemble.predict(X_train)
        y_val_pred = ensemble.predict(X_val)
        y_test_pred = ensemble.predict(X_test)

        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        mlflow.log_metrics({
            "train_r2": train_r2, "val_r2": val_r2, "test_r2": test_r2,
            "train_rmse": train_rmse, "val_rmse": val_rmse, "test_rmse": test_rmse,
            "train_mae": train_mae, "val_mae": val_mae, "test_mae": test_mae
        })

        mlflow.sklearn.log_model(ensemble, "model")

        print(f"   Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")

        results.append({
            "model_name": "Ensemble",
            "n_estimators": n_est,
            "weight_type": weight_type,
            "div_by_std": div_std,
            "train_r2": train_r2, "val_r2": val_r2, "test_r2": test_r2,
            "train_rmse": train_rmse, "val_rmse": val_rmse, "test_rmse": test_rmse,
            "train_mae": train_mae, "val_mae": val_mae, "test_mae": test_mae
        })

# Save results
results_df = pd.DataFrame(results)
results_dir = Path(project_root) / "outputs" / "results" / "ml_model_tracking"
results_dir.mkdir(parents=True, exist_ok=True)
results_df.to_csv(results_dir / "demand_forecasting_runs.csv", index=False)

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print(f"{'='*80}")
print(f"\nBest model by test R²:")
best = results_df.nlargest(1, 'test_r2').iloc[0]
print(f"  {best['model_name']}")
print(f"  Test R²: {best['test_r2']:.4f}")
print(f"  Test RMSE: {best['test_rmse']:.2f} pizzas")
print(f"  Test MAE: {best['test_mae']:.2f} pizzas")
print(f"\n[OK] Results saved to: {results_dir / 'demand_forecasting_runs.csv'}")
