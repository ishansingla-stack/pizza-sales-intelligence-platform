"""
Model Selection for Unsupervised Learning (Clustering & Association)
- Unlike Regression, we do NOT ensemble (average) these models.
- Instead, we scan the results to select the single "Champion" configuration.
- Total Inputs: ~48 runs from Step 05
"""

import sys
import os
from pathlib import Path

# Add project root to path (FIXED: 3 levels up)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import mlflow
from src.config_loader import ConfigLoader

# Monkey-patch MLflow
import mlflow.tracking._tracking_service.client
mlflow.tracking._tracking_service.client.TrackingServiceClient._log_url = lambda self, run_id: None

print("=" * 80)
print("MODEL SELECTION - UNSUPERVISED LEARNING (CHAMPION SELECTION)")
print("=" * 80)

# Load configuration
config = ConfigLoader()
config.setup_databricks_env()
mlflow.set_experiment("/pizza-clustering-association")

#%% STEP 1: LOAD RESULTS
print("\n" + "=" * 80)
print("STEP 1: LOAD HYPERPARAMETER TUNING RESULTS")
print("=" * 80)

results_dir = Path(project_root) / "outputs" / "results"
clustering_path = results_dir / "clustering_results.csv"
association_path = results_dir / "association_rules_results.csv"

if not clustering_path.exists() or not association_path.exists():
    print(f"[!] Error: Results files not found in {results_dir}")
    print("    Please run '05_clustering_association_clean.py' first.")
    sys.exit(1)

df_cluster = pd.read_csv(clustering_path)
df_assoc = pd.read_csv(association_path)

print(f"[OK] Loaded Clustering Results: {len(df_cluster)} runs")
print(f"[OK] Loaded Association Results: {len(df_assoc)} runs")

#%% STEP 2: SELECT CLUSTERING CHAMPION
print("\n" + "=" * 80)
print("STEP 2: SELECT CLUSTERING CHAMPION")
print("=" * 80)

# Logic:
# 1. Filter out failed runs (Silhouette <= 0.1)
# 2. Sort by Silhouette Score (Descending - higher is better)

valid_clusters = df_cluster[df_cluster['silhouette_score'] > 0.1].copy()

# Sort by Silhouette (Descending)
valid_clusters.sort_values(by='silhouette_score', ascending=False, inplace=True)

if not valid_clusters.empty:
    champion_cluster = valid_clusters.iloc[0]

    print(f"\n[CHAMPION CLUSTERING MODEL]")
    print(f"  Algorithm:   {champion_cluster['algorithm']}")
    print(f"  Score:       {champion_cluster['silhouette_score']:.4f} (Silhouette)")

    print("\n  Configuration:")
    if champion_cluster['algorithm'] == 'KMeans':
        print(f"    n_clusters: {int(champion_cluster['n_clusters'])}")
        print(f"    init:       {champion_cluster['init']}")
        print(f"    n_init:     {int(champion_cluster['n_init'])}")
    elif champion_cluster['algorithm'] == 'DBSCAN':
        print(f"    eps:        {champion_cluster['eps']}")
        print(f"    min_samples:{int(champion_cluster['min_samples'])}")
        print(f"    metric:     {champion_cluster['metric']}")
    elif champion_cluster['algorithm'] == 'Hierarchical':
        print(f"    n_clusters: {int(champion_cluster['n_clusters'])}")
        print(f"    linkage:    {champion_cluster['linkage']}")
        print(f"    metric:     {champion_cluster['metric']}")
    elif champion_cluster['algorithm'] == 'GMM':
        print(f"    n_components: {int(champion_cluster['n_components'])}")
        print(f"    covariance:   {champion_cluster['covariance_type']}")
        print(f"    init_params:  {champion_cluster['init_params']}")

    # Log Champion Tag to MLflow
    print(f"\n[*] Tagging Champion in MLflow...")
    with mlflow.start_run(run_name="CHAMPION_CLUSTERING_SELECTION"):
        mlflow.log_param("selected_algorithm", champion_cluster['algorithm'])
        mlflow.log_metric("champion_silhouette", champion_cluster['silhouette_score'])
        mlflow.set_tag("model_status", "production_ready")
else:
    print("[!] No valid clustering models found.")
    champion_cluster = None

#%% STEP 3: SELECT ASSOCIATION RULES CHAMPION
print("\n" + "=" * 80)
print("STEP 3: SELECT MARKET BASKET CHAMPION")
print("=" * 80)

# Logic:
# 1. Must have at least 1 rule (column: association_rules)
# 2. Sort by avg_lift (Primary - descending)
# 3. Secondary: association_rules count (descending)

valid_assoc = df_assoc[df_assoc['association_rules'] > 0].copy()

# Sort by avg_lift (descending), then by association_rules (descending)
valid_assoc.sort_values(by=['avg_lift', 'association_rules'], ascending=[False, False], inplace=True)

if not valid_assoc.empty:
    champion_assoc = valid_assoc.iloc[0]

    print(f"\n[CHAMPION ASSOCIATION MODEL]")
    print(f"  Algorithm:   {champion_assoc['algorithm']}")
    print(f"  Avg Lift:    {champion_assoc['avg_lift']:.4f}")
    print(f"  Rules Found: {int(champion_assoc['association_rules'])}")
    print(f"  Avg Confidence: {champion_assoc['avg_confidence']:.4f}")
    print(f"  Avg Leverage:   {champion_assoc['avg_leverage']:.5f}")

    print("\n  Configuration:")
    print(f"    Min Support:    {champion_assoc['min_support']}")
    print(f"    Metric:         {champion_assoc['metric']}")
    print(f"    Min Threshold:  {champion_assoc['min_threshold']}")

    print(f"\n[*] Tagging Champion in MLflow...")
    with mlflow.start_run(run_name="CHAMPION_MARKET_BASKET_SELECTION"):
        mlflow.log_param("selected_algorithm", champion_assoc['algorithm'])
        mlflow.log_param("selected_support", champion_assoc['min_support'])
        mlflow.log_param("selected_metric", champion_assoc['metric'])
        mlflow.log_metric("champion_lift", champion_assoc['avg_lift'])
        mlflow.log_metric("champion_confidence", champion_assoc['avg_confidence'])
        mlflow.log_metric("champion_rules_count", champion_assoc['association_rules'])
        mlflow.set_tag("model_status", "production_ready")
else:
    print("[!] No valid association rules found.")
    champion_assoc = None

#%% STEP 4: GENERATE PRODUCTION CONFIG
print("\n" + "=" * 80)
print("STEP 4: EXPORT PRODUCTION CONFIG")
print("=" * 80)

# In a real system, this JSON/YAML file is what the dashboard reads
# to know which parameters to use for live data.

if champion_cluster is not None and champion_assoc is not None:
    production_config = {
        "clustering": {
            "algorithm": champion_cluster['algorithm'],
            "silhouette_score": float(champion_cluster['silhouette_score']),
            "params": {}
        },
        "association_rules": {
            "algorithm": champion_assoc['algorithm'],
            "avg_lift": float(champion_assoc['avg_lift']),
            "avg_confidence": float(champion_assoc['avg_confidence']),
            "rules_count": int(champion_assoc['association_rules']),
            "params": {
                "min_support": float(champion_assoc['min_support']),
                "metric": champion_assoc['metric'],
                "min_threshold": float(champion_assoc['min_threshold'])
            }
        }
    }

    # Add clustering params based on algorithm
    if champion_cluster['algorithm'] == 'KMeans':
        production_config["clustering"]["params"] = {
            "n_clusters": int(champion_cluster['n_clusters']),
            "init": champion_cluster['init'],
            "n_init": int(champion_cluster['n_init'])
        }
    elif champion_cluster['algorithm'] == 'DBSCAN':
        production_config["clustering"]["params"] = {
            "eps": float(champion_cluster['eps']),
            "min_samples": int(champion_cluster['min_samples']),
            "metric": champion_cluster['metric']
        }
    elif champion_cluster['algorithm'] == 'Hierarchical':
        production_config["clustering"]["params"] = {
            "n_clusters": int(champion_cluster['n_clusters']),
            "linkage": champion_cluster['linkage'],
            "metric": champion_cluster['metric']
        }
    elif champion_cluster['algorithm'] == 'GMM':
        production_config["clustering"]["params"] = {
            "n_components": int(champion_cluster['n_components']),
            "covariance_type": champion_cluster['covariance_type'],
            "init_params": champion_cluster['init_params']
        }

    import json
    config_path = results_dir / "production_config_unsupervised.json"
    with open(config_path, 'w') as f:
        json.dump(production_config, f, indent=4)

    print(f"[OK] Production configuration saved to:")
    print(f"     {config_path}")

    # Display the config
    print("\nProduction Configuration:")
    print(json.dumps(production_config, indent=2))
else:
    print("[!] Cannot generate production config - missing champion models")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("1. Scanned 48 hyperparameter runs (32 clustering + 16 association).")
print("2. Selected best Clustering model based on Silhouette Score (descending).")
print("3. Selected best Association rules based on avg_lift (descending).")
print("4. Saved parameters to 'production_config_unsupervised.json'.")
print("\nNext Steps:")
print("  - Use production_config_unsupervised.json for deployment")
print("  - Apply champion models to new data")
print("=" * 80)
