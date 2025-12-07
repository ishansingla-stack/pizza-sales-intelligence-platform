"""
Deploy Phase 1 Models - Demand, Clustering, Association
Exports champion models from MLflow for deployment
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import mlflow
import mlflow.sklearn
import pandas as pd
import json
from src.config_loader import ConfigLoader

print("=" * 80)
print("PHASE 1 DEPLOYMENT - EXPORT CHAMPION MODELS")
print("=" * 80)

config = ConfigLoader()
config.setup_databricks_env()
client = mlflow.tracking.MlflowClient()

# Output directories
models_dir = project_root / "outputs" / "models" / "production"
results_dir = project_root / "outputs" / "results" / "ml_model_tracking"
models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

production_config = {}

# ============================================================================
# 1. DEMAND FORECASTING CHAMPION
# ============================================================================
print("\n" + "=" * 80)
print("1. DEMAND FORECASTING")
print("=" * 80)

exp_demand = client.get_experiment_by_name('/pizza-demand-forecasting')
runs_demand = client.search_runs(
    experiment_ids=[exp_demand.experiment_id],
    max_results=100,
    order_by=["metrics.test_r2 DESC"]
)

print(f"\n[*] Found {len(runs_demand)} demand forecasting runs")

# Get champion (best test R²)
champion_demand = runs_demand[0]
demand_metrics = {
    'test_r2': champion_demand.data.metrics.get('test_r2'),
    'test_rmse': champion_demand.data.metrics.get('test_rmse'),
    'test_mae': champion_demand.data.metrics.get('test_mae'),
    'val_r2': champion_demand.data.metrics.get('val_r2'),
}

print(f"\n[CHAMPION] {champion_demand.info.run_name}")
print(f"  Test R²: {demand_metrics['test_r2']:.4f}")
print(f"  Test RMSE: {demand_metrics['test_rmse']:.2f} pizzas/hour")
print(f"  Test MAE: {demand_metrics['test_mae']:.2f} pizzas/hour")

# Download model
print(f"\n[*] Downloading model from MLflow...")
model_uri = f"runs:/{champion_demand.info.run_id}/model"
model_path = models_dir / "demand_forecasting"
mlflow.sklearn.save_model(
    mlflow.sklearn.load_model(model_uri),
    str(model_path)
)
print(f"[OK] Saved to: {model_path}")

production_config['demand_forecasting'] = {
    'model_path': str(model_path),
    'run_id': champion_demand.info.run_id,
    'run_name': champion_demand.info.run_name,
    'metrics': demand_metrics,
    'params': dict(champion_demand.data.params)
}

# ============================================================================
# 2. CLUSTERING CHAMPION
# ============================================================================
print("\n" + "=" * 80)
print("2. CUSTOMER SEGMENTATION (CLUSTERING)")
print("=" * 80)

exp_cluster = client.get_experiment_by_name('/pizza-clustering-association')
all_runs_cluster = client.search_runs(
    experiment_ids=[exp_cluster.experiment_id],
    max_results=200
)

# Filter clustering runs by algorithm (KMeans, DBSCAN, GMM, Hierarchical)
clustering_algos = ['KMeans', 'DBSCAN', 'GMM', 'Hierarchical']
runs_cluster = [r for r in all_runs_cluster if r.data.params.get('algorithm') in clustering_algos]
runs_cluster = sorted(runs_cluster, key=lambda r: r.data.metrics.get('silhouette_score', -999), reverse=True)

print(f"\n[*] Found {len(runs_cluster)} clustering runs")

# Get champion (best silhouette score)
champion_cluster = runs_cluster[0]
cluster_metrics = {
    'silhouette_score': champion_cluster.data.metrics.get('silhouette_score'),
    'n_clusters': champion_cluster.data.metrics.get('n_clusters'),
    'davies_bouldin': champion_cluster.data.metrics.get('davies_bouldin'),
}

print(f"\n[CHAMPION] {champion_cluster.info.run_name}")
print(f"  Silhouette Score: {cluster_metrics['silhouette_score']:.4f}")
print(f"  Number of Clusters: {int(cluster_metrics['n_clusters']) if cluster_metrics['n_clusters'] else 'N/A'}")

# Download model
print(f"\n[*] Downloading model from MLflow...")
model_uri = f"runs:/{champion_cluster.info.run_id}/model"
model_path = models_dir / "clustering"
mlflow.sklearn.save_model(
    mlflow.sklearn.load_model(model_uri),
    str(model_path)
)
print(f"[OK] Saved to: {model_path}")

production_config['clustering'] = {
    'model_path': str(model_path),
    'run_id': champion_cluster.info.run_id,
    'run_name': champion_cluster.info.run_name,
    'metrics': cluster_metrics,
    'params': dict(champion_cluster.data.params)
}

# ============================================================================
# 3. ASSOCIATION RULES CHAMPION
# ============================================================================
print("\n" + "=" * 80)
print("3. PRODUCT RECOMMENDATIONS (ASSOCIATION RULES)")
print("=" * 80)

# Filter association runs by algorithm (Apriori, FP-Growth)
association_algos = ['Apriori', 'FP-Growth']
runs_assoc = [r for r in all_runs_cluster if r.data.params.get('algorithm') in association_algos]
runs_assoc = sorted(runs_assoc, key=lambda r: r.data.metrics.get('avg_lift', -999), reverse=True)

print(f"\n[*] Found {len(runs_assoc)} association runs")

# Get champion (best avg lift)
champion_assoc = runs_assoc[0]
assoc_metrics = {
    'avg_lift': champion_assoc.data.metrics.get('avg_lift'),
    'rules_count': champion_assoc.data.metrics.get('rules_count'),
    'avg_confidence': champion_assoc.data.metrics.get('avg_confidence'),
}

print(f"\n[CHAMPION] {champion_assoc.info.run_name}")
print(f"  Average Lift: {assoc_metrics['avg_lift']:.4f}")
print(f"  Rules Count: {int(assoc_metrics['rules_count']) if assoc_metrics['rules_count'] else 'N/A'}")

# Download rules artifact
print(f"\n[*] Downloading association rules from MLflow...")
rules_artifact_path = client.download_artifacts(champion_assoc.info.run_id, "association_rules.csv")
rules_df = pd.read_csv(rules_artifact_path)

# Save to production location
rules_path = models_dir / "association_rules.csv"
rules_df.to_csv(rules_path, index=False)
print(f"[OK] Saved {len(rules_df)} rules to: {rules_path}")

production_config['association_rules'] = {
    'rules_path': str(rules_path),
    'run_id': champion_assoc.info.run_id,
    'run_name': champion_assoc.info.run_name,
    'metrics': assoc_metrics,
    'params': dict(champion_assoc.data.params)
}

# ============================================================================
# 4. SAVE PRODUCTION CONFIG
# ============================================================================
print("\n" + "=" * 80)
print("4. SAVE PRODUCTION CONFIGURATION")
print("=" * 80)

config_path = results_dir / "production_config.json"
with open(config_path, 'w') as f:
    json.dump(production_config, f, indent=2)

print(f"\n[OK] Production config saved to: {config_path}")

# ============================================================================
# DEPLOYMENT SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1 DEPLOYMENT COMPLETE")
print("=" * 80)

print("\n[DEPLOYED MODELS]")
print(f"  1. Demand Forecasting: {production_config['demand_forecasting']['run_name']}")
print(f"     Test R²: {production_config['demand_forecasting']['metrics']['test_r2']:.4f}")

print(f"\n  2. Customer Segmentation: {production_config['clustering']['run_name']}")
print(f"     Silhouette: {production_config['clustering']['metrics']['silhouette_score']:.4f}")

print(f"\n  3. Product Recommendations: {production_config['association_rules']['run_name']}")
print(f"     Avg Lift: {production_config['association_rules']['metrics']['avg_lift']:.4f}")

print("\n[NEXT STEPS]")
print("  - Models exported to: outputs/models/production/")
print("  - Update Streamlit app to load these models")
print("  - Revenue prediction will be added in Phase 2")

print("\n" + "=" * 80)
