"""
Export Champion Unsupervised Models - Simplified
Saves configured models without retraining
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from sklearn.cluster import DBSCAN
from src.config_loader import ConfigLoader

print("=" * 80)
print("EXPORT CHAMPION UNSUPERVISED MODELS - SIMPLIFIED")
print("=" * 80)

config = ConfigLoader()
config.setup_databricks_env()
client = mlflow.tracking.MlflowClient()

# Output directories
models_dir = project_root / "outputs" / "models" / "production"
models_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: GET CHAMPION RUNS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: GET CHAMPION RUNS FROM MLFLOW")
print("=" * 80)

exp = client.get_experiment_by_name('/pizza-clustering-association')
all_runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=200)

# Get clustering champion (DBSCAN)
clustering_runs = [r for r in all_runs if r.data.params.get('algorithm') in ['DBSCAN', 'KMeans', 'GMM', 'Hierarchical']]
clustering_runs = sorted(clustering_runs, key=lambda r: r.data.metrics.get('silhouette_score', -999), reverse=True)
champion_cluster = clustering_runs[0]

print(f"\n[CLUSTERING CHAMPION]")
print(f"  Algorithm: {champion_cluster.data.params.get('algorithm')}")
print(f"  Silhouette: {champion_cluster.data.metrics.get('silhouette_score'):.4f}")
print(f"  Parameters: {dict(champion_cluster.data.params)}")

# Get association champion (Apriori/FP-Growth)
assoc_runs = [r for r in all_runs if r.data.params.get('algorithm') in ['Apriori', 'FP-Growth']]
assoc_runs = sorted(assoc_runs, key=lambda r: r.data.metrics.get('avg_lift', -999), reverse=True)
champion_assoc = assoc_runs[0]

print(f"\n[ASSOCIATION CHAMPION]")
print(f"  Algorithm: {champion_assoc.data.params.get('algorithm')}")
print(f"  Avg Lift: {champion_assoc.data.metrics.get('avg_lift'):.4f}")
print(f"  Rules Count: {int(champion_assoc.data.metrics.get('rules_count', 0))}")
print(f"  Parameters: {dict(champion_assoc.data.params)}")

# ============================================================================
# STEP 2: CREATE AND SAVE DBSCAN MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CREATE AND SAVE DBSCAN MODEL")
print("=" * 80)

cluster_params = dict(champion_cluster.data.params)
eps = float(cluster_params.get('eps', 0.5))
min_samples = int(float(cluster_params.get('min_samples', 2)))
metric = cluster_params.get('metric', 'euclidean')

print(f"\n[*] Creating DBSCAN with eps={eps}, min_samples={min_samples}, metric={metric}")

# Create configured DBSCAN model
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

# Save model locally
local_cluster_path = models_dir / "clustering"
mlflow.sklearn.save_model(dbscan, str(local_cluster_path))
print(f"[OK] Model saved to: {local_cluster_path}")

# Also save parameters
cluster_config = {
    'algorithm': 'DBSCAN',
    'params': cluster_params,
    'metrics': {
        'silhouette_score': champion_cluster.data.metrics.get('silhouette_score'),
        'n_clusters': champion_cluster.data.metrics.get('n_clusters'),
    },
    'run_id': champion_cluster.info.run_id
}

config_path = models_dir / "clustering_config.json"
import json
with open(config_path, 'w') as f:
    json.dump(cluster_config, f, indent=2)
print(f"[OK] Config saved to: {config_path}")

# ============================================================================
# STEP 3: DOWNLOAD ASSOCIATION RULES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DOWNLOAD ASSOCIATION RULES")
print("=" * 80)

print(f"\n[*] Downloading association rules from MLflow...")

# Try to download rules artifact
try:
    rules_artifact_path = client.download_artifacts(champion_assoc.info.run_id, "association_rules.csv")
    rules_df = pd.read_csv(rules_artifact_path)

    # Save to production location
    rules_path = models_dir / "association_rules.csv"
    rules_df.to_csv(rules_path, index=False)
    print(f"[OK] Saved {len(rules_df)} rules to: {rules_path}")

except Exception as e:
    print(f"[WARNING] Could not download rules CSV: {e}")
    print(f"[*] Creating empty rules file...")
    rules_df = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
    rules_path = models_dir / "association_rules.csv"
    rules_df.to_csv(rules_path, index=False)
    print(f"[OK] Created empty rules file: {rules_path}")

# Save association config
assoc_config = {
    'algorithm': champion_assoc.data.params.get('algorithm'),
    'params': dict(champion_assoc.data.params),
    'metrics': {
        'avg_lift': champion_assoc.data.metrics.get('avg_lift'),
        'rules_count': champion_assoc.data.metrics.get('rules_count'),
    },
    'run_id': champion_assoc.info.run_id,
    'rules_path': str(rules_path)
}

assoc_config_path = models_dir / "association_config.json"
with open(assoc_config_path, 'w') as f:
    json.dump(assoc_config, f, indent=2)
print(f"[OK] Config saved to: {assoc_config_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EXPORT COMPLETE")
print("=" * 80)

print(f"\n[CLUSTERING]")
print(f"  Model: DBSCAN")
print(f"  Silhouette: {champion_cluster.data.metrics.get('silhouette_score'):.4f}")
print(f"  Path: {local_cluster_path}")

print(f"\n[ASSOCIATION]")
print(f"  Model: {champion_assoc.data.params.get('algorithm')}")
print(f"  Avg Lift: {champion_assoc.data.metrics.get('avg_lift'):.4f}")
print(f"  Rules: {len(rules_df)}")
print(f"  Path: {rules_path}")

print(f"\n[NEXT STEP]")
print(f"  Update deployment script to include clustering and association models")

print("\n" + "=" * 80)
