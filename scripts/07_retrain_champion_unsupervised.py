"""
Retrain Champion Unsupervised Models - Clustering and Association
Retrains DBSCAN and Apriori with champion parameters and saves models properly
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
from sklearn.cluster import DBSCAN
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from src.config_loader import ConfigLoader

print("=" * 80)
print("RETRAIN CHAMPION UNSUPERVISED MODELS")
print("=" * 80)

config = ConfigLoader()
config.setup_databricks_env()
client = mlflow.tracking.MlflowClient()

# Output directories
models_dir = project_root / "outputs" / "models" / "production"
models_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: GET CHAMPION PARAMETERS FROM MLFLOW
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: GET CHAMPION PARAMETERS")
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

cluster_params = dict(champion_cluster.data.params)

# Get association champion (Apriori/FP-Growth)
assoc_runs = [r for r in all_runs if r.data.params.get('algorithm') in ['Apriori', 'FP-Growth']]
assoc_runs = sorted(assoc_runs, key=lambda r: r.data.metrics.get('avg_lift', -999), reverse=True)
champion_assoc = assoc_runs[0]

print(f"\n[ASSOCIATION CHAMPION]")
print(f"  Algorithm: {champion_assoc.data.params.get('algorithm')}")
print(f"  Avg Lift: {champion_assoc.data.metrics.get('avg_lift'):.4f}")
print(f"  Rules Count: {int(champion_assoc.data.metrics.get('rules_count', 0))}")
print(f"  Parameters: {dict(champion_assoc.data.params)}")

assoc_params = dict(champion_assoc.data.params)

# ============================================================================
# STEP 2: LOAD PROCESSED DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: LOAD PROCESSED DATA")
print("=" * 80)

# Load clustering data
clustering_data_path = project_root / "data" / "processed" / "clustering" / "customer_features.parquet"
print(f"\n[*] Loading clustering data from: {clustering_data_path}")
customer_features = pd.read_parquet(clustering_data_path)
print(f"[OK] Loaded {len(customer_features)} customers with {customer_features.shape[1]} features")

# Load association data (transactions)
basket_path = project_root / "data" / "processed" / "association" / "basket_data.parquet"
print(f"\n[*] Loading association data from: {basket_path}")
basket_df = pd.read_parquet(basket_path)
print(f"[OK] Loaded {len(basket_df)} transactions")

# ============================================================================
# STEP 3: RETRAIN CLUSTERING MODEL (DBSCAN)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: RETRAIN CLUSTERING MODEL (DBSCAN)")
print("=" * 80)

mlflow.set_experiment("/pizza-clustering-association")

with mlflow.start_run(run_name="CHAMPION_DBSCAN_RETRAIN"):
    # Create DBSCAN model
    eps = float(cluster_params.get('eps', 0.5))
    min_samples = int(float(cluster_params.get('min_samples', 2)))
    metric = cluster_params.get('metric', 'euclidean')

    print(f"\n[*] Training DBSCAN with eps={eps}, min_samples={min_samples}, metric={metric}")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    clusters = dbscan.fit_predict(customer_features)

    # Calculate metrics
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    if n_clusters > 1 and len(set(clusters)) > 1:
        # Filter out noise points for silhouette score
        mask = clusters != -1
        if mask.sum() > 0:
            silhouette = silhouette_score(customer_features[mask], clusters[mask])
            davies_bouldin = davies_bouldin_score(customer_features[mask], clusters[mask])
        else:
            silhouette = -1
            davies_bouldin = 999
    else:
        silhouette = -1
        davies_bouldin = 999

    print(f"[OK] Training complete")
    print(f"     Clusters: {n_clusters}")
    print(f"     Noise points: {n_noise}")
    print(f"     Silhouette: {silhouette:.4f}")

    # Log parameters and metrics
    mlflow.log_params({
        'algorithm': 'DBSCAN',
        'eps': eps,
        'min_samples': min_samples,
        'metric': metric
    })

    mlflow.log_metrics({
        'silhouette_score': silhouette,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'davies_bouldin': davies_bouldin
    })

    # Save model
    mlflow.sklearn.log_model(dbscan, "model")

    # Also save locally
    local_cluster_path = models_dir / "clustering"
    mlflow.sklearn.save_model(dbscan, str(local_cluster_path))
    print(f"\n[OK] Model saved to: {local_cluster_path}")

    cluster_run_id = mlflow.active_run().info.run_id

# ============================================================================
# STEP 4: RETRAIN ASSOCIATION RULES (APRIORI)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: RETRAIN ASSOCIATION RULES (APRIORI)")
print("=" * 80)

with mlflow.start_run(run_name="CHAMPION_APRIORI_RETRAIN"):
    # Prepare transactions (convert to basket format for mlxtend)
    print(f"\n[*] Preparing transactions...")

    # Pivot to basket format (one-hot encoding)
    basket = basket_df.groupby(['order_id', 'pizza_name_id'])['pizza_name_id'].count().unstack(fill_value=0)
    basket = (basket > 0).astype(int)

    print(f"[OK] Prepared basket with {len(basket)} transactions and {len(basket.columns)} items")

    # Run Apriori
    min_support = float(assoc_params.get('min_support', 0.001))
    min_threshold = float(assoc_params.get('min_threshold', 0.0))
    metric_type = assoc_params.get('metric', 'lift')

    print(f"\n[*] Running Apriori with min_support={min_support}, metric={metric_type}")

    # Get frequent itemsets
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

    # Generate association rules
    if len(frequent_itemsets) > 0:
        rules_df = association_rules(frequent_itemsets, metric=metric_type, min_threshold=min_threshold)

        # Format antecedents and consequents as comma-separated strings
        rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ','.join(list(x)))
        rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ','.join(list(x)))
    else:
        rules_df = pd.DataFrame()

    print(f"[OK] Generated {len(rules_df)} association rules")

    # Calculate metrics
    if len(rules_df) > 0:
        avg_lift = rules_df['lift'].mean()
        avg_confidence = rules_df['confidence'].mean()
        max_lift = rules_df['lift'].max()
    else:
        avg_lift = 0
        avg_confidence = 0
        max_lift = 0

    print(f"     Avg Lift: {avg_lift:.4f}")
    print(f"     Avg Confidence: {avg_confidence:.4f}")

    # Log parameters and metrics
    mlflow.log_params({
        'algorithm': 'Apriori',
        'min_support': min_support,
        'min_threshold': min_threshold,
        'metric': metric_type
    })

    mlflow.log_metrics({
        'rules_count': len(rules_df),
        'avg_lift': avg_lift,
        'avg_confidence': avg_confidence,
        'max_lift': max_lift
    })

    # Save rules as artifact
    rules_path = models_dir / "association_rules.csv"
    rules_df.to_csv(rules_path, index=False)
    mlflow.log_artifact(str(rules_path), "association_rules.csv")

    print(f"\n[OK] Rules saved to: {rules_path}")

    # Save frequent itemsets and rules as pickle for reconstruction
    assoc_model = {
        'frequent_itemsets': frequent_itemsets,
        'rules_df': rules_df,
        'params': assoc_params
    }

    pickle_path = models_dir / "association_model.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(assoc_model, f)

    mlflow.log_artifact(str(pickle_path), "association_model.pkl")
    print(f"[OK] Model saved to: {pickle_path}")

    assoc_run_id = mlflow.active_run().info.run_id

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("RETRAINING COMPLETE")
print("=" * 80)

print(f"\n[CLUSTERING]")
print(f"  Model: DBSCAN")
print(f"  Silhouette: {silhouette:.4f}")
print(f"  Clusters: {n_clusters}")
print(f"  Run ID: {cluster_run_id}")
print(f"  Local Path: {local_cluster_path}")

print(f"\n[ASSOCIATION]")
print(f"  Model: Apriori")
print(f"  Rules: {len(rules_df)}")
print(f"  Avg Lift: {avg_lift:.4f}")
print(f"  Run ID: {assoc_run_id}")
print(f"  Local Path: {rules_path}")

print(f"\n[NEXT STEP]")
print(f"  Run deployment script to export all 3 models")

print("\n" + "=" * 80)
