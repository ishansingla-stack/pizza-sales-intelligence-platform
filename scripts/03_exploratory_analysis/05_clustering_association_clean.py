"""
Pizza Intelligence - Clustering & Association Rules (Final Optimized)
Requirements Met:
 1. NO DATA LEAKAGE (Product attributes only)
 2. HYPERPARAMETER TUNING (3 Params x 2 Values per model)
 3. OPTIMIZED LOGIC (Efficient nested loops)
"""

import sys
import os
from pathlib import Path
import itertools
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import mlflow
import warnings
warnings.filterwarnings('ignore')

# Monkey-patch MLflow to prevent emoji output errors on Windows
import mlflow.tracking._tracking_service.client
mlflow.tracking._tracking_service.client.TrackingServiceClient._log_url = lambda self, run_id: None

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Association rules
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.config_loader import ConfigLoader

print("=" * 80)
print("PIZZA INTELLIGENCE - CLUSTERING & ASSOCIATION (FINAL)")
print("=" * 80)

# Load configuration
print("\n[*] Loading configuration...")
config = ConfigLoader()
config.setup_databricks_env()

# Set MLflow experiment
mlflow.set_experiment("/pizza-clustering-association")
print(f"[OK] MLflow experiment: /pizza-clustering-association")

#%% PART 1: CLUSTERING ANALYSIS (NO LEAKAGE)
print("\n" + "=" * 80)
print("PART 1: CLUSTERING - FEATURE ENGINEERING")
print("=" * 80)

# Load raw data
data_path = config.get_local_data_path()
if not os.path.exists(data_path):
    print(f"[!] Data file not found: {data_path}")
    sys.exit(1)

df = pd.read_excel(data_path, sheet_name='pizza_sales')
print(f"[OK] Loaded {len(df):,} records")

# Aggregate by pizza name - ONLY LEGITIMATE ATTRIBUTES
print("\n[*] Creating Product Identity Features (NO LEAKAGE)...")
pizza_summary = df.groupby('pizza_name').agg({
    'unit_price': 'mean',            # Intrinsic
    'pizza_size': lambda x: x.mode()[0],
    'pizza_category': lambda x: x.mode()[0],
    'pizza_ingredients': 'first'
}).reset_index()

# Feature Engineering
pizza_summary['ingredient_count'] = pizza_summary['pizza_ingredients'].str.split(',').str.len()

# Encode categorical variables
size_encoder = LabelEncoder()
category_encoder = LabelEncoder()

pizza_summary['size_encoded'] = size_encoder.fit_transform(pizza_summary['pizza_size'])
pizza_summary['category_encoded'] = category_encoder.fit_transform(pizza_summary['pizza_category'])

# STRICTLY Product Features (No Sales Data)
cluster_features = ['unit_price', 'ingredient_count', 'size_encoded', 'category_encoded']
X_cluster = pizza_summary[cluster_features].copy()

# Scale features
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

print(f"[OK] Scaled Features: {X_cluster_scaled.shape}")
print(f"Features used: {cluster_features}")

#%% HELPER: Grid Search Function
# Global list to collect all clustering results
all_clustering_results = []

def run_clustering_grid_search(model_name, model_class, param_grid, data):
    """
    Runs grid search for clustering models enforcing 3x2 parameter grid.
    """
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\n--- Tuning {model_name} ({len(combinations)} configs) ---")

    results = []

    for i, params in enumerate(combinations):
        start_time = time.time()
        try:
            # Instantiate model
            if model_name == "DBSCAN":
                model = model_class(**params)
                labels = model.fit_predict(data)
            elif model_name == "GMM":
                model = model_class(**params, random_state=42)
                labels = model.fit_predict(data)
            elif model_name == "Hierarchical":
                model = model_class(**params)  # No random_state for AgglomerativeClustering
                labels = model.fit_predict(data)
            else:  # KMeans
                model = model_class(**params, random_state=42)
                labels = model.fit_predict(data)

            # Validation: Need >1 cluster and no "all noise"
            unique_labels = set(labels) - {-1}
            if len(unique_labels) < 2:
                silhouette = -1
                db_score = -1
            else:
                # Handle noise for DBSCAN metrics
                mask = labels != -1
                if mask.sum() > 1:
                    silhouette = silhouette_score(data[mask], labels[mask])
                    db_score = davies_bouldin_score(data[mask], labels[mask])
                else:
                    silhouette = -1
                    db_score = -1

            training_time = time.time() - start_time

            # Print status
            param_str = ", ".join([f"{k}={v}" for k,v in params.items()])
            print(f" Run {i+1}: {param_str} | Silhouette={silhouette:.3f}")

            # Log to MLflow
            with mlflow.start_run(run_name=f"{model_name}_run_{i+1}"):
                mlflow.log_params(params)
                mlflow.log_param("algorithm", model_name)
                mlflow.log_metric("silhouette_score", silhouette)
                mlflow.log_metric("davies_bouldin_score", db_score)

            # Store result with algorithm name
            result_row = {
                'algorithm': model_name,
                **params,
                'silhouette_score': silhouette,
                'davies_bouldin_score': db_score,
                'training_time': training_time,
                'labels': labels
            }
            results.append(result_row)

            # Add to global results (without labels)
            global all_clustering_results
            all_clustering_results.append({k: v for k, v in result_row.items() if k != 'labels'})

        except Exception as e:
            print(f" Run {i+1} Failed: {e}")

    # Return best result
    if not results:
        return None, -1, None

    best_run = max(results, key=lambda x: x['silhouette_score'])
    print(f" >> BEST {model_name}: Silhouette={best_run['silhouette_score']:.3f}")
    return best_run, best_run['silhouette_score'], best_run['labels']

#%% 1. K-MEANS (3 Params x 2 Values)
print("\n" + "=" * 80)
print("1. K-MEANS TUNING")
kmeans_grid = {
    'n_clusters': [3, 5],            # Param 1
    'init': ['k-means++', 'random'], # Param 2
    'n_init': [10, 20]               # Param 3
}
best_km_params, score_km, labels_km = run_clustering_grid_search("KMeans", KMeans, kmeans_grid, X_cluster_scaled)
pizza_summary['kmeans_cluster'] = labels_km

#%% 2. DBSCAN (3 Params x 2 Values)
print("\n" + "=" * 80)
print("2. DBSCAN TUNING")
dbscan_grid = {
    'eps': [0.5, 1.0],                 # Param 1
    'min_samples': [2, 4],             # Param 2
    'metric': ['euclidean', 'manhattan'] # Param 3
}
best_db_params, score_db, labels_db = run_clustering_grid_search("DBSCAN", DBSCAN, dbscan_grid, X_cluster_scaled)
if labels_db is not None:
    pizza_summary['dbscan_cluster'] = labels_db

#%% 3. HIERARCHICAL (3 Params x 2 Values)
print("\n" + "=" * 80)
print("3. HIERARCHICAL TUNING")
# Note: 'ward' only works with euclidean, so we use average/complete to allow metric tuning
hier_grid = {
    'n_clusters': [3, 5],               # Param 1
    'linkage': ['average', 'complete'], # Param 2
    'metric': ['euclidean', 'manhattan']# Param 3
}
best_hc_params, score_hc, labels_hc = run_clustering_grid_search("Hierarchical", AgglomerativeClustering, hier_grid, X_cluster_scaled)
pizza_summary['hier_cluster'] = labels_hc

#%% 4. GMM (3 Params x 2 Values)
print("\n" + "=" * 80)
print("4. GMM TUNING")
gmm_grid = {
    'n_components': [3, 5],               # Param 1
    'covariance_type': ['full', 'tied'],  # Param 2
    'init_params': ['kmeans', 'random']   # Param 3
}
best_gmm_params, score_gmm, labels_gmm = run_clustering_grid_search("GMM", GaussianMixture, gmm_grid, X_cluster_scaled)
pizza_summary['gmm_cluster'] = labels_gmm

#%% PART 2: ASSOCIATION RULES (OPTIMIZED GRID SEARCH)
print("\n" + "=" * 80)
print("PART 2: MARKET BASKET ANALYSIS (3 Params x 2 Values)")
print("=" * 80)

# Transaction Encoding
transactions = df.groupby('order_id')['pizza_name'].apply(list).values.tolist()
te = TransactionEncoder()
basket_df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

# Association Rules Hyperparameters
assoc_params = {
    'min_support': [0.005, 0.01],     # Param 1
    'metric': ['lift', 'confidence'], # Param 2
    'min_threshold': [0.3, 0.5]       # Param 3
}

print(f"Hyperparameters: {assoc_params}")

# Global list to collect association results
all_association_results = []

# Run for both Algorithms
for algo_name, algo_func in [('Apriori', apriori), ('FP-Growth', fpgrowth)]:
    print(f"\n--- Tuning {algo_name} ---")

    # 1. OUTER LOOP: Support (Expensive Itemset Generation)
    for support in assoc_params['min_support']:
        print(f" [Calculating Itemsets] Support={support}...")

        # Calculate Itemsets ONCE per support level
        frequent_itemsets = algo_func(basket_df, min_support=support, use_colnames=True)

        if len(frequent_itemsets) == 0:
            print("  No itemsets found. Skipping...")
            continue

        # 2. INNER LOOPS: Rules Generation (Fast)
        for metric in assoc_params['metric']:
            for threshold in assoc_params['min_threshold']:

                with mlflow.start_run(run_name=f"{algo_name}_sup{support}_{metric}_{threshold}"):
                    # Generate rules
                    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=threshold)

                    if len(rules) > 0:
                        # Calculate Leverage
                        rules['leverage'] = rules['support'] - (rules['antecedent support'] * rules['consequent support'])
                        avg_lift = rules['lift'].mean()
                        avg_confidence = rules['confidence'].mean()
                        avg_leverage = rules['leverage'].mean()
                        rule_count = len(rules)
                    else:
                        avg_lift = 0
                        avg_confidence = 0
                        avg_leverage = 0
                        rule_count = 0

                    # Log
                    mlflow.log_param("algorithm", algo_name)
                    mlflow.log_param("min_support", support)
                    mlflow.log_param("metric", metric)
                    mlflow.log_param("min_threshold", threshold)

                    mlflow.log_metric("association_rules", rule_count)
                    mlflow.log_metric("avg_lift", avg_lift)
                    mlflow.log_metric("avg_confidence", avg_confidence)
                    mlflow.log_metric("avg_leverage", avg_leverage)

                    print(f"  {metric}={threshold}: {rule_count} rules found.")

                    # Collect results
                    all_association_results.append({
                        'algorithm': algo_name,
                        'min_support': support,
                        'metric': metric,
                        'min_threshold': threshold,
                        'association_rules': rule_count,
                        'avg_lift': avg_lift,
                        'avg_confidence': avg_confidence,
                        'avg_leverage': avg_leverage
                    })

#%% FINAL SAVING
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_dir = Path(project_root) / "outputs" / "results"
output_dir.mkdir(parents=True, exist_ok=True)

# Save Hyperparameter Tuning Results to CSV
clustering_df = pd.DataFrame(all_clustering_results)
association_df = pd.DataFrame(all_association_results)

clustering_df.to_csv(output_dir / "clustering_results.csv", index=False)
association_df.to_csv(output_dir / "association_rules_results.csv", index=False)

print(f"[OK] Saved clustering results: {output_dir / 'clustering_results.csv'}")
print(f"[OK] Saved association results: {output_dir / 'association_rules_results.csv'}")

# Save Clusters
pizza_summary.to_csv(output_dir / "pizza_clusters_final.csv", index=False)
print(f"[OK] Saved clusters: {output_dir / 'pizza_clusters_final.csv'}")

print("\n[COMPLETE] All models tuned and logged.")
print(f"View results: {config.get_databricks_host()}/#mlflow/experiments")
