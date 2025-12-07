"""
Generate Association Rules for Production - Standalone Script
Reads raw Excel data and generates association rules without needing processed files
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from src.config_loader import ConfigLoader

print("=" * 80)
print("GENERATE ASSOCIATION RULES FOR PRODUCTION (FP-GROWTH)")
print("=" * 80)

# Setup Databricks
config = ConfigLoader()
config.setup_databricks_env()

# Set MLflow experiment
experiment_name = "/Shared/pizza_intelligence_association_rules"
mlflow.set_experiment(experiment_name)
print(f"[OK] MLflow experiment set to: {experiment_name}")

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOAD RAW DATA")
print("=" * 80)

data_path = project_root / "data" / "raw" / "Data_Model_-_Pizza_Sales.xlsx"
print(f"\n[*] Loading data from: {data_path}")

# Read pizza sales data (single sheet with all transaction data)
transactions = pd.read_excel(data_path, sheet_name='pizza_sales')

print(f"[OK] Loaded {len(transactions)} transaction records")
print(f"[OK] Columns: {list(transactions.columns)}")

# ============================================================================
# STEP 2: PREPARE TRANSACTION DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: PREPARE TRANSACTION DATA")
print("=" * 80)

# We already have order_id and pizza_name - perfect for association rules!
print(f"[*] Using columns: order_id, pizza_name")
print(f"[*] Unique orders: {transactions['order_id'].nunique()}")
print(f"[*] Unique pizzas: {transactions['pizza_name'].nunique()}")

# Create basket format (pivot to one-hot encoding)
print(f"\n[*] Creating basket format...")
basket = transactions.groupby(['order_id', 'pizza_name'])['quantity'].sum().unstack(fill_value=0)
basket = (basket > 0).astype(int)

print(f"[OK] Created basket with {len(basket)} orders and {len(basket.columns)} unique pizzas")

# ============================================================================
# STEP 3: GENERATE ASSOCIATION RULES - CHAMPION PARAMETERS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: GENERATE ASSOCIATION RULES (FP-GROWTH)")
print("=" * 80)

# Champion parameters from previous runs
min_support = 0.005  # 0.5%
min_threshold = 0.5
metric_type = 'lift'

print(f"\n[*] Running Apriori with:")
print(f"    - min_support: {min_support}")
print(f"    - metric: {metric_type}")
print(f"    - min_threshold: {min_threshold}")

# Start MLflow run
with mlflow.start_run(run_name="PRODUCTION_ASSOCIATION_RULES"):
    # Log parameters
    mlflow.log_param("algorithm", "FP-Growth")
    mlflow.log_param("min_support", min_support)
    mlflow.log_param("metric", metric_type)
    mlflow.log_param("min_threshold", min_threshold)

    # Get frequent itemsets using FP-Growth (faster than Apriori)
    print(f"\n[*] Finding frequent itemsets using FP-Growth...")
    frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    print(f"[OK] Found {len(frequent_itemsets)} frequent itemsets")

    # Generate association rules
    print(f"\n[*] Generating association rules...")
    if len(frequent_itemsets) > 0:
        rules_df = association_rules(
            frequent_itemsets,
            metric=metric_type,
            min_threshold=min_threshold
        )

        # Format antecedents and consequents as comma-separated strings
        rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ','.join(list(x)))
        rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ','.join(list(x)))

        print(f"[OK] Generated {len(rules_df)} association rules")
    else:
        rules_df = pd.DataFrame()
        print(f"[WARNING] No frequent itemsets found!")

    # Calculate and log metrics
    if len(rules_df) > 0:
        avg_lift = rules_df['lift'].mean()
        avg_confidence = rules_df['confidence'].mean()
        avg_leverage = rules_df['leverage'].mean()
        max_lift = rules_df['lift'].max()
        max_confidence = rules_df['confidence'].max()

        mlflow.log_metric("association_rules", len(rules_df))
        mlflow.log_metric("avg_lift", avg_lift)
        mlflow.log_metric("avg_confidence", avg_confidence)
        mlflow.log_metric("avg_leverage", avg_leverage)
        mlflow.log_metric("max_lift", max_lift)
        mlflow.log_metric("max_confidence", max_confidence)

        print(f"\n[METRICS]")
        print(f"  Rules: {len(rules_df)}")
        print(f"  Avg Lift: {avg_lift:.4f}")
        print(f"  Avg Confidence: {avg_confidence:.4f}")
        print(f"  Max Lift: {max_lift:.4f}")
    else:
        mlflow.log_metric("association_rules", 0)
        mlflow.log_metric("avg_lift", 0.0)
        mlflow.log_metric("avg_confidence", 0.0)
        mlflow.log_metric("avg_leverage", 0.0)

    # ========================================================================
    # STEP 4: SAVE TO PRODUCTION FOLDER
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: SAVE TO PRODUCTION FOLDER")
    print("=" * 80)

    # Create production directory
    models_dir = project_root / "outputs" / "models" / "production"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save rules CSV
    rules_path = models_dir / "association_rules.csv"
    rules_df.to_csv(rules_path, index=False)
    print(f"\n[OK] Saved rules CSV to: {rules_path}")

    # Log as MLflow artifact
    mlflow.log_artifact(str(rules_path), "association_rules")
    print(f"[OK] Logged to MLflow artifacts")

    # Save model as pickle (for programmatic access)
    assoc_model = {
        'frequent_itemsets': frequent_itemsets,
        'rules_df': rules_df,
        'params': {
            'algorithm': 'FP-Growth',
            'min_support': min_support,
            'metric': metric_type,
            'min_threshold': min_threshold
        },
        'metrics': {
            'num_rules': len(rules_df),
            'avg_lift': avg_lift if len(rules_df) > 0 else 0.0,
            'avg_confidence': avg_confidence if len(rules_df) > 0 else 0.0
        }
    }

    pickle_path = models_dir / "association_model.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(assoc_model, f)

    print(f"[OK] Saved model pickle to: {pickle_path}")
    mlflow.log_artifact(str(pickle_path), "association_model")

    # Save config JSON
    import json
    config_data = {
        'model_type': 'FP-Growth',
        'status': 'deployed',
        'params': assoc_model['params'],
        'metrics': assoc_model['metrics'],
        'num_rules': len(rules_df),
        'use_case': 'Product recommendations and bundle suggestions'
    }

    config_path = models_dir / "association_rules_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"[OK] Saved config to: {config_path}")
    mlflow.log_artifact(str(config_path), "config")

    # Get MLflow run info
    run_id = mlflow.active_run().info.run_id
    print(f"\n[SUCCESS] MLflow Run ID: {run_id}")

print("\n" + "=" * 80)
print("ASSOCIATION RULES GENERATION COMPLETE!")
print("=" * 80)
print(f"\nGenerated {len(rules_df)} association rules")
print(f"Files saved to: {models_dir}")
print("\nReady for Streamlit integration!")
