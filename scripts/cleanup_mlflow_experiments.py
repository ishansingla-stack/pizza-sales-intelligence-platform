"""
Clean up old MLflow experiments in Databricks
Deletes experiments from before we fixed the quantity/revenue split
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import mlflow
from src.config_loader import ConfigLoader

print("=" * 80)
print("CLEANUP OLD MLFLOW EXPERIMENTS")
print("=" * 80)

# Load configuration
config = ConfigLoader()
config.setup_databricks_env()

# Set experiment
mlflow.set_experiment("/pizza-sales-prediction")

print("\n[*] Connecting to MLflow...")
client = mlflow.tracking.MlflowClient()

# Get all runs in the experiment
experiment = client.get_experiment_by_name("/pizza-sales-prediction")
if experiment:
    experiment_id = experiment.experiment_id
    print(f"[OK] Found experiment: {experiment.name} (ID: {experiment_id})")

    # Get all runs
    all_runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=5000
    )

    print(f"\n[INFO] Total runs in experiment: {len(all_runs)}")

    # Group runs by tags or naming patterns
    revenue_runs = []
    quantity_runs = []
    old_runs = []
    other_runs = []

    for run in all_runs:
        run_name = run.data.tags.get('mlflow.runName', '')

        # Identify run types
        if 'revenue' in run_name.lower() or 'total_price' in run_name.lower():
            revenue_runs.append(run)
        elif 'quantity' in run_name.lower() or 'volume' in run_name.lower():
            quantity_runs.append(run)
        elif run_name == '':
            old_runs.append(run)
        else:
            other_runs.append(run)

    print("\n" + "=" * 80)
    print("RUN BREAKDOWN")
    print("=" * 80)
    print(f"Revenue runs: {len(revenue_runs)}")
    print(f"Quantity runs: {len(quantity_runs)}")
    print(f"Unnamed/old runs: {len(old_runs)}")
    print(f"Other runs: {len(other_runs)}")

    # Ask user which runs to delete
    print("\n" + "=" * 80)
    print("CLEANUP OPTIONS")
    print("=" * 80)
    print("1. Delete unnamed/old runs only")
    print("2. Delete ALL runs (start fresh)")
    print("3. Keep only the latest 88 revenue + 112 quantity runs")
    print("4. Cancel (no deletion)")

    choice = input("\nEnter your choice (1-4): ").strip()

    runs_to_delete = []

    if choice == '1':
        runs_to_delete = old_runs
        print(f"\n[*] Will delete {len(runs_to_delete)} unnamed/old runs")
    elif choice == '2':
        runs_to_delete = all_runs
        print(f"\n[*] Will delete ALL {len(runs_to_delete)} runs")
    elif choice == '3':
        # Keep latest runs based on creation time
        revenue_sorted = sorted(revenue_runs, key=lambda x: x.info.start_time, reverse=True)
        quantity_sorted = sorted(quantity_runs, key=lambda x: x.info.start_time, reverse=True)

        old_revenue = revenue_sorted[88:] if len(revenue_sorted) > 88 else []
        old_quantity = quantity_sorted[112:] if len(quantity_sorted) > 112 else []

        runs_to_delete = old_revenue + old_quantity + old_runs
        print(f"\n[*] Will delete {len(runs_to_delete)} old runs (keeping latest 88 revenue + 112 quantity)")
    else:
        print("\n[CANCELLED] No runs will be deleted")
        sys.exit(0)

    # Confirm deletion
    if runs_to_delete:
        confirm = input(f"\nAre you sure you want to delete {len(runs_to_delete)} runs? (yes/no): ").strip().lower()

        if confirm == 'yes':
            print("\n[*] Deleting runs...")
            deleted_count = 0

            for run in runs_to_delete:
                try:
                    client.delete_run(run.info.run_id)
                    deleted_count += 1
                    if deleted_count % 10 == 0:
                        print(f"   Deleted {deleted_count}/{len(runs_to_delete)} runs...")
                except Exception as e:
                    print(f"   [ERROR] Failed to delete run {run.info.run_id}: {e}")

            print(f"\n[OK] Successfully deleted {deleted_count} runs")
            print(f"[INFO] Remaining runs: {len(all_runs) - deleted_count}")
        else:
            print("\n[CANCELLED] Deletion aborted")
    else:
        print("\n[INFO] No runs to delete")

else:
    print("[ERROR] Experiment '/pizza-sales-prediction' not found")

print("\n" + "=" * 80)
print("CLEANUP COMPLETE")
print("=" * 80)
