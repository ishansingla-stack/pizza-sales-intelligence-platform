import mlflow
from src.config_loader import ConfigLoader

config = ConfigLoader()
config.setup_databricks_env()

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('/pizza-clustering-association')

print(f"Experiment: {exp.name}")
print(f"Experiment ID: {exp.experiment_id}")

runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=200)
print(f"\nTotal runs: {len(runs)}")

# Check task types
print(f"\nBy task_type:")
clustering_count = 0
association_count = 0
other_count = 0

for run in runs:
    task_type = run.data.params.get('task_type')
    if task_type == 'clustering':
        clustering_count += 1
    elif task_type == 'association':
        association_count += 1
    else:
        other_count += 1

print(f"  Clustering: {clustering_count}")
print(f"  Association: {association_count}")
print(f"  Other/None: {other_count}")

# Show recent association runs
print(f"\nRecent association runs:")
assoc_runs = [r for r in runs if r.data.params.get('task_type') == 'association']
for i, run in enumerate(assoc_runs[:5], 1):
    print(f"  {i}. {run.info.run_name}")
    print(f"     Metrics: {dict(run.data.metrics)}")
    print(f"     Params: {dict(run.data.params)}")
