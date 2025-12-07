import mlflow
from src.config_loader import ConfigLoader
from collections import Counter

config = ConfigLoader()
config.setup_databricks_env()

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('/pizza-clustering-association')

runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=200)
print(f"Total runs: {len(runs)}\n")

# Check what parameters exist
all_params = []
for run in runs:
    all_params.extend(list(run.data.params.keys()))

param_counts = Counter(all_params)
print("Parameters found:")
for param, count in param_counts.most_common(20):
    print(f"  {param}: {count} runs")

print("\n" + "="*80)
print("Sample runs:")
for i, run in enumerate(runs[:5], 1):
    print(f"\n{i}. {run.info.run_name}")
    print(f"   Params: {list(run.data.params.keys())}")
    print(f"   Metrics: {list(run.data.metrics.keys())}")
    if run.data.params.get('algorithm'):
        print(f"   Algorithm: {run.data.params['algorithm']}")

# Try to identify clustering vs association by algorithm
print("\n" + "="*80)
print("By algorithm:")
algorithms = Counter([r.data.params.get('algorithm') for r in runs if r.data.params.get('algorithm')])
for algo, count in algorithms.most_common():
    print(f"  {algo}: {count}")
