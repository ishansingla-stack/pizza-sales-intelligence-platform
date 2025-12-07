import mlflow
from src.config_loader import ConfigLoader

config = ConfigLoader()
config.setup_databricks_env()

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('/pizza-clustering-association')

all_runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=200)

# Get association runs
assoc_runs = [r for r in all_runs if r.data.params.get('algorithm') in ['Apriori', 'FP-Growth']]

print(f"Total association runs: {len(assoc_runs)}\n")

# Filter runs with actual rules
runs_with_rules = []
for run in assoc_runs:
    rules_count = run.data.metrics.get('rules_count', 0)
    avg_lift = run.data.metrics.get('avg_lift', 0)
    if rules_count > 0:
        runs_with_rules.append({
            'run_name': run.info.run_name,
            'algorithm': run.data.params.get('algorithm'),
            'rules_count': rules_count,
            'avg_lift': avg_lift,
            'run_id': run.info.run_id
        })

# Sort by avg_lift
runs_with_rules = sorted(runs_with_rules, key=lambda x: x['avg_lift'], reverse=True)

print(f"Runs with actual rules: {len(runs_with_rules)}\n")

if runs_with_rules:
    print("Top 10 by avg_lift:")
    for i, run in enumerate(runs_with_rules[:10], 1):
        print(f"  {i}. {run['algorithm']}: {run['rules_count']} rules, lift={run['avg_lift']:.4f}")
        print(f"     Run ID: {run['run_id']}")
else:
    print("No runs found with rules!")
