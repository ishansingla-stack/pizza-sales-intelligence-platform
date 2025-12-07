import mlflow
from src.config_loader import ConfigLoader

config = ConfigLoader()
config.setup_databricks_env()

client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('/pizza-sales-prediction')
runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=200)

print(f'Total revenue runs: {len(runs)}')

base_models = [r for r in runs if r.data.params.get('model_type') != 'Ensemble']
print(f'Base models: {len(base_models)}')

# Count by model type
from collections import Counter
model_counts = Counter([r.data.params.get('model_type') for r in base_models])
print(f'\nBy model type:')
for model, count in sorted(model_counts.items()):
    print(f'  {model}: {count}')

ensembles = [r for r in runs if r.data.params.get('model_type') == 'Ensemble']
print(f'\nEnsembles: {len(ensembles)}')
