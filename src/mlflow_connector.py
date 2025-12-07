"""
MLflow Connector for Streamlit App
Connects to Databricks MLflow Model Registry to load trained models
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os

class MLflowModelConnector:
    """
    Connects Streamlit app to Databricks MLflow models
    """

    def __init__(self, tracking_uri: str = None, registry_uri: str = None):
        """
        Initialize MLflow connection

        Args:
            tracking_uri: Databricks MLflow tracking URI
            registry_uri: Databricks MLflow registry URI
        """
        # Set tracking URI (from environment or parameter)
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI')
        self.registry_uri = registry_uri or os.getenv('MLFLOW_REGISTRY_URI')

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)

        self.loaded_models = {}

    def load_model_from_registry(self, model_name: str, stage: str = "Production") -> Any:
        """
        Load model from MLflow Model Registry

        Args:
            model_name: Name of registered model
            stage: Model stage (Production, Staging, None)

        Returns:
            Loaded model
        """
        model_uri = f"models:/{model_name}/{stage}"

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            self.loaded_models[model_name] = model
            print(f"✅ Loaded model: {model_name} ({stage})")
            return model
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
            return None

    def load_model_from_run(self, run_id: str, artifact_path: str = "model") -> Any:
        """
        Load model from specific MLflow run

        Args:
            run_id: MLflow run ID
            artifact_path: Path to model artifact

        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            self.loaded_models[run_id] = model
            print(f"✅ Loaded model from run: {run_id}")
            return model
        except Exception as e:
            print(f"❌ Error loading model from run {run_id}: {e}")
            return None

    def predict(self, model_name: str, input_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using loaded model

        Args:
            model_name: Name of model to use
            input_data: Input features as DataFrame

        Returns:
            Predictions array
        """
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded. Call load_model_from_registry() first.")

        model = self.loaded_models[model_name]
        predictions = model.predict(input_data)

        return predictions

    def get_model_metadata(self, model_name: str, stage: str = "Production") -> Dict:
        """
        Get metadata for a registered model

        Args:
            model_name: Name of registered model
            stage: Model stage

        Returns:
            Dictionary with model metadata
        """
        try:
            client = mlflow.tracking.MlflowClient()

            # Get model version
            model_versions = client.get_latest_versions(model_name, stages=[stage])

            if len(model_versions) == 0:
                return {"error": f"No model found in {stage} stage"}

            mv = model_versions[0]

            # Get run details
            run = client.get_run(mv.run_id)

            metadata = {
                "model_name": model_name,
                "version": mv.version,
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "created_timestamp": mv.creation_timestamp,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }

            return metadata

        except Exception as e:
            return {"error": str(e)}

    def list_registered_models(self) -> pd.DataFrame:
        """
        List all registered models in MLflow

        Returns:
            DataFrame with model information
        """
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.search_registered_models()

            model_list = []
            for model in models:
                for mv in model.latest_versions:
                    model_list.append({
                        'name': model.name,
                        'version': mv.version,
                        'stage': mv.current_stage,
                        'run_id': mv.run_id
                    })

            return pd.DataFrame(model_list)

        except Exception as e:
            print(f"Error listing models: {e}")
            return pd.DataFrame()

    def get_experiment_runs(self, experiment_name: str) -> pd.DataFrame:
        """
        Get all runs from an experiment

        Args:
            experiment_name: Name of experiment

        Returns:
            DataFrame with run information
        """
        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)

            if experiment is None:
                print(f"Experiment {experiment_name} not found")
                return pd.DataFrame()

            # Search runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.test_rmse ASC"]  # Order by best RMSE
            )

            return runs

        except Exception as e:
            print(f"Error getting experiment runs: {e}")
            return pd.DataFrame()


# Example usage configuration
class StreamlitMLflowConfig:
    """
    Configuration for connecting Streamlit to Databricks MLflow

    FILL IN YOUR DATABRICKS DETAILS:
    """

    # Databricks workspace URL
    DATABRICKS_HOST = "https://your-workspace.cloud.databricks.com"

    # Personal Access Token (keep this secure!)
    DATABRICKS_TOKEN = "dapi1234567890abcdef"  # Replace with your token

    # MLflow tracking URI
    TRACKING_URI = f"{DATABRICKS_HOST}"

    # Environment variables (recommended for security)
    @staticmethod
    def setup_environment():
        """
        Set up environment variables for MLflow connection
        """
        os.environ['DATABRICKS_HOST'] = StreamlitMLflowConfig.DATABRICKS_HOST
        os.environ['DATABRICKS_TOKEN'] = StreamlitMLflowConfig.DATABRICKS_TOKEN
        os.environ['MLFLOW_TRACKING_URI'] = f"databricks"


# Example usage in Streamlit
def example_streamlit_usage():
    """
    Example of how to use MLflow connector in Streamlit app
    """

    # Setup (do this once at app startup)
    StreamlitMLflowConfig.setup_environment()

    # Create connector
    connector = MLflowModelConnector()

    # Load production model
    model = connector.load_model_from_registry(
        model_name="pizza_sales_forecast",
        stage="Production"
    )

    # Get model metadata
    metadata = connector.get_model_metadata("pizza_sales_forecast", "Production")
    print(f"Model RMSE: {metadata['metrics'].get('test_rmse', 'N/A')}")
    print(f"Model R²: {metadata['metrics'].get('test_r2', 'N/A')}")

    # Make predictions
    # Prepare input data (same features used in training)
    input_data = pd.DataFrame({
        'month': [12],
        'day_of_week': [4],
        'week': [52],
        'quarter': [4],
        'is_weekend': [0],
        'is_extreme_heat': [0],
        'is_snowbird_season': [1],
        'is_perfect_weather': [0],
        'month_sin': [0.0],
        'month_cos': [1.0],
        'day_sin': [-0.5],
        'day_cos': [0.87],
        'total_quantity': [300],
        'num_orders': [150]
    })

    # Get predictions
    predictions = connector.predict("pizza_sales_forecast", input_data)
    print(f"Predicted revenue: ${predictions[0]:,.2f}")

    # List all models
    all_models = connector.list_registered_models()
    print(all_models)


if __name__ == "__main__":
    # Test connection
    print("=" * 80)
    print("MLflow Connector Test")
    print("=" * 80)
    print("\nTo use this connector:")
    print("1. Fill in your Databricks credentials in StreamlitMLflowConfig")
    print("2. Import this module in your Streamlit app")
    print("3. Use MLflowModelConnector to load and use models")
    print("\nExample:")
    print("  from src.mlflow_connector import MLflowModelConnector, StreamlitMLflowConfig")
    print("  StreamlitMLflowConfig.setup_environment()")
    print("  connector = MLflowModelConnector()")
    print("  model = connector.load_model_from_registry('pizza_sales_forecast', 'Production')")
