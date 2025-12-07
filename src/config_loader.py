"""
Configuration Loader for Databricks Connection
Loads settings from config file and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Load and manage configuration for Databricks and MLflow"""

    def __init__(self, config_path: str = None):
        """
        Initialize config loader

        Args:
            config_path: Path to YAML config file
        """
        if config_path is None:
            # Default to config/databricks_config.yaml
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "databricks_config.yaml"

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}")
            return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key (supports nested keys with '.')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def setup_databricks_env(self):
        """Set up environment variables for Databricks connection"""
        # Set Databricks host and token
        host = self.get('databricks.host')
        token = self.get('databricks.token')

        if host:
            os.environ['DATABRICKS_HOST'] = host
        if token:
            os.environ['DATABRICKS_TOKEN'] = token

        # Set MLflow tracking URI to Databricks
        os.environ['MLFLOW_TRACKING_URI'] = 'databricks'

        print(" Databricks environment configured")
        print(f"   Host: {host}")
        print(f"   MLflow: databricks")

    def get_databricks_host(self) -> str:
        """Get Databricks host URL"""
        return self.get('databricks.host', os.getenv('DATABRICKS_HOST', ''))

    def get_databricks_token(self) -> str:
        """Get Databricks access token"""
        return self.get('databricks.token', os.getenv('DATABRICKS_TOKEN', ''))

    def get_cluster_id(self) -> str:
        """Get Databricks cluster ID"""
        return self.get('databricks.cluster_id', os.getenv('DATABRICKS_CLUSTER_ID', ''))

    def get_experiment_name(self) -> str:
        """Get MLflow experiment name"""
        return self.get('mlflow.experiment_name', '/pizza-intelligence')

    def get_model_name(self) -> str:
        """Get MLflow model registry name"""
        return self.get('mlflow.model_name', 'pizza_sales_forecast')

    def get_local_data_path(self) -> str:
        """Get local data file path"""
        return self.get('data.local_data_path', './data/raw/Data_Model_-_Pizza_Sales.xlsx')

    def get_processed_data_path(self) -> str:
        """Get processed data directory"""
        return self.get('paths.processed_data', './data/processed/')

    def get_results_path(self) -> str:
        """Get results output directory"""
        return self.get('paths.results', './outputs/results/')

    def create_output_dirs(self):
        """Create output directories if they don't exist"""
        dirs = [
            self.get_processed_data_path(),
            self.get_results_path(),
            self.get('paths.models', './outputs/models/')
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f" Created directory: {dir_path}")


# Global config instance
_config = None

def get_config() -> ConfigLoader:
    """Get global config instance"""
    global _config
    if _config is None:
        _config = ConfigLoader()
    return _config


if __name__ == "__main__":
    # Test configuration
    print("=" * 80)
    print("Configuration Test")
    print("=" * 80)

    config = ConfigLoader()

    print(f"\nDatabricks Host: {config.get_databricks_host()}")
    print(f"Cluster ID: {config.get_cluster_id()}")
    print(f"Experiment Name: {config.get_experiment_name()}")
    print(f"Model Name: {config.get_model_name()}")
    print(f"Local Data Path: {config.get_local_data_path()}")

    print("\nSetting up environment...")
    config.setup_databricks_env()

    print("\nCreating output directories...")
    config.create_output_dirs()

    print("\n Configuration loaded successfully!")
