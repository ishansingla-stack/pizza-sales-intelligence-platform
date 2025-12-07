"""
Data Loader Module for Pizza Sales Intelligence Platform
Supports: Excel, CSV, and Database connections
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union
import yaml
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Versatile data loader supporting multiple data sources:
    - Excel files
    - CSV files
    - Database connections (PostgreSQL, MySQL, SQLite)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data loader with optional config file
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.data = None
        self.metadata = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            return {}
    
    def load_from_excel(self, filepath: str, sheet_name: str = 'pizza_sales') -> pd.DataFrame:
        """
        Load data from Excel file
        
        Args:
            filepath: Path to Excel file
            sheet_name: Name of sheet to load
            
        Returns:
            DataFrame with loaded data
        """
        try:
            self.data = pd.read_excel(filepath, sheet_name=sheet_name)
            self._process_dates()
            self._validate_data()
            self.metadata['source'] = 'excel'
            self.metadata['filepath'] = filepath
            self.metadata['load_time'] = datetime.now()
            print(f"Successfully loaded {len(self.data)} records from Excel")
            return self.data
        except Exception as e:
            raise Exception(f"Error loading Excel file: {e}")
    
    def load_from_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional pandas read_csv arguments
            
        Returns:
            DataFrame with loaded data
        """
        try:
            self.data = pd.read_csv(filepath, **kwargs)
            self._process_dates()
            self._validate_data()
            self.metadata['source'] = 'csv'
            self.metadata['filepath'] = filepath
            self.metadata['load_time'] = datetime.now()
            print(f"Successfully loaded {len(self.data)} records from CSV")
            return self.data
        except Exception as e:
            raise Exception(f"Error loading CSV file: {e}")
    
    def load_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """
        Load data from database using SQLAlchemy
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            
        Returns:
            DataFrame with loaded data
        """
        try:
            from sqlalchemy import create_engine
            engine = create_engine(connection_string)
            self.data = pd.read_sql_query(query, engine)
            self._process_dates()
            self._validate_data()
            self.metadata['source'] = 'database'
            self.metadata['connection'] = connection_string.split('@')[-1] if '@' in connection_string else 'local'
            self.metadata['load_time'] = datetime.now()
            print(f"Successfully loaded {len(self.data)} records from database")
            return self.data
        except ImportError:
            raise Exception("SQLAlchemy not installed. Run: pip install sqlalchemy")
        except Exception as e:
            raise Exception(f"Error loading from database: {e}")
    
    def _process_dates(self):
        """Process date columns and ensure correct datetime format"""
        if self.data is not None and 'order_date' in self.data.columns:
            self.data['order_date'] = pd.to_datetime(self.data['order_date'])
            
        if self.data is not None and 'order_time' in self.data.columns:
            # Convert time string to datetime time
            if self.data['order_time'].dtype == 'object':
                self.data['order_time'] = pd.to_datetime(self.data['order_time'], format='%H:%M:%S').dt.time
    
    def _validate_data(self):
        """Validate required columns exist"""
        required_columns = [
            'order_id', 'pizza_id', 'quantity', 'order_date',
            'order_time', 'unit_price', 'total_price', 'pizza_size',
            'pizza_category', 'pizza_ingredients', 'pizza_name'
        ]
        
        if self.data is not None:
            missing_cols = set(required_columns) - set(self.data.columns)
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
            
            # Data quality check
            null_counts = self.data.isnull().sum()
            if null_counts.any():
                print(f"Warning: Found null values:\n{null_counts[null_counts > 0]}")
    
    def get_data_summary(self) -> Dict:
        """Get comprehensive data summary"""
        if self.data is None:
            return {"error": "No data loaded"}
        
        return {
            "total_records": len(self.data),
            "unique_orders": self.data['order_id'].nunique(),
            "unique_pizzas": self.data['pizza_name'].nunique(),
            "date_range": {
                "start": str(self.data['order_date'].min()),
                "end": str(self.data['order_date'].max())
            },
            "categories": self.data['pizza_category'].value_counts().to_dict(),
            "sizes": self.data['pizza_size'].value_counts().to_dict(),
            "total_revenue": self.data['total_price'].sum(),
            "metadata": self.metadata
        }
