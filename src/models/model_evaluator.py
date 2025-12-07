"""
Comprehensive Model Evaluation Framework
Tests multiple models for each task and selects the best based on metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, davies_bouldin_score
)

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Classification models  
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Clustering models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
# Optional: sklearn_extra requires C++ build tools on Windows
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False
    KMedoids = None

# Additional models
import xgboost as xgb
import lightgbm as lgb
# Optional: catboost requires Visual Studio 2022 or C++ Build Tools on Windows
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    CatBoostRegressor = None
    CatBoostClassifier = None

import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation framework for all ML tasks
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.best_models = {}
        
    def evaluate_sales_forecasting(self, X, y, test_size=0.2):
        """
        Test all regression models for sales forecasting
        Returns the best model based on RMSE
        """
        print("\n" + "="*60)
        print("🎯 SALES FORECASTING MODEL COMPARISON")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to test
        models = {
            'Naive Baseline': self._create_naive_model(y_train, len(y_test)),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1),
            'KNN Regression': KNeighborsRegressor(n_neighbors=5),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=-1),
            'MLP Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        # Add CatBoost only if available
        if HAS_CATBOOST:
            models['CatBoost'] = CatBoostRegressor(iterations=100, depth=5, random_seed=42, verbose=False)
        
        results = []
        
        for name, model in models.items():
            try:
                if name == 'Naive Baseline':
                    # Special handling for naive baseline
                    y_pred = model
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'Model': name,
                    'MAE': round(mae, 2),
                    'RMSE': round(rmse, 2),
                    'MAPE': round(mape, 2),
                    'R2': round(r2, 4)
                })
                
                if self.verbose:
                    print(f"✓ {name:<25} RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
                    
            except Exception as e:
                print(f"✗ {name:<25} Failed: {str(e)[:50]}")
        
        # Create results dataframe and sort by RMSE
        results_df = pd.DataFrame(results).sort_values('RMSE')
        
        # Store best model
        best_model_name = results_df.iloc[0]['Model']
        if best_model_name != 'Naive Baseline':
            self.best_models['sales_forecasting'] = {
                'model': models[best_model_name],
                'scaler': scaler,
                'metrics': results_df.iloc[0].to_dict()
            }
        
        print("\n📊 RANKING (Best to Worst by RMSE):")
        print(results_df.to_string(index=False))
        print(f"\n🏆 WINNER: {best_model_name} (RMSE: {results_df.iloc[0]['RMSE']})")
        
        self.results['sales_forecasting'] = results_df
        return results_df
    
    def evaluate_ingredient_forecasting(self, X, y, test_size=0.2):
        """
        Test models for ingredient usage forecasting
        """
        print("\n" + "="*60)
        print("🥬 INGREDIENT FORECASTING MODEL COMPARISON")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Naive (Last Value)': self._create_naive_model(y_train, len(y_test)),
            'Moving Average': self._create_moving_avg_model(y_train, len(y_test), window=7),
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'KNN': KNeighborsRegressor(n_neighbors=7),
            'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=-1),
            'MLP Neural Net': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
        }
        
        # Add CatBoost only if available
        if HAS_CATBOOST:
            models['CatBoost'] = CatBoostRegressor(iterations=100, depth=5, random_seed=42, verbose=False)
        
        results = []
        
        for name, model in models.items():
            try:
                if name in ['Naive (Last Value)', 'Moving Average']:
                    y_pred = model
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
                
                results.append({
                    'Model': name,
                    'MAE': round(mae, 2),
                    'RMSE': round(rmse, 2),
                    'MAPE': round(mape, 2)
                })
                
                if self.verbose:
                    print(f"✓ {name:<25} RMSE: {rmse:.2f}")
                    
            except Exception as e:
                print(f"✗ {name:<25} Failed: {str(e)[:50]}")
        
        results_df = pd.DataFrame(results).sort_values('RMSE')
        
        print("\n📊 RANKING (Best to Worst by RMSE):")
        print(results_df.to_string(index=False))
        print(f"\n🏆 WINNER: {results_df.iloc[0]['Model']} (RMSE: {results_df.iloc[0]['RMSE']})")
        
        self.results['ingredient_forecasting'] = results_df
        return results_df
    
    def evaluate_promotion_classification(self, X, y, test_size=0.2):
        """
        Test classification models for promotion recommendation
        """
        print("\n" + "="*60)
        print("🏷️ PROMOTION CLASSIFICATION MODEL COMPARISON")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=-1),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', probability=True)
        }
        
        # Add CatBoost only if available
        if HAS_CATBOOST:
            models['CatBoost'] = CatBoostClassifier(iterations=100, depth=5, random_seed=42, verbose=False)
        
        results = []
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results.append({
                    'Model': name,
                    'Accuracy': round(accuracy, 4),
                    'Precision': round(precision, 4),
                    'Recall': round(recall, 4),
                    'F1': round(f1, 4)
                })
                
                if self.verbose:
                    print(f"✓ {name:<25} Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
                    
            except Exception as e:
                print(f"✗ {name:<25} Failed: {str(e)[:50]}")
        
        results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
        
        print("\n📊 RANKING (Best to Worst by F1 Score):")
        print(results_df.to_string(index=False))
        print(f"\n🏆 WINNER: {results_df.iloc[0]['Model']} (F1: {results_df.iloc[0]['F1']})")
        
        self.results['promotion_classification'] = results_df
        return results_df
    
    def evaluate_clustering(self, X, n_clusters=5):
        """
        Test clustering models for product segmentation
        """
        print("\n" + "="*60)
        print("🎨 CLUSTERING MODEL COMPARISON")
        print("="*60)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = {
            'K-Means': KMeans(n_clusters=n_clusters, random_state=42),
            'Hierarchical': AgglomerativeClustering(n_clusters=n_clusters),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Gaussian Mixture': GaussianMixture(n_components=n_clusters, random_state=42)
        }
        
        # Add K-Medoids only if available
        if HAS_KMEDOIDS:
            models['K-Medoids'] = KMedoids(n_clusters=n_clusters, random_state=42)
        
        results = []
        
        for name, model in models.items():
            try:
                labels = model.fit_predict(X_scaled)
                
                # Handle DBSCAN which might have -1 labels (noise)
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                    davies_bouldin = davies_bouldin_score(X_scaled, labels)
                    n_clusters_found = len(np.unique(labels[labels >= 0]))
                else:
                    silhouette = -1
                    davies_bouldin = 999
                    n_clusters_found = 1
                
                results.append({
                    'Model': name,
                    'Silhouette': round(silhouette, 4),
                    'Davies-Bouldin': round(davies_bouldin, 4),
                    'N_Clusters': n_clusters_found
                })
                
                if self.verbose:
                    print(f"✓ {name:<25} Silhouette: {silhouette:.4f} | Clusters: {n_clusters_found}")
                    
            except Exception as e:
                print(f"✗ {name:<25} Failed: {str(e)[:50]}")
        
        results_df = pd.DataFrame(results).sort_values('Silhouette', ascending=False)
        
        print("\n📊 RANKING (Best to Worst by Silhouette Score):")
        print(results_df.to_string(index=False))
        print(f"\n🏆 WINNER: {results_df.iloc[0]['Model']} (Silhouette: {results_df.iloc[0]['Silhouette']})")
        
        self.results['clustering'] = results_df
        return results_df
    
    def _create_naive_model(self, y_train, test_length):
        """Create naive forecast (last value repeated)"""
        return np.full(test_length, y_train.iloc[-1] if hasattr(y_train, 'iloc') else y_train[-1])
    
    def _create_moving_avg_model(self, y_train, test_length, window=7):
        """Create moving average forecast"""
        if hasattr(y_train, 'rolling'):
            last_avg = y_train.rolling(window=window).mean().iloc[-1]
        else:
            last_avg = np.mean(y_train[-window:])
        return np.full(test_length, last_avg)
    
    def get_all_results(self):
        """Return all evaluation results"""
        return self.results
    
    def save_results(self, filepath):
        """Save results to Excel file with multiple sheets"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for task, df in self.results.items():
                df.to_excel(writer, sheet_name=task, index=False)
        print(f"✅ Results saved to {filepath}")
