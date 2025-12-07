# Results Directory Organization

This directory contains organized outputs from the Pizza Intelligence Platform analysis and modeling.

## Directory Structure

```
outputs/results/
├── ml_model_tracking/     # MLflow experiment tracking CSVs
├── dashboard_data/        # Files used by Streamlit dashboard
└── config/                # Production configurations
```

## 📊 ML Model Tracking (`ml_model_tracking/`)

Contains MLflow experiment tracking results for all trained models. These files record hyperparameters, metrics, and performance for model selection and comparison.

| File | Description | Runs |
|------|-------------|------|
| `revenue_prediction_runs.csv` | Revenue prediction experiments (regression) | 88 |
| `quantity_prediction_runs.csv` | Sales volume prediction experiments (regression) | 112 |
| `clustering_experiments.csv` | Customer segmentation experiments (KMeans, DBSCAN, etc.) | 32 |
| `association_experiments.csv` | Market basket analysis experiments (Apriori) | 16 |
| `classification_category_experiments.csv` | Pizza category classification experiments | 11 |
| `classification_confusion_matrix.csv` | Confusion matrices for classification models | 4 |

### Model Performance Highlights

**Revenue Prediction (88 runs):**
- Best Ensemble: 5-model uniform-weighted diverse ensemble
  - Val RMSE: 2.5192, Test RMSE: 2.4916, R²: 0.6871
- Includes: DecisionTree, RandomForest, XGBoost, LightGBM, Ridge, Lasso, ElasticNet, NeuralNetwork, SVR models
- 80 individual models + 8 ensemble configurations

**Sales Volume Prediction (112 runs):**
- Best Ensemble: 5-model uniform-weighted diverse ensemble
  - Val RMSE: 0.0508, Test RMSE: 0.1396, R²: 0.9990
- Includes: Same model types as revenue prediction
- 104 individual models + 8 ensemble configurations

## 🎨 Dashboard Data (`dashboard_data/`)

Contains processed results used by the Streamlit dashboard. These files are loaded by `dashboards/streamlit_app.py`.

| File | Description | Used By |
|------|-------------|---------|
| `bundle_recommendations.csv` | Association rules for product bundling (460 rules) | Bundle Recommendations page |
| `customer_segments.csv` | Pizza clustering with segment assignments (32 pizzas) | Customer Segments page |

**Important:** If you rename these files, update the file paths in `dashboards/streamlit_app.py` (lines 72-73).

## ⚙️ Configuration (`config/`)

| File | Description |
|------|-------------|
| `production_config_unsupervised.json` | Production configuration for unsupervised learning models |

---

## File Naming Convention

- **MLflow Tracking:** `{target}_{task}_runs.csv` or `{task}_experiments.csv`
- **Dashboard Data:** `{business_concept}.csv` (e.g., `bundle_recommendations.csv`)

## Last Updated

Organization implemented: December 6, 2025
- Revenue ensembles: 88 runs (80 models + 8 ensembles)
- Quantity ensembles: 112 runs (104 models + 8 ensembles)
- Clustering: 32 experiments
- Association rules: 16 experiments

---

## Related Documentation

- [DEPLOYMENT.md](../../DEPLOYMENT.md) - Dashboard deployment guide
- [README.md](../../README.md) - Project overview
- MLflow Experiments: Databricks workspace `/pizza-sales-prediction`
