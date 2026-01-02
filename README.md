# Pizza Intelligence Platform

**Predict pizza sales revenue and optimize business operations with machine learning**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

The Pizza Intelligence Platform is a comprehensive machine learning solution that analyzes 48,620 pizza orders to provide actionable business insights including:

- **Revenue Forecasting**: Ensemble model (R² = 0.699, RMSE = 2.44)
- **Demand Forecasting**: Hour-by-hour prediction (R² = 0.692, RMSE = 4.53 pizzas/hour)
- **Bundle Recommendations**: 460 association rules discovered (Max Lift: 1.38x)
- **Customer Segmentation**: DBSCAN clustering (Silhouette: 0.831)
- **Interactive Dashboard**: 6-page Streamlit application

## Features

### Dashboard Pages
1. **Executive Dashboard** - Key metrics and insights
2. **Bundle Recommendations** - Market basket analysis
3. **Customer Segments** - Clustering insights
4. **Sales Forecasting** - Predictive revenue models
5. **Staffing & Peak Hours** - Predictive staffing optimization
6. **Business Metrics** - Performance analytics

### Analytics Breakdown
- **Descriptive**: 35% (Historical patterns)
- **Predictive**: 50% (Forecasting & predictions)
- **Prescriptive**: 15% (Actionable recommendations)

## Model Performance

### Revenue Forecasting
- **Champion Model**: Ensemble #5 (n=5, uniform weights)
- **Test R²**: 0.6988 (69.9% variance explained)
- **Test RMSE**: 2.4381
- **Test MAE**: 0.6539
- **Models Trained**: 88 runs (80 individual + 8 ensembles)

### Demand Forecasting (Staffing Optimization)
- **Champion Model**: Ensemble #5 (n=5, uniform weights)
- **Test R²**: 0.6923 (69.2% variance explained)
- **Test RMSE**: 4.53 pizzas/hour
- **Test MAE**: 3.56 pizzas/hour

### Clustering (Customer Segmentation)
- **Champion**: DBSCAN (run_2)
- **Silhouette Score**: 0.8305 (excellent cluster separation)
- **Parameters**: eps=0.5, min_samples=2, metric=manhattan

### Association Rules (Bundle Recommendations)
- **Algorithm**: FP-Growth (sup=0.005, lift=0.5)
- **Rules Generated**: 460
- **Average Lift**: 1.0929
- **Max Lift**: 1.3761
- **Average Confidence**: 0.0907 (9.07%)

## Quick Start

### Local Deployment

```bash
# 1. Clone repository
git clone https://github.com/ishansingla-stack/pizza-sales-intelligence-platform.git
cd pizza-sales-intelligence-platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run dashboard
streamlit run dashboards/streamlit_app.py
```

Dashboard opens at: **http://localhost:8501**

### Streamlit Cloud Deployment

1. **Fork/Clone** this repository to your GitHub account
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **"New app"**
4. Configure:
   - **Repository**: `your-username/pizza-sales-intelligence-platform`
   - **Branch**: `main`
   - **Main file path**: `dashboards/streamlit_app.py`
5. Click **"Deploy!"**

**Live in 2-3 minutes!**

## Project Structure

```
pizza-intelligence/
├── dashboards/
│   └── streamlit_app.py          # Main dashboard application
├── data/
│   └── raw/
│       └── Data_Model_-_Pizza_Sales.xlsx  # Source data (48,620 records)
├── outputs/
│   └── results/
│       ├── dashboard_data/        # Dashboard CSV files
│       ├── ml_model_tracking/     # MLflow experiment results
│       └── config/                # Production configurations
├── reports/
│   ├── BUSINESS_INSIGHTS_REPORT.md      # Business analysis
│   ├── TECHNICAL_DOCUMENTATION.md       # Model details
│   └── DATA_PREPARATION.md              # Data cleaning docs
├── scripts/
│   ├── 01_data_preparation/       # Data preprocessing
│   ├── 02_model_training/         # ML model training
│   ├── 03_exploratory_analysis/   # Clustering & association
│   └── 04_model_selection/        # Hyperparameter tuning & ensembles
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── requirements.txt               # Python dependencies
├── DEPLOYMENT.md                  # Detailed deployment guide
└── README.md                      # This file
```

## Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Full deployment guide
- [BUSINESS_INSIGHTS_REPORT.md](reports/BUSINESS_INSIGHTS_REPORT.md) - ROI analysis & recommendations
- [TECHNICAL_DOCUMENTATION.md](reports/TECHNICAL_DOCUMENTATION.md) - Model performance & architecture
- [DATA_PREPARATION.md](reports/DATA_PREPARATION.md) - Data cleaning & feature engineering

## Key Results

### Business Impact
- **Revenue Opportunity**: $120K-$250K annual increase from bundles
- **Cost Savings**: $30K-$45K from menu optimization
- **Total Projected Impact**: $150K-$295K annually

### Model Achievements
- **69.9% R²**: Revenue forecasting with ensemble methods
- **69.2% R²**: Demand forecasting for staffing optimization
- **1.38x Max Lift**: Bundle recommendations from association rules
- **0.831 Silhouette**: Excellent customer cluster separation

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.12+ |
| **ML Frameworks** | scikit-learn, XGBoost, LightGBM |
| **Experiment Tracking** | MLflow (Databricks) |
| **Dashboard** | Streamlit, Plotly |
| **Data Processing** | pandas, numpy |
| **Association Rules** | mlxtend (Apriori, FP-Growth) |

## Core Requirements

See [requirements.txt](requirements.txt) for the complete list of dependencies.

**Key packages:**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=4.0.0
streamlit>=1.28.0
plotly>=5.17.0
mlxtend>=0.22.0
mlflow>=2.9.0
```

## Dataset

- **Source**: Pizza sales transaction data
- **Records**: 48,620 orders
- **Pizzas**: 32 varieties
- **Features**: 51 engineered features
- **Time Period**: Full year of sales


## License

MIT License - See LICENSE file for details

## Author

**CIS 508 - Business Intelligence & Data Analytics**
Arizona State University
December 2025

## Support

For questions or issues:
1. Check [DEPLOYMENT.md](DEPLOYMENT.md) for deployment troubleshooting
2. Review documentation in [reports/](reports/)
3. Open an issue on GitHub

---

**Built with**: Python, Streamlit, Plotly, scikit-learn, XGBoost, MLflow
**Powered by**: Machine Learning, Predictive Analytics, Data Science
