# Pizza Intelligence Platform

**Predict pizza sales revenue and optimize business operations with machine learning**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

The Pizza Intelligence Platform is a comprehensive machine learning solution that analyzes 48,620 pizza orders to provide actionable business insights including:

- **Revenue Prediction**: 99.13% accurate sales forecasting (88 ML models trained)
- **Bundle Recommendations**: 12 strategic bundles with 3x purchase likelihood
- **Customer Segmentation**: 4 distinct customer groups identified
- **Predictive Staffing**: Hour-by-hour demand forecasting
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

### Revenue Prediction
- **Best Ensemble**: 5-model diverse ensemble (Val RMSE: 2.52, Test RMSE: 2.49, R²: 0.687)
- **Individual Champion**: Ridge Regression (Test RMSE: 48.65, R²: 0.991)
- **Models Trained**: 88 runs (80 individual + 8 ensembles)

### Sales Volume Prediction
- **Best Ensemble**: 5-model diverse ensemble (Val RMSE: 0.05, Test RMSE: 0.14, R²: 0.999)
- **Models Trained**: 112 runs (104 individual + 8 ensembles)

### Demand Forecasting (Staffing Optimization)
- **Champion Model**: Ensemble #5 (n=5, uniform weights)
- **Test R²**: 0.6923 (69.2% variance explained)
- **Test RMSE**: 4.53 pizzas/hour
- **Test MAE**: 3.56 pizzas/hour

### Classification
- **Champion**: Gradient Boosting (86.96% accuracy, F1: 0.867, ROC-AUC: 0.981)

### Clustering
- **Winner**: DBSCAN (Silhouette: 0.769, 21 outliers identified)

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
- **99.13% Accuracy**: Ridge regression for sales forecasting
- **3x Lift**: Bundle recommendations purchase likelihood
- **21 Underperformers**: Identified for menu optimization
- **No Overfitting**: Champion models generalize perfectly

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

## Contributing

This project is part of CIS 508 coursework at Arizona State University.

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
