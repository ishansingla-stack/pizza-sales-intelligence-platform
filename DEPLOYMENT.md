# Pizza Intelligence Platform - Deployment Guide

## 🚀 Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Streamlit Cloud account (free tier available at [share.streamlit.io](https://share.streamlit.io))
- All required data files in the repository

### Deployment Steps

#### 1. **Prepare Your Repository**
```bash
# Ensure all files are committed
git add .
git commit -m "Prepare for Streamlit deployment"
git push origin main
```

#### 2. **Deploy to Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Configure deployment:
   - **Repository**: `your-username/pizza-intelligence-platform`
   - **Branch**: `main`
   - **Main file path**: `dashboards/streamlit_app.py`
   - **App URL**: Choose your custom URL

5. Click "Deploy!"

#### 3. **Verify Deployment**
The app should be live at `https://your-app-name.streamlit.app` within 2-3 minutes.

---

## 💻 Local Deployment

### 1. **Install Dependencies**
```bash
cd pizza-intelligence
pip install -r requirements.txt
```

### 2. **Run Locally**
```bash
streamlit run dashboards/streamlit_app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📁 Required Data Files

Ensure these files exist before deployment:

### Raw Data
- `data/raw/Data_Model_-_Pizza_Sales.xlsx`

### Analysis Results (in `outputs/results/`)
- `dashboard_data/bundle_recommendations.csv` - Bundle recommendations data
- `dashboard_data/customer_segments.csv` - Customer segmentation data
- `ml_model_tracking/revenue_prediction_runs.csv` - Revenue forecasting models (88 runs)
- `ml_model_tracking/quantity_prediction_runs.csv` - Sales volume models (112 runs)

### Generate Missing Results
If any result files are missing, run the corresponding scripts:

```bash
# Association rules analysis
python scripts/03_exploratory_analysis/01_association_rules.py

# Clustering analysis
python scripts/03_exploratory_analysis/02_clustering.py

# Sales prediction models
python scripts/02_model_training/01_sales_prediction.py
python scripts/02_model_training/02_quantity_prediction.py
```

---

## 🎨 Dashboard Features

### 6 Interactive Pages:
1. **📊 Executive Dashboard** - Key metrics and insights
2. **🍕 Bundle Recommendations** - Market basket analysis
3. **👥 Customer Segments** - Clustering insights
4. **📈 Sales Forecasting** - Predictive revenue models
5. **⏰ Staffing & Peak Hours** - Predictive staffing optimization (NEW!)
6. **📉 Business Metrics** - Performance analytics

### Analytics Breakdown:
- **Descriptive**: 35% (Historical patterns)
- **Predictive**: 50% (Forecasting & predictions)
- **Prescriptive**: 15% (Actionable recommendations)

---

## 🔧 Configuration

### Streamlit Configuration
Configuration is managed in `.streamlit/config.toml`:
- Theme colors (red accent)
- Server settings
- Upload limits
- Browser preferences

### Environment Variables
No environment variables required for basic deployment. All data is loaded from local files.

For MLflow tracking (optional):
```bash
export MLFLOW_TRACKING_URI="databricks"
export DATABRICKS_HOST="your-host"
export DATABRICKS_TOKEN="your-token"
```

---

## 🐛 Troubleshooting

### Common Issues:

**1. Missing Data Files**
```
Error: FileNotFoundError: data/raw/Data_Model_-_Pizza_Sales.xlsx
```
**Solution**: Ensure raw data file exists in `data/raw/` directory

**2. Missing Result CSVs**
```
Error: FileNotFoundError: outputs/results/association_rules.csv
```
**Solution**: Run the corresponding analysis script (see "Generate Missing Results" above)

**3. Import Errors**
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Install requirements: `pip install -r requirements.txt`

**4. Large File Upload Limit**
If deploying to Streamlit Cloud with large data files, consider:
- Using Git LFS for large files
- Compressing data files
- Using remote data sources

---

## 📊 Performance Optimization

### For Large Datasets:
1. **Enable Caching**: Already implemented with `@st.cache_data`
2. **Data Compression**: Use Parquet format (already used)
3. **Lazy Loading**: Load data only when needed (implemented)

### Memory Management:
Current dashboard is optimized for datasets up to 100k rows. For larger datasets:
- Increase Streamlit Cloud resources (paid tier)
- Implement data sampling
- Use database backends

---

## 🔒 Security Considerations

### Data Privacy:
- All data is processed locally (no external API calls)
- No sensitive credentials in repository
- Use environment variables for secrets

### Streamlit Secrets (if needed):
Create `.streamlit/secrets.toml` for sensitive data:
```toml
[mlflow]
tracking_uri = "your-mlflow-uri"
username = "your-username"
password = "your-password"
```

**⚠️ Important**: Add `.streamlit/secrets.toml` to `.gitignore`

---

## 📈 Monitoring & Maintenance

### Streamlit Cloud Monitoring:
- View app logs in Streamlit Cloud dashboard
- Monitor app usage and performance
- Set up email alerts for app downtime

### Updates:
To update the deployed app:
```bash
git add .
git commit -m "Update dashboard features"
git push origin main
```
Streamlit Cloud will automatically redeploy.

---

## 🆘 Support

- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Create an issue in your repository

---

## 📝 License

This project is part of CIS 508 coursework at Arizona State University.

**Built with**: Python, Streamlit, Plotly, scikit-learn, XGBoost, MLflow
**Powered by**: Machine Learning, Predictive Analytics, Data Science
