# ✅ Local IDE Setup - Ready to Go!

Your Pizza Intelligence project is now configured to **run locally in your IDE** while connecting to Databricks for MLflow tracking.

---

## 🎯 What's Ready

### ✅ Configuration Files
- [config/databricks_config.yaml.template](config/databricks_config.yaml.template) - Template (copy this)
- [src/config_loader.py](src/config_loader.py) - Configuration loader utility
- [.gitignore](.gitignore) - Protects your secrets

### ✅ Local Scripts (Ready to Run)
- [scripts/01_data_preparation_local.py](scripts/01_data_preparation_local.py) ✅ Ready
- [scripts/02_model_training_regression_local.py](scripts/02_model_training_regression_local.py) ✅ Ready

### 📝 Scripts to Create (Same pattern)
- `scripts/03_hyperparameter_tuning_regression_local.py` - Coming next
- `scripts/04_model_training_classification_local.py` - Coming next
- `scripts/05_clustering_association_local.py` - Coming next

### 📚 Documentation
- [LOCAL_IDE_SETUP.md](LOCAL_IDE_SETUP.md) - Complete setup guide
- [DATABRICKS_PROJECT_SUMMARY.md](DATABRICKS_PROJECT_SUMMARY.md) - Project overview

---

## 🚀 Quick Start (3 Steps)

### Step 1: Create Your Config File
```bash
cd pizza-intelligence
cp config/databricks_config.yaml.template config/databricks_config.yaml
```

### Step 2: Edit Config with Your Credentials

Open `config/databricks_config.yaml` and update:
```yaml
databricks:
  host: "https://YOUR-WORKSPACE.cloud.databricks.com"
  token: "YOUR_TOKEN_HERE"  # Get from Databricks User Settings

mlflow:
  experiment_name: "/Users/YOUR_EMAIL@example.com/pizza-intelligence"
```

**Get your token:**
1. Log into Databricks
2. User Settings → Developer → Access Tokens
3. Generate New Token → Copy it

### Step 3: Run the Scripts!
```bash
# Install dependencies
pip install pyyaml

# Run data preparation
python scripts/01_data_preparation_local.py

# Run model training
python scripts/02_model_training_regression_local.py
```

---

## 📊 What Happens When You Run

### Script 1: Data Preparation
```
✅ Runs on your machine (fast!)
✅ Reads: data/raw/Data_Model_-_Pizza_Sales.xlsx
✅ Creates 40+ features
✅ Saves to: data/processed/*.parquet
✅ Logs metadata to Databricks MLflow
```

### Script 2: Model Training
```
✅ Runs on your machine
✅ Trains 15 regression models
✅ Saves results to: outputs/results/base_models_regression.csv
✅ Logs ALL models to Databricks MLflow
```

---

## 🔧 Where Things Run

| Component | Runs Where? |
|-----------|-------------|
| Code execution | 🖥️ Your local machine |
| Data loading | 🖥️ Local files |
| Model training | 🖥️ Your CPU/GPU |
| Results saving | 🖥️ Local outputs/ folder |
| MLflow tracking | ☁️ Databricks (logged remotely) |
| Model Registry | ☁️ Databricks |

**Best of both worlds:** Fast local development + Centralized MLflow tracking!

---

## 📁 File Structure

```
pizza-intelligence/
├── config/
│   ├── databricks_config.yaml.template  ✅ Template provided
│   └── databricks_config.yaml           👈 YOU CREATE THIS
│
├── src/
│   ├── config_loader.py                 ✅ Ready
│   └── mlflow_connector.py              ✅ Ready (for Streamlit later)
│
├── scripts/
│   ├── 01_data_preparation_local.py     ✅ Ready to run
│   ├── 02_model_training_regression_local.py  ✅ Ready to run
│   ├── 03_hyperparameter_tuning_regression_local.py  📝 Create next
│   ├── 04_model_training_classification_local.py     📝 Create next
│   └── 05_clustering_association_local.py             📝 Create next
│
├── data/
│   ├── raw/
│   │   └── Data_Model_-_Pizza_Sales.xlsx  👈 Your data here
│   └── processed/                          ✅ Auto-created by scripts
│
├── outputs/
│   ├── results/                            ✅ Auto-created
│   └── models/                             ✅ Auto-created
│
├── .gitignore                              ✅ Protects secrets
├── LOCAL_IDE_SETUP.md                      ✅ Full guide
└── LOCAL_IDE_READY.md                      📄 You are here!
```

---

## 🎯 Next Steps for You

### Immediate (Do Now):
1. ✅ Copy config template: `cp config/databricks_config.yaml.template config/databricks_config.yaml`
2. ✅ Get Databricks token from User Settings
3. ✅ Edit `config/databricks_config.yaml` with your credentials
4. ✅ Run: `python scripts/01_data_preparation_local.py`
5. ✅ Run: `python scripts/02_model_training_regression_local.py`
6. ✅ Check results in `outputs/results/`
7. ✅ Check MLflow UI in Databricks

### Next (If You Want More Scripts):
Let me know if you want me to create the remaining 3 scripts:
- Hyperparameter tuning (03)
- Classification models (04)
- Clustering & association rules (05)

They'll follow the same pattern as scripts 01 and 02.

---

## 💡 Example Run

```bash
$ python scripts/01_data_preparation_local.py

================================================================================
PIZZA INTELLIGENCE - DATA PREPARATION
================================================================================

📋 Loading configuration...
✅ Databricks environment configured
   Host: https://your-workspace.cloud.databricks.com
   MLflow: databricks
✅ MLflow experiment: /Users/your_email@example.com/pizza-intelligence

📊 Loading data...
✅ Loaded 48,620 records
Date range: 2015-01-01 to 2015-12-31

================================================================================
DATA QUALITY REPORT
================================================================================
✅ No missing values found
✅ Duplicates: 0

================================================================================
FEATURE ENGINEERING
================================================================================
✅ Feature engineering complete
Original columns: 12
New columns: 45

================================================================================
SAVING PROCESSED DATA
================================================================================
✅ Saved processed datasets:
   - Full features: 48620 rows
   - Daily sales: 365 rows
   - Pizza features: 48620 rows
   - Classification: 48620 rows

📁 Location: ./data/processed/

================================================================================
LOGGING TO MLFLOW
================================================================================
✅ Logged to MLflow run: abc123def456
   Experiment: /Users/your_email@example.com/pizza-intelligence
   Run ID: abc123def456

================================================================================
✅ DATA PREPARATION COMPLETE
================================================================================

Next step: Run 02_model_training_regression_local.py
```

---

## 🔒 Security Checklist

- ✅ `.gitignore` includes `config/databricks_config.yaml`
- ✅ Template file (.template) is safe to commit
- ✅ Your actual config file will NOT be committed
- ✅ No tokens hardcoded in scripts
- ✅ Config loaded from YAML file

**Before your first commit:**
```bash
git status
# Verify databricks_config.yaml is NOT listed!
```

---

## 🐛 Troubleshooting

### "No such file: databricks_config.yaml"
```bash
# Create it from template:
cp config/databricks_config.yaml.template config/databricks_config.yaml
# Then edit with your credentials
```

### "MLflow authentication error"
- Check your token in `config/databricks_config.yaml`
- Verify token hasn't expired (90 days)
- Generate new token if needed

### "ModuleNotFoundError: No module named 'src'"
```bash
# Run from project root:
cd pizza-intelligence
python scripts/01_data_preparation_local.py
```

### "Data file not found"
```bash
# Make sure your Excel file is here:
ls data/raw/Data_Model_-_Pizza_Sales.xlsx
```

---

## 📞 Need Help?

- 📖 Read: [LOCAL_IDE_SETUP.md](LOCAL_IDE_SETUP.md) - Full setup guide
- 📊 Read: [DATABRICKS_PROJECT_SUMMARY.md](DATABRICKS_PROJECT_SUMMARY.md) - Project overview
- 🐛 Check: Troubleshooting section above
- 💬 Ask me for help!

---

## 🎉 You're Ready!

Everything is set up for **local IDE development** with **Databricks MLflow integration**.

**Start with:**
```bash
python scripts/01_data_preparation_local.py
```

**Then:**
```bash
python scripts/02_model_training_regression_local.py
```

**View results:**
- Local: `outputs/results/`
- MLflow: Databricks Experiments UI

---

**Want me to create the remaining 3 scripts?** Just ask! 🚀
