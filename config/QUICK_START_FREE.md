# 🚀 Quick Start - Databricks Community Edition (FREE)

**Perfect! You have the free version. Everything is already set up!**

---

## ✅ Your Setup (2 Steps)

### Step 1: Update Your Email in Config

Open the file you already have open: `config/databricks_config.yaml`

Find this line:
```yaml
experiment_name: "/Users/your_email@example.com/pizza-intelligence"
```

Change it to YOUR email:
```yaml
experiment_name: "/Users/YOUR_ACTUAL_EMAIL@asu.edu/pizza-intelligence"
```

Example:
```yaml
experiment_name: "/Users/john.smith@asu.edu/pizza-intelligence"
```

**That's it for config!** ✅

### Step 2: Run First Script

```bash
pip install pyyaml
python scripts/01_data_preparation_local.py
```

---

## 🎯 What Happens

```
Your Laptop (FREE!)           Databricks (FREE!)
─────────────────             ─────────────────

📊 Load Excel data
🔧 Create features
💾 Save locally
🤖 Train models               ──────────────►  📝 Log to MLflow
📁 Save results                              📊 Track experiments
                                             🏆 Model Registry

Everything FREE!              Everything FREE!
```

**Benefits:**
- ✅ No clusters needed
- ✅ Runs on your laptop
- ✅ Zero costs
- ✅ Full MLflow tracking
- ✅ Perfect for your project!

---

## 📋 All 5 Scripts Ready

Run them in order:

```bash
# 1. Data Preparation (~1 min)
python scripts/01_data_preparation_local.py

# 2. Base Models (~10 min)
python scripts/02_model_training_regression_local.py

# 3. Hyperparameter Tuning (~60 min)
python scripts/03_hyperparameter_tuning_regression_local.py

# 4. Classification (~10 min)
python scripts/04_model_training_classification_local.py

# 5. Clustering & Bundles (~10 min)
python scripts/05_clustering_association_local.py
```

**Total: ~90 minutes**

---

## 📊 View Results

### Local Files (Immediate)
```
outputs/results/
├── base_models_regression.csv
├── tuned_models_regression.csv
├── classification_category.csv
├── pizza_clusters.csv
└── association_rules.csv
```

### Databricks MLflow (Cloud)
1. Go to: https://community.cloud.databricks.com/
2. Click **Machine Learning** → **Experiments**
3. Find your experiment
4. View all runs, metrics, models!

---

## 🎉 What You Get

- ✅ **30+ models** evaluated
- ✅ **540+ hyperparameter** combinations tested
- ✅ **DBSCAN** clustering
- ✅ **Support, Confidence, Lift, Leverage** calculated
- ✅ **Best model** in Model Registry
- ✅ **All experiments** tracked in MLflow
- ✅ **Zero costs!**

---

## 💡 Pro Tip

Run hyperparameter tuning overnight:
```bash
# Start before bed
nohup python scripts/03_hyperparameter_tuning_regression_local.py > tuning.log 2>&1 &

# Check progress in morning
tail -f tuning.log
```

---

## ❓ Need Help?

**Quick troubleshooting:**
- Not working? Check token in config file
- Can't find results? Check `outputs/results/` folder
- MLflow empty? Verify experiment name has YOUR email

**Detailed guides:**
- [DATABRICKS_FREE_VERSION_GUIDE.md](DATABRICKS_FREE_VERSION_GUIDE.md) - Complete free version guide
- [SCRIPTS_COMPLETE.md](SCRIPTS_COMPLETE.md) - All scripts explained
- [LOCAL_IDE_SETUP.md](LOCAL_IDE_SETUP.md) - Full setup details

---

**You're all set! Just update your email and run the scripts.** 🚀

```bash
python scripts/01_data_preparation_local.py
```
