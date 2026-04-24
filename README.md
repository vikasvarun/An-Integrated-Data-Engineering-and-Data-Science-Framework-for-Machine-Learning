# 🤖 An Integrated Data Engineering and Data Science Framework for Machine Learning
### *Automated Machine Learning Platform with Interactive Model Selection*

> **B.Tech Major Project | 2026**  

---

## 📌 Overview

**An Integrated Data Engineering and Data Science Framework for Machine Learning** is a no-code, end-to-end machine learning web application that enables users to upload datasets, explore data, train and compare dozens of ML models, interpret results with SHAP explainability, and deploy predictions — all through an intuitive browser-based UI.

No ML expertise required. Just bring your CSV.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💎 Premium UI | Dark sidebar, responsive cards, clean professional design |
| 📤 CSV Upload | Drag-and-drop with instant schema preview & statistics |
| 🎯 Auto Task Detection | Automatically identifies Classification vs Regression |
| 🔧 Smart Preprocessing | Imputation, encoding, scaling, outlier removal |
| 📊 EDA Analytics | Correlation heatmaps, distribution plots, missing data analysis |
| 🚀 Multi-Model Training | Compare 15+ models simultaneously using PyCaret |
| 🔗 Ensemble Orchestrator | Voting, Stacking, and Blending ensemble creation |
| 🧠 Explainable AI | SHAP values + Feature Importance visualization |
| 🧪 Single & Batch Prediction | Real-time inference or CSV bulk prediction with download |

---

## 🗂️ Repository Structure

```
framework-ml/
│
├── APP.PY                   # Main Streamlit app (latest version)
├── APP2.PY                  # Alternate/experimental version
├── automl_app.py            # Earlier standalone version
├── README.md                # This file
│
├── 📊 Sample Dataset
│   └── diabetes.csv         # Pima Indians Diabetes dataset (classification demo)
│
├── 📄 Reference
│   └── conference_101719.pdf   # Related research / project paper
│
└── 🤖 Sample Trained Models
    └── *.pkl                # Sample models trained using this project
                             # (best single models + voting, stacking & blending ensembles)
```

> 💡 The `.pkl` files in this repo are real models trained through the platform's pipeline — saved automatically during development and testing sessions. They serve as proof-of-concept outputs and can be loaded directly for inference without retraining.

---

## 🧪 Try It Instantly with the Sample Dataset

The repo includes `diabetes.csv` — the classic **Pima Indians Diabetes** dataset — so you can test the full pipeline without needing your own data.

| Property | Value |
|---|---|
| Rows | 768 |
| Features | 8 numeric (Glucose, BMI, Age, etc.) |
| Target | `Outcome` — `0` (No Diabetes) / `1` (Diabetes) |
| Task Type | Binary Classification |

**Quickstart with this dataset:**
1. Run the app → **Upload Dataset** → upload `diabetes.csv`
2. **Column Selection** → set `Outcome` as target (auto-detects Classification)
3. **Preprocessing** → enable imputation + scaling → Apply
4. **Train Models** → Start Training → Use Best Single Model
5. **Results** → view AUC, Confusion Matrix, SHAP values
6. **Testing** → enter feature values → get a diabetes risk prediction

---

## ⚙️ Prerequisites

- Python **3.9 – 3.11** (PyCaret is not yet compatible with Python 3.12+)
- `pip` package manager
- Recommended: A virtual environment (`venv` or `conda`)

---

## 🚀 Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/framework-ml.git
cd framework-ml
```

### Step 2 — Create a Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ First-time install may take 5–10 minutes due to PyCaret's dependencies.

### Step 4 — Run the Application

```bash
# Latest version (recommended)
streamlit run APP.PY

# Or the alternate version
streamlit run APP2.PY
```

The app will open automatically at `http://localhost:8501`

---

## 📦 requirements.txt

Create this file in your project root:

```txt
streamlit>=1.32.0
pycaret[full]==3.2.0
pandas>=1.5.0
numpy>=1.23.0
plotly>=5.18.0
scikit-learn>=1.3.0
shap>=0.43.0
matplotlib>=3.7.0
```

---

## 🗺️ Application Workflow

The platform is structured as a **6-step linear pipeline**:

```
📤 Upload  →  🎯 Column Selection  →  🔧 Preprocessing & EDA
    →  🚀 Train Models  →  📊 Model Results  →  🧪 Testing & Deployment
```

---

### 📤 Page 1 — Upload Dataset

- Upload any `.csv` file (max **10 MB**)
- Instantly see: row/column counts, data types, missing value percentages
- Full summary statistics table
- Proceed to column selection when ready

---

### 🎯 Page 2 — Column Selection

- **Select Target Column** — the variable you want to predict
- The app **auto-detects** the problem type:
  - `object` dtype or `< 20 unique values` → **Classification**
  - Continuous numeric with `≥ 20 unique values` → **Regression**
- View target distribution with bar chart (classification) or histogram (regression)
- Choose features: use **all columns** (recommended) or **select specific ones**
- Class imbalance warnings shown if any class has < 10 samples

---

### 🔧 Page 3 — Preprocessing & EDA

Configure preprocessing options:

| Option | Effect |
|---|---|
| ✅ Impute missing values | Fills numeric NaN with **median**, categorical with **mode** |
| ✅ Encode categorical variables | Label/one-hot encoding via PyCaret |
| ✅ Normalize/Scale features | Standard scaling for all numeric features |
| 🎯 Remove outliers (IQR) | Drops rows outside `Q1 - 1.5×IQR` to `Q3 + 1.5×IQR` |
| 🎯 Remove small classes | Drops classes with fewer than N samples (classification only) |

**EDA Tabs:**
- **Missing Data** — bar chart of null counts
- **Correlation Matrix** — Pearson heatmap (interactive)
- **Distribution Analytics** — histogram + box plots per feature

---

### 🚀 Page 4 — Train Models

Configure training parameters:

| Setting | Range | Default |
|---|---|---|
| Cross-Validation Folds | 2–10 | 5 |
| Training Set Size | 60%–90% | 70% |
| Optimization Metric | Accuracy / AUC / F1 / R2 / MAE etc. | Accuracy / R2 |

Click **🚀 START TRAINING** — PyCaret will:
1. Run `setup()` to initialize the pipeline
2. Run `compare_models()` to benchmark all available models
3. Return a ranked leaderboard of top 10 models

**After training, choose:**
- ✅ **Use Best Single Model** — finalize and save the top performer
- 🔗 **Create Ensemble** — select 2–5 models and choose:
  - `Voting` / `Blending` → `blend_models()`
  - `Stacking` → `stack_models()` with a meta-learner

Models are auto-saved as `.pkl` files with a timestamp.

---

### 📊 Page 5 — Model Results

Four analysis tabs:

| Tab | Contents |
|---|---|
| 📊 Performance Comparison | Lollipop chart — top 5 models vs key metrics |
| 📈 Full Leaderboard | Highlighted comparison table for all trained models |
| 🧠 Explainable AI (SHAP) | SHAP global feature importance (falls back to standard feature importance if unavailable) |
| 🔍 Visual Insights | Confusion Matrix + AUC-ROC (classification) or Residuals + Prediction Error (regression) |

---

### 🧪 Page 6 — Testing & Deployment

**Single Prediction:**
- Fill in feature values using auto-generated number inputs and dropdowns
- See the predicted class/value instantly
- Classification shows confidence probabilities per class

**Batch Prediction:**
- Upload a new CSV with the same feature columns
- Download results with predictions appended as a new column
- Prediction distribution bar chart shown inline

---

## 🧠 Supported Models

PyCaret automatically benchmarks all applicable models. Examples include:

**Classification:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM, KNN, Naive Bayes, Extra Trees, AdaBoost, Ridge Classifier, and more.

**Regression:** Linear Regression, Lasso, Ridge, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, KNN, SVR, Huber, and more.

---

## 🐛 Common Issues & Fixes

**`least populated class` error during setup**
> The app automatically retries with `train_size=0.7` and reduced folds. For very small datasets, go to Preprocessing and enable "Remove small classes."

**SHAP visualization not showing**
> SHAP is not supported for all model types (e.g., Voting Classifier). The app automatically falls back to standard Feature Importance plots.

**PyCaret installation fails**
> Ensure you're on Python 3.9–3.11. Run `pip install pycaret[full]==3.2.0 --no-cache-dir`

**Streamlit rerun loop**
> This is expected behavior — `st.rerun()` is used for page navigation. It resolves automatically.

**File size error**
> The app enforces a 10MB limit. Reduce your CSV size or sample the dataset before uploading.

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning UI with `tune_model()`
- [ ] Time series forecasting support
- [ ] MLflow / experiment tracking integration
- [ ] Docker containerization
- [ ] REST API endpoint for model serving
- [ ] User authentication for multi-user deployment
