# Employee Attrition Prediction with Survival Analysis (IBM HR Analytics)

Final project for AI course — Group 8 

**Project focus:** proactive HR risk prediction using survival analysis + an interpretable classifier.

License: MIT

## 1) Project Description
This project builds an **AI co‑pilot for HR** that estimates **time‑to‑leave** and **short‑term attrition risk** for employees using the IBM HR Analytics sample dataset (1,470 rows, ~35 features). We combine **survival analysis** (Kaplan–Meier, Cox, Random Survival Forest) with a simple **classification baseline** (e.g., Logistic/XGBoost) and **explainable outputs** (SHAP/feature importances).

**Business goal:** move from reactive hiring to **proactive retention**, reducing surprise resignations and cost while keeping decisions fair and consistent.

## 2) Repository Structure
```
.
├─ src/
│  ├─ data_prep.py
│  ├─ survival_setup.py
│  ├─ train_survival.py
│  ├─ train_classifier.py
│  ├─ evaluate.py
│  └─ utils.py
├─ notebooks/
│  ├─ 01_EDA.ipynb
│  ├─ 02_KM_Curves.ipynb
│  └─ 03_Cox_RSF_Baselines.ipynb
├─ configs/
│  ├─ default.yaml
│  └─ classifier.yaml
├─ results/
│  ├─ figures/        # KM plots, ROC, SHAP
│  ├─ tables/         # risk tables (3/6/12 months)
│  └─ metrics.json
├─ models/            # saved models/checkpoints
├─ data/              # (optional) keep empty; add README with source link
│  └─ README.md
├─ requirements.txt
├─ environment.yml
├─ README.md
└─ LICENSE
```

## 3) Environment & Setup
Option A — **pip**
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Option B — **conda**
```bash
conda env create -f environment.yml
conda activate hr-attrition
```

Minimal packages (pin as needed): `pandas`, `numpy`, `scikit-learn`, `lifelines`, `scikit-survival` (or `sksurv`), `xgboost`, `matplotlib`, `seaborn`, `shap`.

## 4) Data Source
Dataset: **IBM HR Analytics – Employee Attrition & Performance** (synthetic, anonymized). Obtain from Kaggle or course drive, save as `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

## 5) Reproduce the Results
```bash
# 1) Data preparation and survival columns
python -m src.data_prep --input data/WA_Fn-UseC_-HR-Employee-Attrition.csv --output data/processed.csv

# 2) Survival problem setup (event, duration in months)
python -m src.survival_setup --input data/processed.csv --output data/survival.csv

# 3) Train survival models (KM/Cox/RSF)
python -m src.train_survival --config configs/default.yaml --data data/survival.csv --save_dir models/

# 4) Train classifier baseline (e.g., XGBoost/LogReg)
python -m src.train_classifier --config configs/classifier.yaml --data data/processed.csv --save_dir models/

# 5) Evaluate & export figures/tables
python -m src.evaluate --survival data/survival.csv --models_dir models/ --out_dir results/
```

Outputs include:
- **KM plots** by key factors (e.g., OverTime)
- **Risk table** at **3/6/12 months**
- **Model comparison** (C‑index / accuracy, ROC)
- **Explainability** (coefficients, feature importance, SHAP)

## 6) Results Summary (example placeholder)
- C‑index (Cox): 0.73; RSF: 0.76. - Classifier AUC‑ROC: 0.86; Accuracy: 0.84 on hold‑out. - OverTime, MonthlyIncome (low), JobLevel, DistanceFromHome, and JobSatisfaction are key signals.

> Replace the numbers with your team’s final metrics and add 1–2 figures in `results/figures`.

## 7) Notes / Limitations
- Dataset is **synthetic**; insights may not transfer directly to a real company.  
- Class imbalance requires careful thresholds; prefer **calibrated** probabilities.  
- Explanations should be used for guidance, not automated decisions.

## 8) How to Train Your Own Model
- Put your CSV in `data/` with similar columns.  
- Adjust `configs/*.yaml` for feature list, hyperparameters, and horizons (3/6/12 months).  
- Re‑run the five commands in Section 5.

## 9) Authors & Attribution
- Student: Nguyen Xuan Nhi (ID: 1137158) — Group 8  
- Reference: *IBM HR Analytics Employee Attrition & Performance* dataset by IBM (synthetic).

## 10) License
Choose a license (e.g., MIT) and add it as `LICENSE` in the repo.
