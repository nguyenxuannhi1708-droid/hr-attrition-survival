import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

import xgboost as xgb
import shap

# ========= 1. LOAD & PREPROCESS DATA =========

# Đọc file CSV (đúng tên file em đang có trong folder)
df = pd.read_csv("hr_data.csv")

# event: 1 = nghỉ việc, 0 = còn ở lại
df["event"] = (df["Attrition"] == "Yes").astype(int)
# duration: thời gian làm việc (tháng)
df["duration"] = df["YearsAtCompany"] * 12

# Bỏ các cột không cần thiết cho mô hình
drop_cols = ["EmployeeNumber", "Attrition", "EmployeeCount", "Over18", "StandardHours"]
drop_cols = [c for c in drop_cols if c in df.columns]
data = df.drop(columns=drop_cols)

# One-hot encoding cho biến phân loại
cat_cols = data.select_dtypes(include=["object"]).columns
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)

X = data_encoded
y_time = df["duration"].values
y_event = df["event"].values

X_train, X_test, ytime_train, ytime_test, yevent_train, yevent_test = train_test_split(
    X, y_time, y_event, test_size=0.3, random_state=42
)

print("Shape X:", X.shape)

# ========= 2. KAPLAN–MEIER (khám phá dữ liệu) =========

def plot_km_by_column(col_name, filename):
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()

    for label, grouped in df.groupby(col_name):
        kmf.fit(grouped["duration"], grouped["event"], label=str(label))
        kmf.plot_survival_function(ax=ax)

    plt.title(f"Survival curves theo {col_name}")
    plt.xlabel("Thời gian (tháng)")
    plt.ylabel("Xác suất còn ở lại")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Ví dụ: KM theo OverTime và Department
plot_km_by_column("OverTime", "km_overtime.png")
plot_km_by_column("Department", "km_department.png")
print("Đã lưu km_overtime.png và km_department.png")

# ========= 3. COX PROPORTIONAL HAZARDS (baseline) =========

train_df = X_train.copy()
train_df["duration"] = ytime_train
train_df["event"] = yevent_train

test_df = X_test.copy()
test_df["duration"] = ytime_test
test_df["event"] = yevent_test

cph = CoxPHFitter()
cph.fit(train_df, duration_col="duration", event_col="event")
cph.print_summary()

pred_partial_haz_train = cph.predict_partial_hazard(train_df)
pred_partial_haz_test = cph.predict_partial_hazard(test_df)

c_index_cox_train = concordance_index(
    train_df["duration"],
    -pred_partial_haz_train.values.ravel(),
    train_df["event"],
)
c_index_cox_test = concordance_index(
    test_df["duration"],
    -pred_partial_haz_test.values.ravel(),
    test_df["event"],
)

print("C-index Cox train:", round(c_index_cox_train, 3))
print("C-index Cox test :", round(c_index_cox_test, 3))

# ========= 4. RANDOM SURVIVAL FOREST (scikit-survival) =========

# ========= 4. RANDOM SURVIVAL FOREST (scikit-survival) =========

def make_y_struct(time, event):
    return np.array(
        [(bool(e), t) for e, t in zip(event, time)],
        dtype=[("event", "?"), ("time", "<f8")],
    )

y_train_struct = make_y_struct(ytime_train, yevent_train)
y_test_struct = make_y_struct(ytime_test, yevent_test)

rsf = RandomSurvivalForest(
    n_estimators=300,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

rsf.fit(X_train, y_train_struct)

# C-index train & test để xem overfitting
ci_rsf_train = concordance_index_censored(
    (1 - yevent_train).astype(bool),
    ytime_train,
    rsf.predict(X_train),
)[0]

ci_rsf_test = concordance_index_censored(
    (1 - yevent_test).astype(bool),
    ytime_test,
    rsf.predict(X_test),
)[0]

print("C-index RSF train:", round(ci_rsf_train, 3))
print("C-index RSF test :", round(ci_rsf_test, 3))

# AUC(t) & Brier
# ========= 4b. AUC(t) & Brier (nếu tính được) =========
times = np.array([12, 24, 36])  # tháng

surv_train = rsf.predict_survival_function(X_train)
surv_test = rsf.predict_survival_function(X_test)

surv_train_array = np.row_stack([fn(times) for fn in surv_train])
surv_test_array = np.row_stack([fn(times) for fn in surv_test])

chf_test = rsf.predict_cumulative_hazard_function(X_test)
chf_test_array = np.row_stack([fn(times) for fn in chf_test])

try:
    auc_scores, mean_auc = cumulative_dynamic_auc(
        y_train_struct, y_test_struct, chf_test_array, times
    )
    print("AUC(t):", list(zip(times, auc_scores)))
    print("Mean AUC:", mean_auc)

    ibs = integrated_brier_score(
        y_train_struct, y_test_struct, surv_test_array, times
    )
    print("Integrated Brier Score:", ibs)
except ValueError as e:
    print("Không tính được AUC(t)/Brier tại các mốc", times, "vì:", e)

# ========= 5. RISK TABLE 3 / 6 / 12 THÁNG =========

horizons = np.array([3, 6, 12])
surv_all = rsf.predict_survival_function(X)
surv_all_array = np.row_stack([fn(horizons) for fn in surv_all])
risk_probs = 1 - surv_all_array  # xác suất nghỉ trước mốc

risk_df = pd.DataFrame(
    {
        "EmployeeID": df["EmployeeNumber"],
        "Prob_leave_3m": risk_probs[:, 0],
        "Prob_leave_6m": risk_probs[:, 1],
        "Prob_leave_12m": risk_probs[:, 2],
    }
)

def risk_tier(p):
    if p >= 0.6:
        return "High"
    elif p >= 0.3:
        return "Medium"
    else:
        return "Low"

risk_df["RiskTier_6m"] = risk_df["Prob_leave_6m"].apply(risk_tier)

risk_df = risk_df.merge(
    df[["EmployeeNumber", "Department", "JobRole"]],
    left_on="EmployeeID",
    right_on="EmployeeNumber",
).drop(columns=["EmployeeNumber"])

print(risk_df.head())
risk_df.to_csv("employee_risk_table_3_6_12m.csv", index=False)
print("Đã lưu employee_risk_table_3_6_12m.csv")

# ========= 6. XGBOOST + SHAP (giải thích mô hình) =========

# Dùng lại X nhưng target là event (Attrition Yes/No)
y_clf = y_event

Xtr, Xte, ytr, yte = train_test_split(X, y_clf, test_size=0.3, random_state=42)

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

eval_set = [(Xtr, ytr), (Xte, yte)]
xgb_model.fit(Xtr, ytr, eval_set=eval_set, verbose=False)

results = xgb_model.evals_result()
train_logloss = results["validation_0"]["logloss"]
test_logloss = results["validation_1"]["logloss"]
epochs = range(1, len(train_logloss) + 1)

plt.figure()
plt.plot(epochs, train_logloss, label="Train logloss")
plt.plot(epochs, test_logloss, label="Test logloss")
plt.xlabel("Boosting rounds")
plt.ylabel("Logloss")
plt.legend()
plt.tight_layout()
plt.savefig("xgb_logloss_overfitting.png")
plt.close()
print("Đã lưu xgb_logloss_overfitting.png")

# SHAP summary plot cho XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(Xte)

plt.figure()
shap.summary_plot(shap_values, Xte, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()
print("Đã lưu shap_summary.png")
