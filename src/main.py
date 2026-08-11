import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

# Import Solution 6
from solution5_cost_sensitive import evaluate_cost_sensitive_model

# =====================================================
# Load Dataset
# =====================================================

data = pd.read_csv("data/sample.csv")

X = data[["time", "vibration"]]
y = data["label"]

# =====================================================
# Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================================================
# Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# Helper Function
# =====================================================

def evaluate_model(name, y_true, y_pred, y_prob):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report")
    print(classification_report(y_true, y_pred))

    print("ROC-AUC:", round(roc_auc_score(y_true, y_prob), 4))


# =====================================================
# 1. BASELINE LOGISTIC REGRESSION
# =====================================================

print("\nRunning Baseline...")

model_base = LogisticRegression()

model_base.fit(X_train, y_train)

y_pred_base = model_base.predict(X_test)

y_prob_base = model_base.predict_proba(X_test)[:, 1]

evaluate_model(
    "1. BASELINE",
    y_test,
    y_pred_base,
    y_prob_base
)

# =====================================================
# 2. CLASS WEIGHTING
# =====================================================

print("\nRunning Class Weighting...")

model_weight = LogisticRegression(
    class_weight='balanced'
)

model_weight.fit(X_train, y_train)

y_pred_weight = model_weight.predict(X_test)

y_prob_weight = model_weight.predict_proba(X_test)[:, 1]

evaluate_model(
    "2. CLASS WEIGHTING",
    y_test,
    y_pred_weight,
    y_prob_weight
)

# =====================================================
# 3. SMOTE
# =====================================================

print("\nRunning SMOTE...")

smote = SMOTE(random_state=42)

X_train_sm, y_train_sm = smote.fit_resample(
    X_train,
    y_train
)

model_smote = LogisticRegression()

model_smote.fit(
    X_train_sm,
    y_train_sm
)

y_pred_smote = model_smote.predict(X_test)

y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

evaluate_model(
    "3. SMOTE",
    y_test,
    y_pred_smote,
    y_prob_smote
)

# =====================================================
# 4. THRESHOLD TUNING
# =====================================================

print("\nRunning Threshold Tuning...")

threshold = 0.30

y_pred_threshold = (
    y_prob_smote >= threshold
).astype(int)

evaluate_model(
    "4. THRESHOLD TUNING",
    y_test,
    y_pred_threshold,
    y_prob_smote
)

# =====================================================
# 5. FOCAL LOSS (SIMULATED)
# =====================================================

print("\nRunning Focal Loss (Simulated)...")

model_focal = LogisticRegression(
    class_weight={
        0: 1,
        1: 3
    }
)

model_focal.fit(
    X_train,
    y_train
)

y_pred_focal = model_focal.predict(X_test)

y_prob_focal = model_focal.predict_proba(X_test)[:, 1]

evaluate_model(
    "5. FOCAL LOSS (SIMULATED)",
    y_test,
    y_pred_focal,
    y_prob_focal
)

# =====================================================
# 6. ECONOMIC COST-SENSITIVE LEARNING
# =====================================================

print("\nRunning Economic Cost-Sensitive Learning...")

best_threshold, max_savings, best_y_pred = evaluate_cost_sensitive_model(
    y_true=y_test,
    y_prob=y_prob_focal,
    tp_benefit=5000,
    fp_cost=500,
    fn_cost=50000
)

evaluate_model(
    f"6. COST-SENSITIVE LEARNING (Threshold={best_threshold:.2f})",
    y_test,
    best_y_pred,
    y_prob_focal
)

print("\nFinancial Evaluation")
print("------------------------------------------")
print(f"Optimal Threshold : {best_threshold:.2f}")
print(f"Maximum Savings   : ${max_savings:,.2f}")

print("\n")
print("=" * 60)
print("ALL EXPERIMENTS COMPLETED")
print("=" * 60)
