import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def evaluate_custom_objective(y_true, y_pred, y_prob):
    """Calculates the custom J score defined in the Stage 5/6 Report"""
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # J = 0.6 * Recall + 0.3 * Precision + 0.1 * ROCAUC
    j_score = (0.6 * recall) + (0.3 * precision) + (0.1 * roc_auc)
    
    return round(recall, 4), round(precision, 4), round(roc_auc, 4), round(j_score, 4)

def adaptive_imbalance_framework(X_train, y_train, X_test, y_test):
    """Automatically selects the model based on Imbalance Ratio (IR)"""
    
    n_majority = sum(y_train == 0)
    n_minority = sum(y_train == 1)
    
    # Calculate Imbalance Ratio (IR)
    if n_minority == 0:
        print("Error: No minority samples found in training data.")
        return
        
    ir = n_majority / n_minority
    print(f"Detected Imbalance Ratio (IR): {round(ir, 2)}")
    
    # AIHF Decision Logic (From Stage 5/6 Table 1)
    if ir < 2:
        print("Selected Method: Baseline Logistic Regression")
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
    elif 2 <= ir < 5:
        print("Selected Method: Class Weighting")
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train, y_train)
        
    elif 5 <= ir < 10:
        print("Selected Method: Threshold Tuning")
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # Custom threshold will be applied during prediction
        
    elif 10 <= ir < 20:
        print("Selected Method: SMOTE + Class Weighting")
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train_sm, y_train_sm)
        
    else: # IR >= 20
        print("Selected Method: Focal Loss (Simulated via Heavy Class Weighting)")
        # Simulating Focal Loss focus on minority class
        model = LogisticRegression(class_weight={0: 1, 1: 10})
        model.fit(X_train, y_train)

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    
    if 5 <= ir < 10:
        # Apply threshold tuning (e.g., 0.3)
        y_pred = (y_prob >= 0.3).astype(int)
    else:
        y_pred = model.predict(X_test)

    # Evaluation
    recall, precision, roc_auc, j_score = evaluate_custom_objective(y_test, y_pred, y_prob)
    
    print("\n--- AIHF Performance Results ---")
    print(f"Recall:    {recall}")
    print(f"Precision: {precision}")
    print(f"ROC-AUC:   {roc_auc}")
    print(f"J-Score:   {j_score}")

if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("data/sample.csv")
    X = data[["time", "vibration"]]
    y = data["label"]

    # Deployment split (Chronological)
    split_index = int(len(data) * 0.7)
    X_train_raw, X_test_raw = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Run the automated framework
    print("==================================================")
    print("INITIALIZING ADAPTIVE IMBALANCE HANDLING FRAMEWORK")
    print("==================================================")
    adaptive_imbalance_framework(X_train, y_train, X_test, y_test)