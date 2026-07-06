import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_cost_sensitive_model(y_true, y_prob, tp_benefit=5000, fp_cost=500, fn_cost=50000):
    """
    Solution 5: Economic Cost-Sensitive Learning
    
    This function evaluates a model based on real industrial costs rather than just F1-score.
    It tests multiple probability thresholds to find the one that yields the highest financial savings.
    
    Formula: Saving = (TP * tp_benefit) - (FP * fp_cost) - (FN * fn_cost)
    
    Parameters:
    - y_true: Actual labels (0 for normal, 1 for fault)
    - y_prob: Predicted probabilities for the fault class
    - tp_benefit: Financial benefit of correctly predicting a fault (True Positive)
    - fp_cost: Cost of unnecessary maintenance / false alarm (False Positive)
    - fn_cost: Cost of a catastrophic missed failure (False Negative)
    
    Returns:
    - best_threshold: The decision threshold that maximizes savings
    - max_savings: The maximum financial savings achieved
    - best_y_pred: The binary predictions at the optimal threshold
    """
    
    best_threshold = 0.5
    max_savings = -float('inf')
    best_y_pred = None
    best_cm = None
    
    # Test thresholds from 0.01 to 0.99 to find the most cost-effective decision boundary
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions based on current threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate True Positives, False Positives, False Negatives
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate total savings using the project's formula
        savings = (tp * tp_benefit) - (fp * fp_cost) - (fn * fn_cost)
        
        # If this threshold results in better savings, save it
        if savings > max_savings:
            max_savings = savings
            best_threshold = threshold
            best_y_pred = y_pred
            best_cm = (tn, fp, fn, tp)
            
    print(f"--- Cost-Sensitive Optimization Results ---")
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Maximized Savings: ${max_savings:,.2f}")
    print(f"Confusion Matrix (TN, FP, FN, TP): {best_cm}")
    
    return best_threshold, max_savings, best_y_pred