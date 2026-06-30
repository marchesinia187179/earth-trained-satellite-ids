"""
This module contains utility functions for calculating various classification metrics.
"""

from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from utils.config import MLConstants


# --- Public Functions ---
def calculate_metrics(y_test, y_pred, y_scores):
    """
    Internal helper to calculate metrics consistently

    :param y_test: ground truth (correct) target values containing actual classes (0 or 1)
    :param y_pred: estimated targets (0 or 1) returned by a classifier's predict method
    :param y_scores: predicted probabilities or decision scores for the positive class
    :return: a dictionary mapping metric names to their calculated and formatted values
    """

    # Security check for number of classes to avoid metric calculation error
    has_classes = len(y_test.unique()) > 1
    if not has_classes:
        print("Warrning: the test data contains only one class. Only some metrics will be calculated!")

    # Get metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = round(accuracy_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS) if has_classes else None
    precision = round(precision_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS) if has_classes else None
    recall = round(recall_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS)
    f1 = round(f1_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS) if has_classes else None
    roc = round(roc_auc_score(y_test, y_scores), MLConstants.DECIMAL_DIGITS) if has_classes else None
    pr = round(average_precision_score(y_test, y_scores), MLConstants.DECIMAL_DIGITS) if has_classes else None
    tpr = round(tp / (tp + fn) if (tp + fn) > 0 else None, MLConstants.DECIMAL_DIGITS)
    fnr = round(fn / (tp + fn) if (tp + fn) > 0 else None, MLConstants.DECIMAL_DIGITS)
    tnr = round(tn / (fp + tn) if (fp + tn) > 0 else None, MLConstants.DECIMAL_DIGITS)
    fpr = round(fp / (fp + tn) if (fp + tn) > 0 else None, MLConstants.DECIMAL_DIGITS)
    
    return {
        "TP": tp,   # True Positives: Number of actual attacks correctly identified as attacks
        "TN": tn,   # True Negatives: Number of normal traffic instances correctly identified as normal
        "FP": fp,   # False Positives: Number of normal traffic instances incorrectly flagged as attacks
        "FN": fn,   # False Negatives: Number of actual attacks that completely bypassed the model
        "Accuracy": accuracy,   # Accuracy: Ratio of correct predictions (both attacks and normal) over total instances
        "Precision": precision,     # Precision: Ratio of true attacks identified over total predicted attacks (measures false alarms)
        "Recall": recall,   # Recall: Ratio of true attacks identified over total actual attacks (same as TPR)
        "F1-Score": f1,     # F1-Score: Harmonic mean of Precision and Recall (balances false alarms and missed attacks)
        "ROC-AUC": roc,     # ROC-AUC: Ability of the model to distinguish between classes across all possible thresholds
        "PR-AUC": pr,   # PR-AUC: Average precision across all recall levels (highly critical for imbalanced attack data)
        "TPR": tpr,     # True Positive Rate: Ratio of positive predictions over Total Actual Positives
        "FNR": fnr,     # False Negative Rate: Ratio of actual positive predicted as negative over Total Actual Positives
        "TNR": tnr,     # Total Negative Rate: Ratio of negative predictions over Total Actual Negatives
        "FPR": fpr      # False Positive Rate: Ratio of actual negative predicted as positive over Total Actual Negatives
    }


if __name__ == "__main__":
    pass
