"""
This module contains utility functions for calculating various classification metrics.
"""
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, 
    f1_score, precision_score, recall_score, roc_auc_score
)
from .config import MLConstants


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
    has_classes = len(np.unique(y_test)) > 1
    
    if not has_classes:
        print("Warning: the test data contains only one class. Only some metrics will be calculated!")

    # Get metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    
    accuracy = round(accuracy_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS) if has_classes else None
    precision = round(precision_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS) if has_classes else None
    recall = round(recall_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS) if has_classes else None
    f1 = round(f1_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS) if has_classes else None
    roc = round(roc_auc_score(y_test, y_scores), MLConstants.DECIMAL_DIGITS) if has_classes else None
    pr = round(average_precision_score(y_test, y_scores), MLConstants.DECIMAL_DIGITS) if has_classes else None
    
    # --- CORREZIONE: Spostato il costrutto 'if-else' all'esterno del round ---
    tpr = round(tp / (tp + fn), MLConstants.DECIMAL_DIGITS) if (tp + fn) > 0 else None
    fnr = round(fn / (tp + fn), MLConstants.DECIMAL_DIGITS) if (tp + fn) > 0 else None
    tnr = round(tn / (fp + tn), MLConstants.DECIMAL_DIGITS) if (fp + tn) > 0 else None
    fpr = round(fp / (fp + tn), MLConstants.DECIMAL_DIGITS) if (fp + tn) > 0 else None
    
    return {
        "TP": int(tp),   # True Positives: Number of actual attacks correctly identified as attacks
        "TN": int(tn),   # True Negatives: Number of normal traffic instances correctly identified as normal
        "FP": int(fp),   # False Positives: Number of normal traffic instances incorrectly flagged as attacks
        "FN": int(fn),   # False Negatives: Number of actual attacks that completely bypassed the model
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


def calculate_mean_and_variance(data, dataset_type, class_type):
    """
    Calculates the mean and variance of each feature in the dataset and add a column for the dataset type. 
    This is useful for understanding the distribution of features and for normalization purposes

    :param data: Input pandas DataFrame containing the dataset
    :param dataset_type: The type of the dataset
    :param class_type: The class of the dataset (e.g., 'normal', 'anomaly', or specific attack type)
    :return: A tuple containing two dictionaries: (mean_record, variance_record)
    """
    # Calculate stats first (isolating only numeric metrics to avoid parsing crashes)
    mean_values = data.mean(numeric_only=True).round(4).to_dict()
    variance_values = data.var(numeric_only=True).round(4).to_dict()

    # Reconstruct dictionaries forcing 'dataset_type' and 'class' as the first keys
    mean_record = {
        'dataset_type': dataset_type, 
        'class': class_type, 
        **mean_values
    }
    
    variance_record = {
        'dataset_type': dataset_type, 
        'class': class_type, 
        **variance_values
    }

    return mean_record, variance_record


if __name__ == "__main__":
    pass