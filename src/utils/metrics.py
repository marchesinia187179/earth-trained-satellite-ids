"""
This module contains utility functions for calculating various classification metrics.
"""
import pandas as pd
import numpy as np

from scipy import stats
from pathlib import Path
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


def calculate_welch_ttest_from_summary(data, model_type, alpha=MLConstants.WELCH_TTEST_ALFA_VALUE):
    """
    Computes Welch's t-test between the reference injection model and all other 
    models in the summary DataFrame across the different experimental seeds.

    The performance metrics (typically F1-Scores) are compared to verify if 
    the reference baseline achieves a statistically significant performance gain.

    :param data: pandas.DataFrame generated by aggregate_welch_ttest_feature_scores_by_seed
    :param model_type: string representing the classifier type (e.g., 'rf', 'dt', 'hgb')
    :param alpha: float significance level threshold for rejecting the null hypothesis
    :return: list of dictionaries containing calculated statistical test records
    """
    # Define the target label string representing the baseline reference model
    reference_model = f"{model_type.upper()} (Aggregate injection)"

    if reference_model not in data["model_name"].values:
        print(f"Error: Reference model '{reference_model}' not found in the DataFrame.")
        return []

    # Extract target statistical scores for the reference model across all seed columns
    # We drop structural identifier and description metadata to isolate pure metrics
    ref_row = data[data["model_name"] == reference_model]
    ref_scores = ref_row.drop(columns=["id", "model_name"]).values.flatten()

    print(f"\n=== Welch's t-test Results (Reference: {reference_model}) ===")
    print(f"Significance Level (Alpha): {alpha}\n")

    results_list = []

    # Iterate through each model entry in the summary dataframe for cross-testing
    for _, row in data.iterrows():
        current_model = row["model_name"]

        # Skip running self-comparison for the reference model
        if current_model == reference_model:
            continue

        # Isolate numerical test values for the compared model
        current_scores = row.drop(labels=["id", "model_name"]).values.astype(float)

        # Execute Welch's t-test (unequal variances assumed)
        t_stat, p_value = stats.ttest_ind(ref_scores, current_scores, equal_var=False)

        # Evaluate significance against target confidence threshold
        is_significant = "YES" if p_value < alpha else "NO"

        # Display results on console logging stream
        print(f"vs {current_model}:")
        print(f"  - t-statistic: {t_stat:.4f}")
        print(f"  - p-value:     {p_value:.6f}")
        print(f"  - Statistically Significant? {is_significant}")
        print("-" * 50)

        # Log results profile properties for downstream CSV generation
        results_list.append({
            "Reference_Model": reference_model,
            "Compared_Model": current_model,
            "t_statistic": t_stat,
            "p_value": p_value,
            "alpha": alpha,
            "is_significant": is_significant
        })

    return results_list


if __name__ == "__main__":
    pass