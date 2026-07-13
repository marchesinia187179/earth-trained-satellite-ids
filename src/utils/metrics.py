"""
This module contains utility functions for calculating various classification metrics.
"""
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from .config import MLConstants
from pathlib import Path


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
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    
    accuracy = round(accuracy_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS) if has_classes else None
    precision = round(precision_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS) if has_classes else None
    recall = round(recall_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS)
    f1 = round(f1_score(y_test, y_pred), MLConstants.DECIMAL_DIGITS) if has_classes else None
    roc = round(roc_auc_score(y_test, y_scores), MLConstants.DECIMAL_DIGITS) if has_classes else None
    pr = round(average_precision_score(y_test, y_scores), MLConstants.DECIMAL_DIGITS) if has_classes else None
    
    # --- CORREZIONE: Spostato il costrutto 'if-else' all'esterno del round ---
    tpr = round(tp / (tp + fn), MLConstants.DECIMAL_DIGITS) if (tp + fn) > 0 else None
    fnr = round(fn / (tp + fn), MLConstants.DECIMAL_DIGITS) if (tp + fn) > 0 else None
    tnr = round(tn / (fp + tn), MLConstants.DECIMAL_DIGITS) if (fp + tn) > 0 else None
    fpr = round(fp / (fp + tn), MLConstants.DECIMAL_DIGITS) if (fp + tn) > 0 else None
    
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


def calculate_mean_and_variance(data, dataset_type, class_type):
    """
    Calculates the mean and variance of each feature in the dataset and add a column for the dataset type. 
    This is useful for understanding the distribution of features and for normalization purposes

    :param data: Input pandas DataFrame containing the dataset
    :param dataset_type: The type of the dataset
    :param class_type: The class of the dataset (e.g., 'normal', 'anomaly', or specific attack type)
    :return: A tuple containing two pandas Series: (mean, variance) for each feature
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


def calculate_kde_limits_csv(config_csv_path, features, output_csv_path):
    """
    Reads a configuration CSV file containing dataset paths, calculates optimal 
    real limits using robust percentiles to avoid outlier distortion, and saves 
    the results into a configuration CSV.
    
    :param config_csv_path: Path to the configuration CSV file that includes a 'path' column
    :param features: List of feature names to analyze (e.g., ['pkts_per_sec', 'total_bytes', 'dst_win_byt'])
    :param output_csv_path: Target path for the generated limits CSV file
    """
    config_path = Path(config_csv_path)
    if not config_path.exists():
        print(f"[ERROR] Configuration CSV file not found at: {config_csv_path}")
        return

    # Load the configuration file
    try:
        df_config = pd.read_csv(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to read configuration CSV: {e}")
        return

    if 'path' not in df_config.columns:
        print("[ERROR] The configuration CSV must contain a 'path' column.")
        return

    # Initialize a dictionary to pool all values for each feature across datasets
    collected_data = {f: [] for f in features}
    
    print("Phase 1: Reading real data from dataset paths...")
    for _, row in df_config.iterrows():
        dataset_path_str = row['path']
        if pd.isna(dataset_path_str):
            continue
            
        file_path = Path(dataset_path_str)
        
        # Check if the dataset file actually exists
        if not file_path.exists():
            print(f" -> [WARNING] Dataset file not found, skipping: {file_path}")
            continue

        try:
            # Preview columns first to read only the required ones and save memory
            df_preview = pd.read_csv(file_path, nrows=1)
            valid_cols = [f for f in features if f in df_preview.columns]
            
            if not valid_cols:
                print(f" -> [SKIP] No target features found in {file_path.name}")
                continue
                
            df = pd.read_csv(file_path, usecols=valid_cols)
            
            for f in valid_cols:
                # Remove NaN and infinite values before analysis
                clean_series = df[f].replace([np.inf, -np.inf], np.nan).dropna()
                collected_data[f].extend(clean_series.tolist())
                
            print(f" -> Successfully processed: {file_path.name}")
        except Exception as e:
            print(f" -> Error reading {file_path.name}: {e}")

    print("\nPhase 2: Calculating optimal robust limits...")
    limits_rows = []
    
    for f in features:
        data = np.array(collected_data[f])
        if len(data) == 0:
            print(f" -> [WARNING] No valid data collected for feature: {f}")
            continue
            
        # Determine the real minimum (force 0 for network metrics that cannot be negative)
        min_val = np.min(data)
        if f in ['pkts_per_sec', 'total_bytes'] and min_val < 0:
            min_val = 0.0
            
        # Calculate a robust maximum using the 99.5th percentile.
        # This drops the top 0.5% of extreme outliers that distort the KDE scale.
        max_robust = np.percentile(data, 99.5)
        
        # Add a 5% visual padding so the curves don't clip against the plot borders
        padding = (max_robust - min_val) * 0.05 if max_robust > min_val else 1.0
        
        xmin = min_val - (padding if min_val > 0 else 0)
        xmax = max_robust + padding
        
        limits_rows.append({
            'feature': f,
            'xmin': xmin,
            'xmax': xmax
        })
    
    # Export results to a persistent CSV configuration file
    if limits_rows:
        df_limits = pd.DataFrame(limits_rows)
        df_limits.to_csv(output_csv_path, index=False)
        print(f"\n[OK] KDE limits successfully saved to: {output_csv_path}")
    else:
        print("\n[FAILED] No limits calculated. Please check your dataset files and feature list.")


if __name__ == "__main__":
    pass
