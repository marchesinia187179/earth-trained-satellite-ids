"""
This module contains utility functions for calculating various classification metrics.
"""
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from .config import MLConstants
from pathlib import Path
from scipy.stats import gaussian_kde


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
    real limits for both X (absolute min/max with padding) and Y (peak density), 
    and saves the results into a configuration CSV. 
    Hardcoded constraints (like forcing 0 for network metrics) have been removed.
    """
    config_path = Path(config_csv_path)
    if not config_path.exists():
        print(f"[ERROR] Configuration CSV file not found at: {config_csv_path}")
        return

    try:
        df_config = pd.read_csv(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to read configuration CSV: {e}")
        return

    if 'path' not in df_config.columns:
        print("[ERROR] The configuration CSV must contain a 'path' column.")
        return

    collected_data = {f: [] for f in features}
    
    print("Phase 1: Reading real data from dataset paths...")
    for _, row in df_config.iterrows():
        dataset_path_str = row['path']
        if pd.isna(dataset_path_str):
            continue
            
        file_path = Path(dataset_path_str)
        if not file_path.exists():
            print(f" -> [WARNING] Dataset file not found, skipping: {file_path}")
            continue

        try:
            df_preview = pd.read_csv(file_path, nrows=1)
            valid_cols = [f for f in features if f in df_preview.columns]
            
            if not valid_cols:
                continue
                
            df = pd.read_csv(file_path, usecols=valid_cols)
            
            for f in valid_cols:
                clean_series = df[f].replace([np.inf, -np.inf], np.nan).dropna()
                collected_data[f].extend(clean_series.tolist())
                
            print(f" -> Successfully processed: {file_path.name}")
        except Exception as e:
            print(f" -> Error reading {file_path.name}: {e}")

    print("\nPhase 2: Calculating purely data-driven X and Y limits...")
    limits_rows = []
    
    for f in features:
        data = np.array(collected_data[f])
        if len(data) == 0:
            continue
            
        # --- X Limits Calculation (No Hardcoded Constraints) ---
        min_val = np.min(data)
        max_robust = np.percentile(data, 100)
        
        # Calculate a 5% visual padding based on the actual range
        padding_x = (max_robust - min_val) * 0.05 if max_robust > min_val else 1.0
        
        # Unconditionally apply padding to both sides to allow the curve to breathe
        xmin = min_val - padding_x
        xmax = max_robust + padding_x
        
        # --- Y Limits Calculation (Density Peak) ---
        ymax = 1.0  # Default fallback
        try:
            # Subsample data if too large, to speed up KDE calculation
            sample_data = data if len(data) <= 10000 else np.random.choice(data, 10000, replace=False)
            
            # Compute the Gaussian KDE evaluated over a grid within our robust X range
            kde = gaussian_kde(sample_data)
            x_grid = np.linspace(xmin, xmax, 500)
            y_densities = kde.evaluate(x_grid)
            
            # The maximum density peak across all datasets
            peak_density = np.max(y_densities)
            
            # Add 30% padding to the top so the highest peak doesn't touch the upper border
            ymax = peak_density * 1.30
        except Exception as e:
            print(f" -> [WARNING] Could not compute Y limit for {f}, using automatic: {e}")
            ymax = None

        limits_rows.append({
            'feature': f,
            'xmin': xmin,
            'xmax': xmax,
            'ymax': ymax
        })
    
    if limits_rows:
        df_limits = pd.DataFrame(limits_rows)
        df_limits.to_csv(output_csv_path, index=False)
        print(f"\n[OK] KDE X and Y limits successfully saved to: {output_csv_path}")


if __name__ == "__main__":
    pass
