"""
Classification logic for evaluating trained models on test datasets.
"""
import pathlib
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from utils.file_utils import append_data_to_csv, get_data_from_csv
from utils.paths import RF_CLASSIFICATION_RESULTS_CSV, IF_CLASSIFICATION_RESULTS_CSV


# --- Result Saving Functions ---
def save_result(model_obj, model_name, dataset_type, testing_dataset, samples, metrics):
    """
    Saves classification results to a specific CSV file based on the model type.

    :param model_obj: The trained model object (e.g., RandomForestClassifier, IsolationForest).
    :param model_name: The name of the model.
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    :param testing_dataset: Name of the dataset used for testing.
    :param samples: Number of samples in the testing dataset.
    :param metrics: Dictionary containing performance metrics (tp, tn, fp, fn, f1, precision, recall, auc_roc).
    """
    if isinstance(model_obj, RandomForestClassifier):
        file_path = RF_CLASSIFICATION_RESULTS_CSV
        model_type_str = "random_forest"
    elif isinstance(model_obj, IsolationForest):
        file_path = IF_CLASSIFICATION_RESULTS_CSV
        model_type_str = "isolation_forest"
    else:
        print(f"Error: Unknown model type for model '{model_name}'. Cannot save results.")
        return

    # Determine the ID for the new entry
    if file_path.exists():
        try:
            existing_df = get_data_from_csv(file_path)
            id = existing_df.shape[0] + 1
        except Exception: # Handle cases where CSV might be empty or corrupted
            id = 1
    else:
        id = 1
    
    results_dict = {
        'id': id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': model_type_str,
        'model_name': model_name,
        'dataset_type': dataset_type,
        'testing_dataset': testing_dataset,
        'samples': samples,
        'tp': metrics['tp'],
        'tn': metrics['tn'],
        'fp': metrics['fp'],
        'fn': metrics['fn'],
        'f1': metrics['f1'] if metrics['f1'] is not None else 'None',
        'precision': metrics['precision'] if metrics['precision'] is not None else 'None',
        'recall': metrics['recall'],
        'auc_roc': metrics.get('auc_roc') if metrics.get('auc_roc') is not None else 'None'
    }

    append_data_to_csv(results_dict, file_path)
    print(f"Classification report appended to: {file_path.name}")


# --- Main Processing Logic ---
def classification_processing(data, models_to_test, dataset_type, testing_dataset="Unknown"):
    """
    Processes classification for one or more models on the given data and saves their metrics.

    :param data: The pandas DataFrame containing the testing data.
    :param models_to_test: A list of dictionaries, where each dictionary contains
                           {'model_obj': model_instance, 'model_name': 'model_identifier'}.
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    :param testing_dataset: Name of the dataset used for testing.
    """
    # Prepare features and labels
    X = data.drop(columns=["label", "attack_cat"])
    y = data["label"]

    for model_entry in models_to_test:
        model_obj = model_entry['model_obj']
        model_name = model_entry['model_name']

        print(f"\nExecuting classification for model: {model_name} (Test Data: {testing_dataset})...")

        y_pred = model_obj.predict(X)
        auc_roc = None

        # AUC-ROC requires the presence of both classes in the test set
        unique_classes = np.unique(y)
        has_both_classes = len(unique_classes) > 1

        if not has_both_classes:
            print(f"Warning: Testing dataset '{testing_dataset}' contains only one class ({unique_classes}). "
                  f"AUC-ROC cannot be calculated and will be set to 'None'.")

        # Special handling for Isolation Forest predictions
        if isinstance(model_obj, IsolationForest):
            # Map IsolationForest predictions: 1 -> 0 (normal), -1 -> 1 (anomaly)
            y_pred = np.where(y_pred == 1, 0, 1)
            if has_both_classes:
                y_scores = -model_obj.decision_function(X)
                auc_roc = roc_auc_score(y, y_scores)
        # Check if it is a RandomForest or has predict_proba method (more robust)
        elif hasattr(model_obj, "predict_proba"):
            if has_both_classes:
                y_scores = model_obj.predict_proba(X)[:, 1]
                auc_roc = roc_auc_score(y, y_scores)

        # Calculate metrics
        f1 = f1_score(y, y_pred) if has_both_classes else None
        precision = precision_score(y, y_pred) if has_both_classes else None
        recall = recall_score(y, y_pred)

        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'f1': f1, 'precision': precision, 'recall': recall,
            'auc_roc': auc_roc
        }

        f1_display = f"{f1:.4f}" if f1 is not None else "None"
        precision_display = f"{precision:.4f}" if precision is not None else "None"
        auc_roc_display = f"{auc_roc:.4f}" if auc_roc is not None else "None"
        print(f"F1-score={f1_display}, Precision={precision_display}, Recall={recall:.4f}, AUC-ROC={auc_roc_display}")
        save_result(model_obj, model_name, dataset_type, testing_dataset, data.shape[0], metrics)
