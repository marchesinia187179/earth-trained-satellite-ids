import pathlib
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from utils.file_utils import append_data_to_csv, get_data_from_csv
from utils.paths import RF_CLASSIFICATION_RESULTS_CSV, IF_CLASSIFICATION_RESULTS_CSV


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
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall']
    }
    if 'auc_roc' in metrics:
        results_dict['auc_roc'] = metrics['auc_roc']

    append_data_to_csv(results_dict, file_path)
    print(f"Classification report appended to: {file_path.name}")


def classification_processing(data, models_to_test, dataset_type, testing_dataset="Unknown"):
    """
    Processes classification for one or more models on the given data and saves their metrics.

    :param data: The pandas DataFrame containing the testing data.
    :param models_to_test: A list of dictionaries, where each dictionary contains
                           {'model_obj': model_instance, 'model_name': 'model_identifier'}.
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    :param testing_dataset: Name of the dataset used for testing.
    """
    X = data.drop(columns=["label", "attack_cat"])
    y = data["label"]

    for model_entry in models_to_test:
        model_obj = model_entry['model_obj']
        model_name = model_entry['model_name']

        print(f"\nExecuting classification for model: {model_name} (Test Data: {testing_dataset})...")

        y_pred = model_obj.predict(X)

        # Special handling for Isolation Forest predictions
        if isinstance(model_obj, IsolationForest):
            # IsolationForest returns 1 for inliers (normal) and -1 for outliers (anomalies)
            # We map 1 -> 0 (normal) and -1 -> 1 (anomaly) to align with standard classification labels
            y_pred = np.where(y_pred == 1, 0, 1)
            y_scores = -model_obj.decision_function(X) # Decision function for AUC-ROC
            auc_roc = roc_auc_score(y, y_scores)
        else:
            auc_roc = None # AUC-ROC is typically for anomaly detection or binary classification with probabilities

        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'f1': f1, 'precision': precision, 'recall': recall,
            'auc_roc': auc_roc
        }

        print(f"F1-score={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        save_result(model_obj, model_name, dataset_type, testing_dataset, data.shape[0], metrics)
