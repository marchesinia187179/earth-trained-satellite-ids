"""
Classification logic for evaluating trained models on test datasets.
"""
import pathlib
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from utils.file_utils import append_data_to_csv, get_data_from_csv, get_model_info
from utils.paths import (
    INDEPENDENT_RESULTS_DIR, DEPENDENT_RESULTS_DIR,
    INDEPENDENT_RF_DIR, DEPENDENT_RF_DIR,
    INDEPENDENT_IF_DIR, DEPENDENT_IF_DIR,
    INDEPENDENT_DIR, DEPENDENT_DIR
)
import joblib


# --- Routine Classification Configuration ---
# This list defines the routine classification tasks.
# Each dictionary specifies a model, its mode (independent/dependent),
# the dataset type, the specific testing dataset name, and the subdirectory
# where the preprocessed testing dataset is expected to be found.
ROUTINE_CLASSIFICATIONS = [
    # --- DEPENDENT MODE ---
    # Random Forest (Models 1-5) on SAT20 and TER20
    *[{'mode': 'dependent', 'model_type': 'random_forest', 'model_name': f'rf_model_{i}', 'dataset_type': d_type, 'testing_dataset_name': t_set, 'data_subdir': 'attack_cat'}
      for i in range(1, 6)
      for d_type, t_sets in [('sat20', ['Syn_DDoS', 'UDP_DDoS']), ('ter20', ['Botnet', 'DDoS', 'Syn_DDoS', 'UDP_DDoS'])]
      for t_set in t_sets],

    # Isolation Forest (Model 1) on SAT20, TER20 and NB15
    *[{'mode': 'dependent', 'model_type': 'isolation_forest', 'model_name': 'if_model_1', 'dataset_type': 'sat20', 'testing_dataset_name': t_set, 'data_subdir': 'attack_cat'}
      for t_set in ['Syn_DDoS', 'UDP_DDoS']],
    *[{'mode': 'dependent', 'model_type': 'isolation_forest', 'model_name': 'if_model_1', 'dataset_type': 'ter20', 'testing_dataset_name': t_set, 'data_subdir': 'attack_cat'}
      for t_set in ['Botnet', 'DDoS', 'Syn_DDoS', 'UDP_DDoS']],
    *[{'mode': 'dependent', 'model_type': 'isolation_forest', 'model_name': 'if_model_1', 'dataset_type': 'nb15', 'testing_dataset_name': t_set, 'data_subdir': 'attack_cat'}
      for t_set in ['DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance']],
    *[{'mode': 'dependent', 'model_type': 'isolation_forest', 'model_name': 'if_model_1', 'dataset_type': 'nb15', 'testing_dataset_name': t_set, 'data_subdir': 'normal_attack'}
      for t_set in ['Normal', 'Normal_DoS', 'Normal_Exploits', 'Normal_Fuzzers', 'Normal_Generic', 'Normal_Reconnaissance']],

    # --- INDEPENDENT MODE ---
    # Random Forest (Models 1-5) on SAT20 and TER20
    *[{'mode': 'independent', 'model_type': 'random_forest', 'model_name': f'rf_model_{i}', 'dataset_type': d_type, 'testing_dataset_name': t_set, 'data_subdir': 'attack_cat'}
      for i in range(1, 6)
      for d_type, t_sets in [('sat20', ['Syn_DDoS', 'UDP_DDoS']), ('ter20', ['Botnet', 'DDoS', 'Syn_DDoS', 'UDP_DDoS'])]
      for t_set in t_sets],

    # Isolation Forest (Model 1) on SAT20, TER20 and NB15
    *[{'mode': 'independent', 'model_type': 'isolation_forest', 'model_name': 'if_model_1', 'dataset_type': 'sat20', 'testing_dataset_name': t_set, 'data_subdir': 'attack_cat'}
      for t_set in ['Syn_DDoS', 'UDP_DDoS']],
    *[{'mode': 'independent', 'model_type': 'isolation_forest', 'model_name': 'if_model_1', 'dataset_type': 'ter20', 'testing_dataset_name': t_set, 'data_subdir': 'attack_cat'}
      for t_set in ['Botnet', 'DDoS', 'Syn_DDoS', 'UDP_DDoS']],
    *[{'mode': 'independent', 'model_type': 'isolation_forest', 'model_name': 'if_model_1', 'dataset_type': 'nb15', 'testing_dataset_name': t_set, 'data_subdir': 'attack_cat'}
      for t_set in ['DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance']],
    *[{'mode': 'independent', 'model_type': 'isolation_forest', 'model_name': 'if_model_1', 'dataset_type': 'nb15', 'testing_dataset_name': t_set, 'data_subdir': 'normal_attack'}
      for t_set in ['Normal', 'Normal_DoS', 'Normal_Exploits', 'Normal_Fuzzers', 'Normal_Generic', 'Normal_Reconnaissance']],
]

# Flatten the list of lists created by comprehensions
ROUTINE_CLASSIFICATIONS = [item for sublist in ROUTINE_CLASSIFICATIONS for item in (sublist if isinstance(sublist, list) else [sublist])]


# --- Result Saving Functions ---
def save_result(model_obj, mode, model_name, dataset_type, testing_dataset, samples, metrics):
    """
    Saves classification results to a specific CSV file based on the model type.

    :param model_obj: The trained model object (e.g., RandomForestClassifier, IsolationForest).
    :param mode: Classification mode ('independent' or 'dependent').
    :param model_name: The name of the model.
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    :param testing_dataset: Name of the dataset used for testing.
    :param samples: Number of samples in the testing dataset.
    :param metrics: Dictionary containing performance metrics (tp, tn, fp, fn, f1, precision, recall, auc_roc).
    """
    base_dir = INDEPENDENT_RESULTS_DIR if mode == 'independent' else DEPENDENT_RESULTS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model_obj, RandomForestClassifier):
        file_path = base_dir / "random_forest_classification_results.csv"
        model_type_str = "random_forest"
    elif isinstance(model_obj, IsolationForest):
        file_path = base_dir / "isolation_forest_classification_results.csv"
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


def run_routine_classifications():
    """
    Executes a predefined set of classification tasks using specified models and datasets.
    This function centralizes routine classification for easy execution and updates.
    It assumes that both models and testing datasets are already preprocessed and saved
    in their respective directories.
    """
    print("\n--- Starting Routine Classification Phase ---")
    if not ROUTINE_CLASSIFICATIONS:
        print("No routine classification tasks defined. Exiting routine classification.")
        return

    for task in ROUTINE_CLASSIFICATIONS:
        mode = task['mode']
        model_type = task['model_type']
        model_name = task['model_name']
        dataset_type = task['dataset_type']
        testing_dataset_name = task['testing_dataset_name']
        data_subdir = task['data_subdir'] # Subdirectory within {dataset_type}_preprocessed

        print(f"\nRunning routine task: Mode={mode}, Model={model_name} ({model_type}), Dataset={dataset_type}/{data_subdir}/{testing_dataset_name}")

        # Determine model path
        model_base_dir = None
        if model_type == 'random_forest':
            model_base_dir = DEPENDENT_RF_DIR if mode == 'dependent' else INDEPENDENT_RF_DIR
        elif model_type == 'isolation_forest':
            model_base_dir = DEPENDENT_IF_DIR if mode == 'dependent' else INDEPENDENT_IF_DIR
        
        if model_base_dir is None:
            print(f"Error: Unknown model type '{model_type}'. Skipping this task.")
            continue

        model_path = model_base_dir / f"{model_name}.joblib"

        if not model_path.exists():
            print(f"Error: Model not found at {model_path}. Skipping this task.")
            continue
        
        try:
            model_obj = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model {model_name} from {model_path}: {e}. Skipping this task.")
            continue

        # Determine data path (assuming preprocessed data is stored)
        data_base_dir = DEPENDENT_DIR if mode == 'dependent' else INDEPENDENT_DIR
        data_path = data_base_dir / f"{dataset_type}_preprocessed" / data_subdir / f"{testing_dataset_name}.csv"

        if not data_path.exists():
            print(f"Error: Preprocessed data not found at {data_path}. Skipping this task.")
            continue
        
        try:
            data = get_data_from_csv(data_path)
        except Exception as e:
            print(f"Error loading data from {data_path}: {e}. Skipping this task.")
            continue

        models_to_test = [{'model_obj': model_obj, 'model_name': model_name}]
        classification_processing(data, mode, models_to_test, dataset_type, testing_dataset_name)
    print("\n--- Routine Classification Phase Completed ---")


# --- Main Processing Logic ---
def classification_processing(data, mode, models_to_test, dataset_type, testing_dataset="Unknown"):
    """
    Processes classification for one or more models on the given data and saves their metrics.

    :param data: The pandas DataFrame containing the testing data.
    :param mode: Classification mode ('independent' or 'dependent').
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
        save_result(model_obj, mode, model_name, dataset_type, testing_dataset, data.shape[0], metrics)
