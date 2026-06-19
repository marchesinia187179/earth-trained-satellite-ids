"""
Classification logic for evaluating trained models on test datasets.
"""
from datetime import datetime
import pandas as pd

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from utils.file_utils import get_data_from_csv, update_or_append_csv
from utils.paths import (
    INDEPENDENT_RESULTS_DIR, DEPENDENT_RESULTS_DIR,
    INDEPENDENT_RF_DIR, DEPENDENT_RF_DIR,
    INDEPENDENT_IF_DIR, DEPENDENT_IF_DIR,
    INDEPENDENT_DIR, DEPENDENT_DIR,
    RF_RESULTS_FILENAME, IF_RESULTS_FILENAME, # Result filenames
    ATTACK_CAT_DIR_NAME, NORMAL_ATTACK_DIR_NAME, # Subdirectory names
    NB15_PREPROCESSED_DIR_NAME, SAT20_PREPROCESSED_DIR_NAME, TER20_PREPROCESSED_DIR_NAME, # Preprocessed dir names
    PREPROCESSED_DIR_SUFFIX, JOINT_DIR_NAME,
    JOINT_NORMAL_SAT20_FILE_STEM, JOINT_NORMAL_TER20_FILE_STEM
)
import joblib


# --- Routine Classification Configuration ---
ROUTINE_CLASSIFICATIONS = []

for mode in ['independent', 'dependent']:
    # =========================================================================
    # RANDOM FOREST MATRIX ENGINES
    # =========================================================================
    
    # --- GRUPPO A: PURE DATASETS ---
    # Target SAT20 (Pure): Evaluate models 1-8 against zero-day satellite floods (Syn_DDoS, UDP_DDoS)
    for i in range(1, 9):
        for t_set in ['Syn_DDoS', 'UDP_DDoS']:
            ROUTINE_CLASSIFICATIONS.append({
                'mode': mode, 'model_type': 'random_forest', 'model_name': f'rf_model_{i}',
                'dataset_type': 'sat20', 'testing_dataset_name': t_set, 'data_subdir': ATTACK_CAT_DIR_NAME
            })

    # Target TER20 (Pure): Map cross-domain vulnerability patterns on non-STIN specific targets
    for i in range(1, 9):
        for t_set in ['Botnet', 'DDoS', 'Syn_DDoS', 'UDP_DDoS']:
            ROUTINE_CLASSIFICATIONS.append({
                'mode': mode, 'model_type': 'random_forest', 'model_name': f'rf_model_{i}',
                'dataset_type': 'ter20', 'testing_dataset_name': t_set, 'data_subdir': ATTACK_CAT_DIR_NAME
            })

    # Target NB15 (Pure): Internal multi-class baseline validation for global model (Model 6)
    for t_set in ['DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance']:
        ROUTINE_CLASSIFICATIONS.append({
            'mode': mode, 'model_type': 'random_forest', 'model_name': 'rf_model_6',
            'dataset_type': 'nb15', 'testing_dataset_name': t_set, 'data_subdir': ATTACK_CAT_DIR_NAME
        })

    # Target NB15 (Cross-Class): Map native cross-class evasion capabilities of specialized models (1-5)
    native_map = {1: 'DoS', 2: 'Exploits', 3: 'Fuzzers', 4: 'Generic', 5: 'Reconnaissance'}
    all_nb15_attacks = ['DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance']
    for i in range(1, 6):
        for t_set in all_nb15_attacks:
            if t_set != native_map[i]:
                ROUTINE_CLASSIFICATIONS.append({
                    'mode': mode, 'model_type': 'random_forest', 'model_name': f'rf_model_{i}',
                    'dataset_type': 'nb15', 'testing_dataset_name': t_set, 'data_subdir': ATTACK_CAT_DIR_NAME
                })

    # --- GRUPPO B: HYBRID / COMBO DATASETS (10:1 Ratio) ---
    # Target SAT20 Combo: Full metrics evaluation (F1/FPR) using balanced single-class satellite injections
    for i in range(1, 9):
        for t_set in ['Normal_Syn_DDoS', 'Normal_UDP_DDoS']:
            ROUTINE_CLASSIFICATIONS.append({
                'mode': mode, 'model_type': 'random_forest', 'model_name': f'rf_model_{i}',
                'dataset_type': 'nb15+sat20', 'testing_dataset_name': t_set, 'data_subdir': f"{JOINT_DIR_NAME}/sat20_joint"
            })

    # Target TER20 Combo: Operational threat identification matrices against single-class terrestrial injections
    for i in range(1, 9):
        for t_set in ['Normal_Botnet', 'Normal_DDoS', 'Normal_Syn_DDoS', 'Normal_UDP_DDoS']:
            ROUTINE_CLASSIFICATIONS.append({
                'mode': mode, 'model_type': 'random_forest', 'model_name': f'rf_model_{i}',
                'dataset_type': 'nb15+ter20', 'testing_dataset_name': t_set, 'data_subdir': f"{JOINT_DIR_NAME}/ter20_joint"
            })

    # Target STIN Cross-Domain Transferability: Cross-evaluate hybrid domain detectors on competing spaces
    # Model 7 (SAT Hybrid) on Terrestrial Aggregates
    ROUTINE_CLASSIFICATIONS.append({
        'mode': mode, 'model_type': 'random_forest', 'model_name': 'rf_model_7',
        'dataset_type': 'nb15+ter20', 'testing_dataset_name': JOINT_NORMAL_TER20_FILE_STEM, 'data_subdir': JOINT_DIR_NAME
    })
    # Model 8 (TER Hybrid) on Satellite Aggregates
    ROUTINE_CLASSIFICATIONS.append({
        'mode': mode, 'model_type': 'random_forest', 'model_name': 'rf_model_8',
        'dataset_type': 'nb15+sat20', 'testing_dataset_name': JOINT_NORMAL_SAT20_FILE_STEM, 'data_subdir': JOINT_DIR_NAME
    })

    # =========================================================================
    # ISOLATION FOREST MATRIX ENGINES (ANOMALY DETECTION)
    # =========================================================================
    
    # --- GRUPPO A: PURE TEST VARIANTS ---
    # Target SAT20 (Pure Test): Measure direct structural outlier variance on raw satellite streams
    for t_set in ['Syn_DDoS', 'UDP_DDoS']:
        ROUTINE_CLASSIFICATIONS.append({
            'mode': mode, 'model_type': 'isolation_forest', 'model_name': 'if_model_1',
            'dataset_type': 'sat20', 'testing_dataset_name': t_set, 'data_subdir': ATTACK_CAT_DIR_NAME
        })

    # Target TER20 (Pure Test): Assess unsupervised anomaly resolution thresholds without terrestrial labels
    for t_set in ['Botnet', 'DDoS', 'Syn_DDoS', 'UDP_DDoS']:
        ROUTINE_CLASSIFICATIONS.append({
            'mode': mode, 'model_type': 'isolation_forest', 'model_name': 'if_model_1',
            'dataset_type': 'ter20', 'testing_dataset_name': t_set, 'data_subdir': ATTACK_CAT_DIR_NAME
        })

    # Target NB15 (Pure Test): Internal anomaly alignment map and precise True False Positive Rate (FPR) verification on pure Normal
    for t_set in ['DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance']:
        ROUTINE_CLASSIFICATIONS.append({
            'mode': mode, 'model_type': 'isolation_forest', 'model_name': 'if_model_1',
            'dataset_type': 'nb15', 'testing_dataset_name': t_set, 'data_subdir': ATTACK_CAT_DIR_NAME
        })

    # --- GRUPPO B: OPERATIONAL COMBO SCENARIOS (10:1 Ratio) ---
    # Target SAT20 Combo: Benchmark real-world False Alarm Rates against satellite multi-source patterns
    for t_set in ['Normal_Syn_DDoS', 'Normal_UDP_DDoS']:
        ROUTINE_CLASSIFICATIONS.append({
            'mode': mode, 'model_type': 'isolation_forest', 'model_name': 'if_model_1',
            'dataset_type': 'nb15+sat20', 'testing_dataset_name': t_set, 'data_subdir': f"{JOINT_DIR_NAME}/sat20_joint"
        })

    # Target TER20 Combo: Evaluate contamination limits on unsupervised boundaries under terrestrial mixtures
    for t_set in ['Normal_Botnet', 'Normal_DDoS', 'Normal_Syn_DDoS', 'Normal_UDP_DDoS']:
        ROUTINE_CLASSIFICATIONS.append({
            'mode': mode, 'model_type': 'isolation_forest', 'model_name': 'if_model_1',
            'dataset_type': 'nb15+ter20', 'testing_dataset_name': t_set, 'data_subdir': f"{JOINT_DIR_NAME}/ter20_joint"
        })

    # Target STIN Cross-Domain Aggregato: Analyze global zero-day transference limits for pure zero-knowledge models
    for d_type, stem in [('sat20', JOINT_NORMAL_SAT20_FILE_STEM), ('ter20', JOINT_NORMAL_TER20_FILE_STEM)]:
        ROUTINE_CLASSIFICATIONS.append({
            'mode': mode, 'model_type': 'isolation_forest', 'model_name': 'if_model_1',
            'dataset_type': f"nb15+{d_type}", 'testing_dataset_name': stem, 'data_subdir': JOINT_DIR_NAME
        })


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
        file_path = base_dir / RF_RESULTS_FILENAME
        model_type_str = "random_forest"
    elif isinstance(model_obj, IsolationForest):
        file_path = base_dir / IF_RESULTS_FILENAME
        model_type_str = "isolation_forest"
    else:
        print(f"Error: Unknown model type for model '{model_name}'. Cannot save results.")
        return

    results_dict = {
        'id': None,
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
        'recall': metrics['recall'] if metrics['recall'] is not None else 'None',
        'auc_roc': metrics['auc_roc'] if metrics['auc_roc'] is not None else 'None'
    }

    match_keys = ['model_name', 'dataset_type', 'testing_dataset']
    update_or_append_csv(file_path, results_dict, match_keys)


def run_routine_classifications(pipeline_mode=None):
    """
    Executes a predefined set of classification tasks using specified models and datasets.
    If pipeline_mode is provided, it filters and runs ONLY the tasks for that specific mode.
    """
    print("\n--- Starting Routine Classification Phase ---")
    if not ROUTINE_CLASSIFICATIONS:
        print("No routine classification tasks defined. Exiting routine classification.")
        return

    for task in ROUTINE_CLASSIFICATIONS:
        mode = task['mode']
        
        # --- FIX: Salta i task che non appartengono alla modalità corrente ---
        if pipeline_mode is not None and mode != pipeline_mode:
            continue
        # ---------------------------------------------------------------------
            
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
        
        # Construct the preprocessed dataset directory name dynamically
        preprocessed_dataset_dir_name = f"{dataset_type}{PREPROCESSED_DIR_SUFFIX}"
        
        # Build the full path to the testing dataset
        if str(data_subdir).startswith(JOINT_DIR_NAME):
            data_path = data_base_dir / data_subdir / f"{testing_dataset_name}.csv"
        elif data_subdir: # If there's a specific subdirectory (e.g., attack_cat, normal_attack)
            data_path = data_base_dir / preprocessed_dataset_dir_name / data_subdir / f"{testing_dataset_name}.csv"
        else: # If the file is directly in the preprocessed dataset directory (e.g., nb15_preprocessed.csv)
            data_path = data_base_dir / preprocessed_dataset_dir_name / f"{testing_dataset_name}.csv"

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
    # Filter only test data to avoid data leakage from training
    test_data = data[data['split_type'] == 'test']

    if test_data.empty:
        print(f"Warning: No samples marked as 'test' found for {testing_dataset}. Skipping processing.")
        return

    # Prepare features and labels
    X = test_data.drop(columns=["label", "attack_cat", "split_type"])
    y = test_data["label"]

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
        recall = recall_score(y, y_pred) if has_both_classes else None

        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'f1': f1, 'precision': precision, 'recall': recall,
            'auc_roc': auc_roc
        }

        f1_display = f"{f1:.4f}" if f1 is not None else "None"
        precision_display = f"{precision:.4f}" if precision is not None else "None"
        recall_display = f"{recall:.4f}" if recall is not None else "None"
        auc_roc_display = f"{auc_roc:.4f}" if auc_roc is not None else "None"

        print(f"F1-score={f1_display}, Precision={precision_display}, Recall={recall_display}, AUC-ROC={auc_roc_display}")
        save_result(model_obj, mode, model_name, dataset_type, testing_dataset, test_data.shape[0], metrics)
