"""
Model training and management for Random Forest and Isolation Forest.
"""
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from utils.file_utils import append_data_to_csv, get_data_from_csv
from utils.paths import (
    INDEPENDENT_RF_DIR, DEPENDENT_RF_DIR,
    INDEPENDENT_IF_DIR, DEPENDENT_IF_DIR,
    INDEPENDENT_DIR, DEPENDENT_DIR, # Base directories for preprocessed data
    RF_INFO_FILENAME, IF_INFO_FILENAME,
    NORMAL_ATTACK_DIR_NAME, JOINT_DIR_NAME, # Subdirectory names
    NB15_PREPROCESSED_DIR_NAME, # Specific preprocessed dataset directory names
    NB15_SCALED_FILE_STEM, # Specific preprocessed filenames (stem)
    JOINT_NORMAL_SAT20_FILE_STEM, JOINT_NORMAL_TER20_FILE_STEM, # Joint filenames (stem)
    PREPROCESSED_MAIN_FILE_SUFFIX # Suffix for main preprocessed file (stem)
)


# --- Routine Model Configuration ---
# Defines the standard set of models to be built during a routine phase.
ROUTINE_MODELS = [
    # Random Forest models on NB15 sub-datasets
    {'model_type': 'random forest', 'dataset_type': 'nb15', 'file_rel_path': f'{NB15_PREPROCESSED_DIR_NAME}/{NORMAL_ATTACK_DIR_NAME}/Normal_DoS.csv'},
    {'model_type': 'random forest', 'dataset_type': 'nb15', 'file_rel_path': f'{NB15_PREPROCESSED_DIR_NAME}/{NORMAL_ATTACK_DIR_NAME}/Normal_Exploits.csv'},
    {'model_type': 'random forest', 'dataset_type': 'nb15', 'file_rel_path': f'{NB15_PREPROCESSED_DIR_NAME}/{NORMAL_ATTACK_DIR_NAME}/Normal_Fuzzers.csv'},
    {'model_type': 'random forest', 'dataset_type': 'nb15', 'file_rel_path': f'{NB15_PREPROCESSED_DIR_NAME}/{NORMAL_ATTACK_DIR_NAME}/Normal_Generic.csv'},
    {'model_type': 'random forest', 'dataset_type': 'nb15', 'file_rel_path': f'{NB15_PREPROCESSED_DIR_NAME}/{NORMAL_ATTACK_DIR_NAME}/Normal_Reconnaissance.csv'},
    {'model_type': 'random forest', 'dataset_type': 'nb15', 'file_rel_path': f'{NB15_PREPROCESSED_DIR_NAME}/{NB15_SCALED_FILE_STEM}.csv'},
    # Random Forest models on Joint datasets
    {'model_type': 'random forest', 'dataset_type': 'nb15+sat20', 'file_rel_path': f'{JOINT_DIR_NAME}/{JOINT_NORMAL_SAT20_FILE_STEM}.csv'},
    {'model_type': 'random forest', 'dataset_type': 'nb15+ter20', 'file_rel_path': f'{JOINT_DIR_NAME}/{JOINT_NORMAL_TER20_FILE_STEM}.csv'},
    # Isolation Forest on NB15
    {'model_type': 'isolation_forest', 'dataset_type': 'nb15', 'file_rel_path': f'{NB15_PREPROCESSED_DIR_NAME}/{NB15_PREPROCESSED_DIR_NAME}{PREPROCESSED_MAIN_FILE_SUFFIX}.csv'}
]


# --- Internal Helper Functions ---
def _calculate_metrics(y_true, y_pred, y_scores):
    """Internal helper to calculate metrics consistently."""
    unique_classes = np.unique(y_true)
    has_both = len(unique_classes) > 1

    f1 = f1_score(y_true, y_pred) if has_both else None
    precision = precision_score(y_true, y_pred) if has_both else None
    recall = recall_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_scores) if has_both else None

    # Fix: labels=[0, 1] ensures a 2x2 matrix even if classes are missing
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'f1': f1, 'precision': precision, 'recall': recall,
        'auc_roc': auc_roc
    }


def _save_model_and_metadata(model, model_type, mode, train_ratio, training_dataset, dataset_type, samples, metrics):
    """Simplifies and unifies saving for RF and IF models."""
    if model_type == 'rf':
        save_dir = INDEPENDENT_RF_DIR if mode == 'independent' else DEPENDENT_RF_DIR
        prefix = 'rf_model'
        csv_name = RF_INFO_FILENAME
    else:
        save_dir = INDEPENDENT_IF_DIR if mode == 'independent' else DEPENDENT_IF_DIR
        prefix = 'if_model'
        csv_name = IF_INFO_FILENAME

    save_dir.mkdir(parents=True, exist_ok=True)
    info_csv = save_dir / csv_name

    existing_files = list(save_dir.glob(f'{prefix}_*.joblib'))
    model_name = f'{prefix}_{len(existing_files) + 1}'
    joblib.dump(model, save_dir / f'{model_name}.joblib')

    params = model.get_params()
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'dataset_type': dataset_type,
        'training_dataset': training_dataset,
        'samples': samples,
        'train_ratio': train_ratio,
        'n_estimators': params['n_estimators'],
        'max_features': params['max_features'],
        'random_state': params.get('random_state', None),
        'n_features_in': int(model.n_features_in_),
    }

    # Type-specific metadata
    if model_type == 'rf':
        results.update({
            'criterion': params.get('criterion', 'gini'),
            'max_depth': params.get('max_depth', None),
            'n_classes': len(model.classes_)
        })
    else:
        results.update({
            'max_samples': params['max_samples'],
            'contamination': params['contamination'],
            'model_offset': float(model.offset_)
        })

    # Metrics formatting (None -> 'None' as requested)
    results.update({
        'tp': metrics['tp'], 'tn': metrics['tn'], 'fp': metrics['fp'], 'fn': metrics['fn'],
        'f1': metrics['f1'] if metrics['f1'] is not None else 'None',
        'precision': metrics['precision'] if metrics['precision'] is not None else 'None',
        'recall': metrics['recall'] if metrics['recall'] is not None else 'None',
        'auc_roc': metrics['auc_roc'] if metrics['auc_roc'] is not None else 'None'
    })

    append_data_to_csv(results, info_csv)
    print(f"Model {model_name} saved to {save_dir.name} and metadata to {info_csv.name}")


# --- Public Training Functions ---
def isolation_forest(data, mode, dataset_type, training_dataset, train_ratio=0.8):
    """
    Trains an Isolation Forest model using normal data and evaluates it on a mixed set.

    :param data: Input DataFrame containing both normal and anomaly samples.
    :param mode: Preprocessing mode used ('independent' or 'dependent').
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    :param training_dataset: Name of the original dataset file.
    :param train_ratio: Proportion of normal data used for training.
    """
    print(f"Training Isolation Forest...")
    
    # Use split_type column to separate data
    train_normal = data[(data['label'] == 0) & (data['split_type'] == 'train')]
    test_normal = data[(data['label'] == 0) & (data['split_type'] == 'test')]
    
    # For Anomaly test set, use only data marked as 'test'
    data_anomaly_test = data[(data['label'] == 1) & (data['split_type'] == 'test')]
    n_anomaly_sample = min(len(data_anomaly_test), int(len(test_normal) * 0.10))
    test_anomaly = data_anomaly_test.sample(n=n_anomaly_sample, random_state=42)

    data_test = pd.concat([test_normal, test_anomaly])

    X_train = train_normal.drop(columns=['attack_cat', 'label', 'split_type'])
    X_test = data_test.drop(columns=['attack_cat', 'label', 'split_type'])
    y_test = data_test['label']

    model = IsolationForest(random_state=42, verbose=3)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred == 1, 0, 1)
    
    y_scores = -model.decision_function(X_test)
    metrics = _calculate_metrics(y_test, y_pred, y_scores)

    _save_model_and_metadata(model, 'if', mode, train_ratio, training_dataset, dataset_type, X_train.shape[0], metrics)
    print("Training process done.")


def random_forest(data, mode, dataset_type, training_dataset, train_ratio=0.8):
    """
    Trains a Random Forest classifier using a supervised learning approach.

    :param data: Input DataFrame containing features and labels.
    :param mode: Preprocessing mode used ('independent' or 'dependent').
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    :param training_dataset: Name of the original dataset file.
    :param train_ratio: Proportion of data used for training.
    """
    print(f"Training Random Forest...")

    train_set = data[data['split_type'] == 'train']
    test_set = data[data['split_type'] == 'test']

    X_train = train_set.drop(columns=["label", "attack_cat", "split_type"])
    y_train = train_set["label"]
    X_test = test_set.drop(columns=["label", "attack_cat", "split_type"])
    y_test = test_set["label"]

    model = RandomForestClassifier(random_state=42, verbose=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    metrics = _calculate_metrics(y_test, y_pred, y_scores)

    _save_model_and_metadata(model, 'rf', mode, train_ratio, training_dataset, dataset_type, X_train.shape[0], metrics)
    print("Training process done.")


def run_routine_models(mode):
    """
    Automatically builds all models defined in ROUTINE_MODELS for the selected mode.

    :param mode: Preprocessing mode ('independent' or 'dependent').
    """
    print(f"\n--- Starting Routine Model Building ({mode.upper()}) ---")
    base_data_dir = INDEPENDENT_DIR if mode == 'independent' else DEPENDENT_DIR

    for task in ROUTINE_MODELS:
        data_path = base_data_dir / task['file_rel_path']
        
        if not data_path.exists():
            print(f"Warning: Dataset for routine training not found at {data_path}. Skipping.")
            continue

        print(f"\n[ROUTINE] Building {task['model_type']} using {task['file_rel_path']}...")
        try:
            data = get_data_from_csv(data_path)
            model_processing(
                data=data,
                mode=mode,
                model_type=task['model_type'],
                dataset_type=task['dataset_type'],
                training_dataset=data_path.stem
            )
        except Exception as e:
            print(f"Error during routine building for {task['file_rel_path']}: {e}")

    print("\n--- Routine Model Building Completed ---")


def model_processing(data, mode, model_type, dataset_type, training_dataset):
    """
    Orchestrator function to trigger training for a specific model type.

    :param data: Input DataFrame.
    :param model_type: Type of model to build ('random forest' or 'isolation forest').
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    :param training_dataset: Name of the dataset file.
    """
    if model_type == "random forest":
        random_forest(data, mode, dataset_type, training_dataset)
    elif model_type == "isolation forest":
        isolation_forest(data, mode, dataset_type, training_dataset)