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
from utils.file_utils import append_data_to_csv
from utils.paths import (
    INDEPENDENT_RF_DIR, DEPENDENT_RF_DIR,
    INDEPENDENT_IF_DIR, DEPENDENT_IF_DIR
)


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
        csv_name = 'random_forest_models_info.csv'
    else:
        save_dir = INDEPENDENT_IF_DIR if mode == 'independent' else DEPENDENT_IF_DIR
        prefix = 'if_model'
        csv_name = 'isolation_forest_models_info.csv'

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
        'recall': metrics['recall'],
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
    
    data_normal = data[data['label'] == 0]
    data_anomaly = data[data['label'] == 1]

    train_normal, test_normal = train_test_split(data_normal, train_size=train_ratio, random_state=42)

    n_anomaly_sample = int(len(test_normal) * 0.10)
    test_anomaly = data_anomaly.sample(n=n_anomaly_sample, random_state=42)

    data_test = pd.concat([test_normal, test_anomaly])

    X_train = train_normal.drop(columns=['attack_cat', 'label'])
    X_test = data_test.drop(columns=['attack_cat', 'label'])
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

    X = data.drop(columns=["label", "attack_cat"])
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)

    model = RandomForestClassifier(random_state=42, verbose=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    metrics = _calculate_metrics(y_test, y_pred, y_scores)

    _save_model_and_metadata(model, 'rf', mode, train_ratio, training_dataset, dataset_type, X_train.shape[0], metrics)
    print("Training process done.")


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