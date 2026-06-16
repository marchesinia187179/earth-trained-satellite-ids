import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from utils.file_utils import append_data_to_csv
from utils.paths import RF_MODELS_SAVED_DIR, RF_INFO_CSV, IF_MODELS_SAVED_DIR, IF_INFO_CSV


def save_isolation_forest(model, train_ratio, training_dataset, dataset_type, samples, metrics):
    print(f"Saving Isolation Forest model to: {IF_MODELS_SAVED_DIR}")
    existing_files = list(IF_MODELS_SAVED_DIR.glob('if_model_*.joblib'))
    id = len(existing_files) + 1
    model_name = f'if_model_{id}'

    joblib.dump(model, IF_MODELS_SAVED_DIR / f'{model_name}.joblib')

    params = model.get_params()

    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'dataset_type': dataset_type,
        'training_dataset': training_dataset,
        'samples': samples,
        'train_ratio': train_ratio,
        
        # --- IPERPARAMETRI DI INPUT ---
        'n_estimators': params['n_estimators'],
        'max_samples': params['max_samples'],
        'max_features': params['max_features'],
        'contamination': params['contamination'],
        'random_state': params.get('random_state', None),
        
        # --- ATTRIBUTI APPRESI DAL MODELLO (Scikit-Learn) ---
        'model_offset': float(model.offset_),
        'max_samples_actual': int(model.max_samples_),
        'n_features_in': int(model.n_features_in_),
        
        # --- METRICHE DI PERFORMANCE ---
        'tp': metrics['tp'],
        'tn': metrics['tn'],
        'fp': metrics['fp'],
        'fn': metrics['fn'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'auc_roc': metrics.get('auc_roc', None)
    }

    append_data_to_csv(results, IF_INFO_CSV)
    print(f"Model metadata saved to {IF_INFO_CSV.name}")


def isolation_forest(data, dataset_type, training_dataset, train_ratio=0.8):
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
    auc_roc = roc_auc_score(y_test, y_scores)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc
    }

    save_isolation_forest(model, train_ratio, training_dataset, dataset_type, X_train.shape[0], metrics)
    print("Training process done.")


def save_random_forest(model, train_ratio, training_dataset, dataset_type, samples, metrics):
    print(f"Saving Random Forest model to: {RF_MODELS_SAVED_DIR}")
    existing_files = list(RF_MODELS_SAVED_DIR.glob('rf_model_*.joblib'))
    id = len(existing_files) + 1
    model_name = f'rf_model_{id}'

    joblib.dump(model, RF_MODELS_SAVED_DIR / f'{model_name}.joblib')

    params = model.get_params()

    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'dataset_type': dataset_type,
        'training_dataset': training_dataset,
        'samples': samples,
        'train_ratio': train_ratio,
        
        # --- IPERPARAMETRI DI INPUT (Specifici per Random Forest) ---
        'n_estimators': params['n_estimators'],
        'criterion': params.get('criterion', 'gini'),
        'max_depth': params.get('max_depth', None),
        'min_samples_split': params.get('min_samples_split', 2),
        'min_samples_leaf': params.get('min_samples_leaf', 1),
        'max_features': params['max_features'],
        'random_state': params.get('random_state', None),
        'n_jobs': params.get('n_jobs', -1),
        
        # --- ATTRIBUTI APPRESI DAL MODELLO (Scikit-Learn) ---
        'n_features_in': int(model.n_features_in_),
        'n_classes': len(model.classes_),
        
        # --- METRICHE DI PERFORMANCE ---
        'tp': metrics['tp'],
        'tn': metrics['tn'],
        'fp': metrics['fp'],
        'fn': metrics['fn'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'auc_roc': metrics.get('auc_roc', None)
    }

    append_data_to_csv(results, RF_INFO_CSV)
    print(f"Model metadata saved to {RF_INFO_CSV.name}")


def random_forest(data, dataset_type, training_dataset, train_ratio=0.8):
    print(f"Training Random Forest...")

    X = data.drop(columns=["label", "attack_cat"])
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)

    model = RandomForestClassifier(random_state=42, verbose=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    save_random_forest(model, train_ratio, training_dataset, dataset_type, X_train.shape[0], metrics)
    print("Training process done.")


def model_processing(data, model_type, dataset_type, training_dataset):
    if model_type == "random forest":
        random_forest(data, dataset_type, training_dataset)
    elif model_type == "isolation forest":
        isolation_forest(data, dataset_type, training_dataset)