import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from utils.file_utils import append_data_to_csv
from utils.paths import RF_MODELS_SAVED_DIR, RF_INFO_CSV, IF_MODELS_SAVED_DIR, IF_INFO_CSV


def save_isolation_forest(model, params, training_dataset, dataset_type, samples, metrics):
    print(f"Saving Isolation Forest model to: {IF_MODELS_SAVED_DIR}")
    existing_files = list(IF_MODELS_SAVED_DIR.glob('if_model_*.joblib'))
    id = len(existing_files) + 1
    model_name = f'if_model_{id}'

    joblib.dump(model, IF_MODELS_SAVED_DIR / f'{model_name}.joblib')

    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'dataset_type': dataset_type,
        'training_dataset': training_dataset,
        'samples': samples,
        'train_ratio': params['train_ratio'],
        
        # --- IPERPARAMETRI DI INPUT ---
        'n_estimators': params['n_estimators'],
        'max_samples': params['max_samples'],
        'max_features': params['max_features'],
        'contamination': params['contamination'],
        'random_state': params.get('random_state', None), # Importante per la riproducibilità!
        
        # --- ATTRIBUTI APPRESI DAL MODELLO (Scikit-Learn) ---
        'model_offset': float(iforest_model.offset_),
        'max_samples_actual': int(iforest_model.max_samples_),
        'n_features_in': int(iforest_model.n_features_in_),
        
        # --- METRICHE DI PERFORMANCE ---
        'tp': metrics['tp'],
        'tn': metrics['tn'],
        'fp': metrics['fp'],
        'fn': metrics['fn'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'auc_roc': metrics.get('auc_roc', None) # La metrica che abbiamo sistemato prima
    }


    


def isolation_forest(data, dataset_type, training_dataset, train_ratio=0.8):
    print(f"Training Isolation Forest...")

    X = data.drop(columns=["label", "attack_cat"])
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)

    model = IsolationForest(random_state=42, verbose=3)
    model.fit(X_train)
    y_pred = model.predict(X_test)

    params = model.get_params()
    params['train_ratio'] = train_ratio

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

    save_random_forest(model, params, training_dataset, dataset_type, X_train.shape[0], metrics)
    print("Training process done.")


def save_random_forest(model, params, training_dataset, dataset_type, samples, metrics):
    print(f"Saving Random Forest model to: {RF_MODELS_SAVED_DIR}")
    existing_files = list(RF_MODELS_SAVED_DIR.glob('rf_model_*.joblib'))
    id = len(existing_files) + 1
    model_name = f'rf_model_{id}'

    joblib.dump(model, RF_MODELS_SAVED_DIR / f'{model_name}.joblib')

    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'dataset_type': dataset_type,
        'training_dataset': training_dataset,
        'samples': samples,
        'train_ratio': params['train_ratio'],
        'n_estimators': params['n_estimators'],
        'max_depth': 'None' if params['max_depth'] is None else params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'max_features': params['max_features'],
        'tp': metrics['tp'],
        'tn': metrics['tn'],
        'fp': metrics['fp'],
        'fn': metrics['fn'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall']
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

    params = model.get_params()
    params['train_ratio'] = train_ratio

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

    save_random_forest(model, params, training_dataset, dataset_type, X_train.shape[0], metrics)
    print("Training process done.")


def model_processing(data, model_type, dataset_type, training_dataset):
    if model_type == "random forest":
        random_forest(data, dataset_type, training_dataset)
    elif model_type == "isolation forest":
        isolation_forest(data, dataset_type, training_dataset)