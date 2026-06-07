import pathlib
from datetime import datetime

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from utils.file_utils import append_data_to_csv, get_data_from_csv
from utils.paths import RESULTS_DIR


def save_result(model_name, dataset_type, testing_dataset, samples, metrics):
    file_path = RESULTS_DIR / "results.csv"
    
    results_dict = {
        'id': get_data_from_csv(file_path).shape[0] + 1,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

    append_data_to_csv(results_dict, file_path)
    print(f"Classification report appended to: {file_path.name}")


def classification_processing(data, model, dataset_type, model_name="Unknown", testing_dataset="Unknown"):
    print(f"Executing classification (Model: {model_name}, Test Data: {testing_dataset})...")
    X = data.drop(columns=["label", "attack_cat"])
    y = data["label"]

    y_pred = model.predict(X)

    f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    precision = precision_score(y, y_pred, average="macro", zero_division=0)
    recall = recall_score(y, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y, y_pred)
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

    print(f"Metrics: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    save_result(model_name, dataset_type, testing_dataset, data.shape[0], metrics)
