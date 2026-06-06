import pathlib

from sklearn.metrics import f1_score, precision_score, recall_score
from utils.file_utils import append_data_to_csv, get_data_from_csv
from utils.paths import RESULTS_DIR


def save_result(model_name, model_training_dataset, testing_dataset, f1, precision, recall):
    file_path = RESULTS_DIR / "results.csv"
    
    results_dict = {
        'id': get_data_from_csv(file_path).shape[0] + 1,
        'model_name': model_name,
        'model_training_dataset': model_training_dataset,
        'testing_dataset': testing_dataset,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    append_data_to_csv(results_dict, file_path)
    print(f"Classification report appended to: {file_path.name}")


def classification_processing(data, model, model_name="Unknown", model_training_dataset="Unknown", testing_dataset="Unknown"):
    print(f"Executing classification (Model: {model_name}, Test Data: {testing_dataset})...")
    X = data.drop(columns=["label", "attack_cat"])
    y = data["label"]

    y_pred = model.predict(X)

    f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    precision = precision_score(y, y_pred, average="macro", zero_division=0)
    recall = recall_score(y, y_pred, average="macro", zero_division=0)

    print(f"Metrics: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    save_result(model_name, model_training_dataset, testing_dataset, f1, precision, recall)
