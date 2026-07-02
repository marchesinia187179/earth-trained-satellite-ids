"""
Classification logic for evaluating trained models on test datasets.
"""
import joblib

from datetime import datetime
from utils.file_utils import update_or_append_csv
from utils.metrics import calculate_metrics
from utils.config import Naming, ProjectPaths


# --- Internal Helper Functions ---
def _save_classification(model_name, metrics, dataset_type, classes, samples, dst_dir):
    """
    Prepares the final classification results dictionary, appends it to the 
    global summary CSV, and triggers sub-directory grouped storage

    :param model_name: name of the evaluated model
    :param metrics: dictionary of calculated evaluation metrics
    :param dataset_type: type of the dataset being used
    :param classes: comma-separated string representing the unique classes in the dataset
    :param samples: total number of rows/samples in the dataset
    :param dst_dir: target root directory for saving the classification outputs
    """
    # Get results
    results = {
        'id': None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'dataset_type': dataset_type,
        'classes': classes,
        'samples': samples
    }

    # Add model metrics
    results.update(metrics)

    # Metrics formatting (None -> 'None')
    results = {k: (v if v is not None else 'None') for k, v in results.items()}
    
    # Save results
    classification_file = dst_dir / Naming.CLASSIFICATIONS
    match_keys = ['model_name', 'dataset_type', 'classes']
    update_or_append_csv(classification_file, results, match_keys)

    print(f"Classification for {dataset_type} on {model_name} saved in {classification_file.name}")


def _classification(model_path, data):
    """
    Extracts the test split from the data, loads the pre-trained model, 
    makes predictions, and evaluates performance metrics

    :param model_path: path to the serialized joblib model file
    :param data: full dataset containing features, labels, and split indicators
    :return: dictionary containing the calculated evaluation metrics
    """
    print(f"Classifying data...")

    # Get testing data, drop columns not necessary and select labels
    test_set = data[data['split_type'] == 'test']
    X_test = test_set.drop(columns=["label", "class", "split_type"])
    y_test = test_set["label"]

    # Load model
    model = joblib.load(model_path)

    # Get metrics
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_scores)

    print(f"Classifying process done.")
    return metrics


# --- Public Functions ---
def classification_processing(model_path, data, type):
    """
    Main orchestration function to run the evaluation workflow. 
    It triggers the classification, extracts metadata, and handles file storage

    :param model_path: path object pointing to the trained model file (.stem will be extracted)
    :param data: input dataset containing the test samples and class metadata
    :param type: type/category description of the dataset
    """
    # Calculate classification
    metrics = _classification(model_path, data)

    # Get classes
    unique_classes = data['class'].unique()
    classes = ", ".join(str(c) for c in unique_classes)

    # Save random forest model and metadata
    dst_dir = ProjectPaths.RESULTS / ProjectPaths.DIR_CSV / ProjectPaths.DIR_CLASSIFICATIONS
    _save_classification(model_path.stem, metrics, type, classes, data.shape[0], )


if __name__ == "__main__":
    pass
