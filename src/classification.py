"""
Classification logic for evaluating trained models on test datasets.
"""
import joblib

from datetime import datetime
from .utils.file_utils import create_directory, update_or_append_csv, _normalize_dataset_classes_filename
from .utils.metrics import calculate_metrics
from .utils.config import Naming, ProjectPaths
from .plotting import plot_probability_distribution, plot_shap_summary


# --- Internal Helper Functions ---
def _save_classification(model_name, metrics, dataset_type, classes, samples, dst_dir):
    """
    Prepares the final classification results dictionary and appends it to the 
    global summary CSV.

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
    
    # Save results in the aggregated classifications master file
    classification_file = dst_dir / Naming.CLASSIFICATIONS
    match_keys = ['model_name', 'dataset_type', 'classes']
    update_or_append_csv(classification_file, results, match_keys)

    print(f"Classification for {dataset_type} on {model_name} saved in {classification_file}")


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
    return metrics, test_set, X_test, y_test, y_scores, model


# --- Public Functions ---
def classification_processing(model_path, data, dataset_type, dataset_name):
    """
    Main orchestration function to run the evaluation workflow. 
    It triggers the classification, extracts metadata, and handles file storage
    """
    # Calculate classification and extract test data
    metrics, test_set, X_test, y_test, y_scores, model = _classification(model_path, data)

    # Get unique classes present in the dataset
    unique_classes = data['class'].unique()
    classes = ", ".join(str(c) for c in unique_classes)

    model_name = model_path.stem
    dataset_stem = dataset_name.lower().replace(' ', '_')
    prob_filename = f"{model_name}_on_{dataset_type.lower()}_{dataset_stem}{Naming.PLOT_EXT}"

    # Create model-specific probability directory and save one plot per model/dataset
    model_prob_dir = create_directory(model_name, ProjectPaths.PROB_PLOTS_DIR)
    prob_output_path = model_prob_dir / prob_filename
    plot_probability_distribution(y_test, y_scores, prob_output_path)

    # Generate SHAP summary plot per model/dataset
    try:
        safe_case = _normalize_dataset_classes_filename(dataset_type, classes)
        shap_filename = f"{model_name}_on_{safe_case}{Naming.PLOT_EXT}"
        model_shap_dir = create_directory(model_name, ProjectPaths.SHAP_PLOTS_DIR)
        shap_output_path = model_shap_dir / shap_filename
        plot_shap_summary(model, X_test, y_test, shap_output_path)
    except Exception as e:
        print(f"Warning: SHAP plotting skipped ({e})")

    # Save the summary CSV file of classifications to the aggregated master file
    dst_dir = create_directory(ProjectPaths.CLASSIFICATIONS_CSV_DIR.name, ProjectPaths.RESULTS_CSV_DIR)
    _save_classification(model_name, metrics, dataset_type, classes, data.shape[0], dst_dir)


if __name__ == "__main__":
    pass
