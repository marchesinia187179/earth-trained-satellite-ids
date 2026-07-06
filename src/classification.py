"""
Classification logic for evaluating trained models on test datasets.
"""
import joblib

from datetime import datetime
from .utils.file_utils import update_or_append_csv
from .utils.metrics import calculate_metrics
from .utils.config import MLConstants, Naming, ProjectPaths, PlotFlags
from .plotting import save_pca_plot, save_pca_plot, save_probability_plot, save_shap_plot


# --- Internal Helper Functions ---
def _save_classification(model_name, metrics, dataset_type, classes, samples):
    """
    Prepares the final classification results dictionary and appends it to the 
    global summary CSV.

    :param model_name: name of the evaluated model
    :param metrics: dictionary of calculated evaluation metrics
    :param dataset_type: type of the dataset being used
    :param classes: comma-separated string representing the unique classes in the dataset
    :param samples: total number of rows/samples in the dataset
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
    classification_file = ProjectPaths.CLASSIFICATIONS_CSV_DIR / Naming.CLASSIFICATIONS
    match_keys = ['model_name', 'dataset_type', 'classes']
    update_or_append_csv(classification_file, results, match_keys)


def _classification(model_path, data):
    """
    Extracts the test split from the data, loads the pre-trained model, 
    makes predictions, and evaluates performance metrics

    :param model_path: path to the serialized joblib model file
    :param data: full dataset containing features, labels, and split indicators
    :return: dictionary containing the calculated evaluation metrics
    """
    # Get testing data, drop columns not necessary and select labels
    test_set = data[data['split_type'] == 'test']
    X_test = test_set.drop(columns=MLConstants.X_DROP_LABELS)
    y_test = test_set[MLConstants.Y_LABEL]

    # Load model
    model = joblib.load(model_path)

    # Get metrics
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_scores)

    return metrics, X_test, y_test, y_scores, model


# --- Public Functions ---
def classification_processing(model_path, data, dataset_type, dataset_name):
    """
    Performs classification using a pre-trained model on a given dataset, evaluates metrics, 
    and generates plots based on configuration flags.

    :param model_path: path to the serialized joblib model file
    :param data: full dataset containing features, labels, and split indicators
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    """
    # Get model name from the model path
    model_name = model_path.stem

    # Get unique classes present in the dataset
    classes = ", ".join(str(c) for c in data['class'].unique())

    print(f"\n--- Classifying {dataset_type} dataset using model: {model_name} ---")

    # Calculate classification and extract test data
    metrics, X_test, y_test, y_scores, model = _classification(model_path, data)

    # Save classification results to the aggregated CSV file
    _save_classification(
        model_name=model_name, 
        metrics=metrics, 
        dataset_type=dataset_type, 
        classes=classes, 
        samples=data.shape[0])

    # --- Save Plots Based on Flags ---
    # Save Probability Distribution plot if enabled
    if PlotFlags.ENABLE_PROBABILITY_PLOTS: 
        save_probability_plot(
            y_test=y_test, 
            y_scores=y_scores, 
            model_name=model_name, 
            dataset_type=dataset_type, 
            dataset_name=dataset_name
        )
    
    # Save SHAP summary plot if enabled
    if PlotFlags.ENABLE_SHAP_PLOTS: 
        save_shap_plot(
            model=model, 
            X_test=X_test, 
            y_test=y_test, 
            model_name=model_name, 
            dataset_type=dataset_type, 
            dataset_name=dataset_name
        )

    print(f"--- Classification for {dataset_type} dataset using model: {model_name} completed ---")

if __name__ == "__main__":
    pass
