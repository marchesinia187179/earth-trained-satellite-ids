"""
Classification logic for evaluating trained models on test datasets.
"""
import joblib

from datetime import datetime

from .utils.file_utils import update_or_append_csv
from .utils.metrics import calculate_metrics
from .utils.config import MLConstants, PlotFlags
from .plotting import save_probability_plot, save_shap_plot, save_pr_curve_plot, save_threshold_metrics_plot


# --- Internal Helper Functions ---
def _save_classification(model_name, metrics, dataset_type, classes, samples, classifications_file):
    """
    Prepares the final classification results dictionary for any evaluated classifier 
    (RF, DT, or HGB) and appends or updates it in the global summary CSV.

    :param model_name: name of the evaluated model
    :param metrics: dictionary of calculated evaluation metrics
    :param dataset_type: type of the dataset being used
    :param classes: comma-separated string representing the unique classes in the dataset
    :param samples: total number of rows/samples in the dataset
    :param classifications_file: file path where to save classification results
    """
    # Initialize the base results dictionary with metadata
    results = {
        'id': None,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'dataset_type': dataset_type,
        'classes': classes,
        'samples': samples
    }

    # Add model-specific evaluation metrics
    results.update(metrics)

    # Clean metrics formatting by converting None values to 'None' strings for CSV consistency
    results = {k: (v if v is not None else 'None') for k, v in results.items()}
    
    # Save or update results in the aggregated classifications master file
    update_or_append_csv(
        file_path=classifications_file, 
        data_dict=results, 
        match_keys=['model_name', 'dataset_type', 'classes'],
        id_column='id'
    )


def _classification(model_path, data):
    """
    Extracts the test split from the data, loads the pre-trained classifier (RF, DT, or HGB), 
    generates predictions, and evaluates performance metrics.

    :param model_path: path to the serialized joblib model file
    :param data: full dataset containing features, labels, and split indicators
    :return: tuple containing (metrics_dict, X_test, y_test, y_scores, model)
    """
    # Extract the testing set based on the split flag
    test_set = data[data['split_type'] == 'test']
    
    # Drop unnecessary columns and separate features from labels
    X_test = test_set.drop(columns=MLConstants.X_DROP_LABELS)
    y_test = test_set[MLConstants.Y_LABEL]

    # Load the serialized model from disk
    model = joblib.load(model_path)

    # Generate predictions and prediction probabilities (valid for RF, DT, and HGB)
    if hasattr(model, "predict_proba"):
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        raise AttributeError(f"[ERROR] Loaded model '{model_path.stem}' does not support 'predict_proba'. "
                             f"Ensure it is a valid RF, DT, or HGB classifier.")

    # Calculate final performance metrics
    metrics = calculate_metrics(y_test, y_pred, y_scores)

    return metrics, X_test, y_test, y_scores, model


# --- Public Functions ---
def classification_processing(model_path, data, dataset_type, dataset_name, classifications_file, plots_dir):
    """
    Performs classification using a pre-trained model (RF, DT, or HGB) on a given dataset, 
    evaluates metrics, and generates plots based on configuration flags.

    :param model_path: path to the serialized joblib model file
    :param data: full dataset containing features, labels, and split indicators
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    :param classifications_file: file path where to save classification results
    :param plots_dir: dir path where to save plots
    """
    # Get model name from the model path
    model_name = model_path.stem

    # Get unique classes present in the dataset
    classes = ", ".join(str(c) for c in data['class'].unique())

    print(f"\n--- Classifying {dataset_type} dataset using model: {model_name} ---")

    # Calculate classification and extract test data (works generically for loaded RF, DT, and HGB)
    metrics, X_test, y_test, y_scores, model = _classification(model_path, data)

    # Save classification results to the aggregated CSV file
    _save_classification(
        model_name=model_name, 
        metrics=metrics, 
        dataset_type=dataset_type, 
        classes=classes, 
        samples=data.shape[0],
        classifications_file=classifications_file
    )

    # --- Save Plots Based on Flags ---
    # Save Probability Distribution plot if enabled
    if PlotFlags.ENABLE_PROBABILITY_PLOTS: 
        save_probability_plot(
            y_test=y_test, 
            y_scores=y_scores, 
            model_name=model_name, 
            dataset_type=dataset_type, 
            dataset_name=dataset_name,
            dst_dir=plots_dir
        )

    # Save Precision-Recall (PR) Curve plot if enabled
    if PlotFlags.ENABLE_PR_CURVE_PLOTS:
        save_pr_curve_plot(
            y_test=y_test, 
            y_scores=y_scores, 
            model_name=model_name, 
            dataset_type=dataset_type, 
            dataset_name=dataset_name,
            dst_dir=plots_dir
        )

    # Save Threshold Sensitivity plot if enabled
    if PlotFlags.ENABLE_THRESHOLD_PLOTS:
        save_threshold_metrics_plot(
            y_test=y_test, 
            y_scores=y_scores, 
            model_name=model_name, 
            dataset_type=dataset_type, 
            dataset_name=dataset_name,
            dst_dir=plots_dir
        )
    
    # Save SHAP summary plot if enabled
    if PlotFlags.ENABLE_SHAP_PLOTS: 
        try:
            # Wrapped in try-except block to handle model-specific SHAP explainer incompatibilities
            # (e.g., HistGradientBoosting might require different explainers than standard Trees)
            save_shap_plot(
                model=model, 
                X_test=X_test, 
                y_test=y_test, 
                model_name=model_name, 
                dataset_type=dataset_type, 
                dataset_name=dataset_name,
                dst_dir=plots_dir
            )
        except Exception as e:
            print(f" -> [WARNING] Failed to generate SHAP plot for model '{model_name}': {e}")

    print(f"--- Classification for {dataset_type} dataset using model: {model_name} completed ---")


if __name__ == "__main__":
    pass
