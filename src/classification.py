"""
Classification logic for evaluating trained models on test datasets.
"""
import joblib

from datetime import datetime
from .utils.file_utils import create_directory, update_or_append_csv
from .utils.metrics import calculate_metrics
from .utils.config import Naming, ProjectPaths, PlotFlags
from .plotting import plot_pca, plot_probability_distribution, plot_shap_summary


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
    X_test = test_set.drop(columns=["label", "class", "split_type"])
    y_test = test_set["label"]

    # Load model
    model = joblib.load(model_path)

    # Get metrics
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_scores)

    return metrics, X_test, y_test, y_scores, model


def _save_pca_plot(data, dataset_type, dataset_name):
    """
    Generates and saves a PCA plot for the given dataset.

    :param data: full dataset containing features, labels, and split indicators
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    """
    # Define the output path for the PCA plot
    pca_output_path = ProjectPaths.PCA_PLOTS_DIR / f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"

    # Extract features and labels for PCA plotting
    X = data.drop(columns=["label", "class", "split_type"])
    y = data["label"]

    # Generate and save the PCA plot
    plot_pca(X, y, pca_output_path)


def _save_probability_plot(y_test, y_scores, model_name, dataset_type, dataset_name):
    """
    Generates and saves a probability distribution plot for the model's predictions.

    :param y_test: true labels for the test set
    :param y_scores: predicted probabilities for the positive class
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    """
    # Create a directory for the model's probability plots
    model_prob_dir = create_directory(model_name, ProjectPaths.PROB_PLOTS_DIR)

    # Define the filename and output path for the probability distribution plot
    prob_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
    
    # Define the full output path for the probability distribution plot
    prob_output_path = model_prob_dir / prob_filename

    # Generate and save the probability distribution plot
    plot_probability_distribution(y_test, y_scores, prob_output_path)


def _save_shap_plot(model, X_test, y_test, model_name, dataset_type, dataset_name):
    """
    Generates and saves a SHAP summary plot for the model's predictions.

    :param model: trained model used for predictions
    :param X_test: features of the test set
    :param y_test: true labels for the test set
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    """
    # Create a directory for the model's SHAP plots
    model_shap_dir = create_directory(model_name, ProjectPaths.SHAP_PLOTS_DIR)

    # Define the filename and output path for the SHAP summary plot
    shap_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
    
    # Define the full output path for the SHAP summary plot
    shap_output_path = model_shap_dir / shap_filename

    # Generate and save the SHAP summary plot
    plot_shap_summary(model, X_test, y_test, shap_output_path)


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

    # --- Generate Plots Based on Flags ---
    # Save PCA plot if enabled
    if PlotFlags.ENABLE_PCA_PLOTS: _save_pca_plot(data, dataset_type, dataset_name)

    # Generate Probability Distribution plot if enabled
    if PlotFlags.ENABLE_PROBABILITY_PLOTS: _save_probability_plot(y_test, y_scores, model_name, dataset_type, dataset_name)
    
    # Generate SHAP summary plot if enabled
    if PlotFlags.ENABLE_SHAP_PLOTS: _save_shap_plot(model, X_test, y_test, model_name, dataset_type, dataset_name)

    print(f"--- Classification for {dataset_type} dataset using model: {model_name} completed ---")

if __name__ == "__main__":
    pass
