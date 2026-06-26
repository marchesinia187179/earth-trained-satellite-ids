"""
Classification logic for evaluating trained models on test datasets.
"""
import joblib
import pandas as pd

from datetime import datetime
from utils.file_utils import create_csv_from_data, create_directory, update_or_append_csv
from utils.metrics import calculate_metrics
from utils.paths import (
    BY_DATASET_DIR_NAME, BY_MODEL_DIR_NAME, CLASSES_DIR_NAME, CLASSIFICATION_SUFFIX, CLASSIFICATIONS_DIR, CLASSIFICATIONS_FILENAME
)


# --- Internal Helper Functions ---
def _save_classification_grouped_by(results, by_model_dir, by_dataset_dir):
    """
    Groups the classification results by model and by dataset type/classes, 
    and saves them into separate CSV files

    :param results: dictionary containing the classification metadata and evaluation metrics
    :param by_model_dir: directory path where model-specific results will be saved
    :param by_dataset_dir: directory path where dataset-specific results will be saved
    """
    # Get results
    results_df = pd.DataFrame(results)

    # Group and save results by model_name
    for model_name, group in results_df.groupby('model_name'):
        create_csv_from_data(group, f"{model_name}{CLASSIFICATION_SUFFIX}", by_model_dir)

    # Group and save results by classes for each dataset_type
    for dataset_type, group in results_df.groupby('dataset_type'):
        curr_dataset_dir = create_directory(dataset_type, by_dataset_dir)
        
        for classes_name, classes_group in group.groupby('classes'):
            curr_classes_dir = create_directory(CLASSES_DIR_NAME, curr_dataset_dir)
            create_csv_from_data(classes_group, f"{classes_name}{CLASSIFICATION_SUFFIX}", curr_classes_dir)












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
    classification_file = dst_dir / CLASSIFICATIONS_FILENAME
    match_keys = ['model_name', 'dataset_type', 'classes']
    update_or_append_csv(classification_file, results, match_keys)

    by_model_dir = create_directory(BY_MODEL_DIR_NAME, dst_dir)
    by_dataset_dir = create_directory(BY_DATASET_DIR_NAME, dst_dir)
    _save_classification_grouped_by(results, by_model_dir, by_dataset_dir)

    print(f"Classification for {dataset_type} on {model_name} saved in {classification_file.name}")


def _classification(model, data):
    """
    Extracts the test split from the data, loads the pre-trained model, 
    makes predictions, and evaluates performance metrics

    :param model: path to the serialized joblib model file
    :param data: full dataset containing features, labels, and split indicators
    :return: dictionary containing the calculated evaluation metrics
    """
    print(f"Classifying data...")

    # Get testing data, drop columns not necessary and select labels
    test_set = data[data['split_type'] == 'test']
    X_test = test_set.drop(columns=["label", "class", "split_type"])
    y_test = test_set["label"]

    # Load model
    rf_model = joblib.load(model)

    # Get metrics
    y_pred = rf_model.predict(X_test)
    y_scores = rf_model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_scores)

    print(f"Classifying process done.")
    return metrics


# --- Main Processing Logic ---
def classification_processing(model, data, type, mode):
    """
    Main orchestration function to run the evaluation workflow. It initializes 
    the output directory, triggers the classification, extracts metadata, 
    and handles file storage

    :param model: path object pointing to the trained model file (.stem will be extracted)
    :param data: input dataset containing the test samples and class metadata
    :param type: type/category description of the dataset
    :param mode: execution mode, used as the main directory folder name
    """
    # Create main directory
    classification_dir = create_directory(mode, CLASSIFICATIONS_DIR)
    
    # Calculate classification
    metrics = _classification(model, data)

    # Get classes
    unique_classes = data['class'].unique()
    classes = ", ".join(str(c) for c in unique_classes)

    # Save random forest model and metadata
    _save_classification(model.stem, metrics, type, classes, data.shape[0], classification_dir)


if __name__ == "__main__":
    pass