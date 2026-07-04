"""
Model training and management for Random Forest.
"""
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from .plotting import plot_feature_importances
from .utils.file_utils import create_csv_from_data, create_directory, update_or_append_csv
from .utils.metrics import calculate_metrics
from .utils.config import MLConstants, Naming, ProjectPaths, PlotFlags


# --- Internal Helper Functions ---
def _get_standardized_model_name(unique_classes):
    """
    Generates a standardized model name based on the unique classes present in the dataset.

    :param unique_classes: list or array of unique class labels
    :return: standardized model name as a string
    """
    # Strip whitespace and convert to lowercase for consistency
    classes = [str(c).strip() for c in unique_classes if str(c).strip()]

    # If there are more than two classes, return a generic name
    if len(classes) > 2:
        return "model_aggregate"
    
    # Identify the attack class (assuming 'normal' is the benign class)
    attack_classes = [c for c in classes if c.lower() != 'normal']
    attack_name = attack_classes[0].lower() if attack_classes else "unknown"
    
    return f"model_{attack_name}"


def _save_feature_importance_and_plots(model, feature_names, plots_dir, csv_dir, model_name):
    """
    Extracts and saves the feature importance of the trained Random Forest model

    :param model: the trained Random Forest model object
    :param feature_names: list of feature names corresponding to the model's input features
    :param dst_dir: target directory for saving the feature importance CSV
    :param plots_dir: target directory for saving the feature importance plot
    :param csv_dir: target directory for saving the feature importance CSV
    :return: None
    """
    # Get feature importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})

    # Calculate standard deviation of feature importances across all trees in the Random Forest
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
    # Save feature importance to CSV using the standardized model name
    feature_importance_name = f'{model_name}_{Naming.FEATURE_IMPORTANCE}'
    create_csv_from_data(importance_df, f"{feature_importance_name}{Naming.EXT}", csv_dir)

    # Plot feature importance if enabled, otherwise skip to save computation time
    if PlotFlags.ENABLE_FEATURE_IMPORTANCE:
        plot_path = plots_dir / f"{model_name}{Naming.PLOT_EXT}"
        plot_feature_importances(importance, feature_names, std, plot_path)
    else:
        print(f"⏭️  Skipping feature importance plot for {model_name} (ENABLE_FEATURE_IMPORTANCE=False)")

    print(f"Feature importance saved to {feature_importance_name}")


def _save_metadata(model, model_name, metrics, dataset_type, classes, samples):
    """
    Saves the metadata of the trained Random Forest model, including its parameters and evaluation metrics, to a CSV file.

    :param model: the trained Random Forest model object
    :param model_name: standardized name for the trained model
    :param metrics: dictionary containing evaluation metrics of the model
    :param dataset_type: string describing the dataset type (nb15, sat20, ...)
    :param classes: string listing the unique classes in the dataset
    :param samples: integer representing the number of samples in the dataset
    :return: None
    """

    # Get model and data params
    params = model.get_params()
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'dataset_type': dataset_type, 
        'classes': classes, 
        'samples': samples, 
        'train_split': MLConstants.TRAIN_SPLIT,
        'n_estimators': params['n_estimators'],
        'max_features': params['max_features'],
        'random_state': params.get('random_state', None),
        'n_features_in': int(model.n_features_in_),
        'criterion': params.get('criterion', 'gini'),
        'max_depth': params.get('max_depth', None),
        'n_classes': len(model.classes_)
    }

    # Add model metrics
    results.update(metrics)

    # Metrics formatting (None -> 'None')
    results = {k: (v if v is not None else 'None') for k, v in results.items()}
    
    # Save metadata
    match_keys = ['model_name', 'dataset_type']
    update_or_append_csv(ProjectPaths.MODELS_INFO, results, match_keys, id_column='id')
    
    print(f"Metadata saved in {ProjectPaths.MODELS_INFO.name}")


def _save_model(model, model_name):
    """
    Saves the trained Random Forest model to disk and updates the models registry.

    :param model: the trained Random Forest model object
    :param model_name: standardized name for the trained model
    :return: None
    """
    # Save model
    model_path = ProjectPaths.MODELS / f'{model_name}.joblib'
    joblib.dump(model, model_path)

    # Save model path in registry
    update_or_append_csv(ProjectPaths.MODELS_REGISTRY, {'path': str(model_path)}, ['path'], id_column='id')
    
    print(f"Model {model_name} saved to {ProjectPaths.MODELS_REGISTRY.name}")


def _random_forest(data):
    """
    Trains a Random Forest model on the provided dataset and returns the trained model, evaluation metrics, and feature names.

    :param data: DataFrame containing the dataset and the 'class' attribute
    :return: tuple containing the trained model, evaluation metrics, and feature names
    """
    # Get training and testing data
    train_set = data[data['split_type'] == 'train']
    test_set = data[data['split_type'] == 'test']

    # Drop columns not necessary
    X_train = train_set.drop(columns=["label", "class", "split_type"])
    X_test = test_set.drop(columns=["label", "class", "split_type"])

    # Select labels
    y_train = train_set["label"]
    y_test = test_set["label"]

    # Build and fit the model
    model = RandomForestClassifier(
        random_state=MLConstants.RANDOM_STATE, 
        verbose=MLConstants.MODEL_VERBOSE
    )
    model.fit(X_train, y_train)

    # Get metrics
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_scores)

    # Get feature names for later use in feature importance plotting
    feature_names = X_train.columns.tolist()

    return model, metrics, feature_names


# --- Public Functions ---
def model_processing(data, type):
    """
    Orchestrates the Random Forest pipeline by creating the destination directory,
    training the model, and saving the resulting model along with its metadata

    :param data: DataFrame containing the dataset and the 'class' attribute
    :param type: string describing the dataset type (nb15, sat20, ...)
    :return: None
    """
    # Create feature importance directories
    feature_importance_plots_dir = create_directory(ProjectPaths.DIR_FEATURE_IMPORTANCE, ProjectPaths.RESULTS_PLOT_DIR)
    feature_importance_csv_dir = create_directory(ProjectPaths.DIR_FEATURE_IMPORTANCE, ProjectPaths.RESULTS_CSV_DIR)

    # Create random forest model
    model, metrics, feature_names = _random_forest(data)

    # Get unique classes and standardized model name
    unique_classes = data['class'].unique()
    classes = ", ".join(str(c) for c in unique_classes)
    model_name = _get_standardized_model_name(unique_classes)

    # --- Save tasks  ---
    # Save model
    _save_model(
        model=model, 
        model_name=model_name
    )

    # Save metadata
    _save_metadata(
        model=model, 
        model_name=model_name, 
        metrics=metrics, 
        dataset_type=type, 
        classes=classes, 
        samples=data.shape[0]
    )

    # Get feature importance and save it
    _save_feature_importance_and_plots(model, feature_names, feature_importance_plots_dir, feature_importance_csv_dir, model_name)


if __name__ == "__main__":
    pass
