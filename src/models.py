"""
Model training and management for Random Forest.
"""
import joblib

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from .plotting import save_feature_importances_plot
from .utils.file_utils import create_directory, update_or_append_csv
from .utils.metrics import calculate_metrics
from .utils.config import MLConstants, ProjectPaths, PlotFlags


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
    update_or_append_csv(
        file_path=ProjectPaths.MODELS_INFO,
        data_dict=results,
        match_keys=['model_name', 'dataset_type'],
        id_column='id'
    )


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
    update_or_append_csv(
        file_path=ProjectPaths.MODELS_REGISTRY, 
        data_dict={'path': str(model_path)}, 
        match_keys=['path'], 
        id_column='id'
    )


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
    X_train = train_set.drop(columns=MLConstants.X_DROP_LABELS)
    X_test = test_set.drop(columns=MLConstants.X_DROP_LABELS)

    # Select labels
    y_train = train_set[MLConstants.Y_LABEL]
    y_test = test_set[MLConstants.Y_LABEL]

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
def model_processing(data, dataset_type):
    """
    Orchestrates the Random Forest pipeline by creating the destination directory,
    training the model, and saving the resulting model along with its metadata

    :param data: DataFrame containing the dataset and the 'class' attribute
    :param dataset_type: string describing the dataset type (nb15, sat20, ...)
    :return: None
    """
    print(f"\n--- Starting Model Building for {dataset_type} ---")

    # Create feature importance directory
    feature_importance_plots_dir = create_directory(ProjectPaths.DIR_FEATURE_IMPORTANCE, ProjectPaths.RESULTS_PLOT_DIR)

    # Create random forest model
    model, metrics, feature_names = _random_forest(data)

    # Get unique classes and standardized model name
    unique_classes = data['class'].unique()
    classes = ", ".join(str(c) for c in unique_classes)
    model_name = _get_standardized_model_name(unique_classes)

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
        dataset_type=dataset_type, 
        classes=classes, 
        samples=data.shape[0]
    )

    # Save feature importance plot
    if PlotFlags.ENABLE_FEATURE_IMPORTANCE:
        save_feature_importances_plot(
            model=model, 
            model_name=model_name,
            feature_names=feature_names, 
            dst_dir=feature_importance_plots_dir
        )

    print(f"--- Model Building for {dataset_type} Completed ---\n")


if __name__ == "__main__":
    pass
