"""
Model training and management for Random Forest.
"""
import joblib
import pandas as pd

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from .plotting import save_feature_importances_plot
from .utils.file_utils import create_directory, update_or_append_csv
from .utils.metrics import calculate_metrics
from .utils.config import MLConstants, Naming, ProjectPaths, PlotFlags


# --- Internal Helper Functions ---
def _get_standardized_model_name(dataset_type, unique_classes, model_type):
    """
    Generates a standardized model name based on the unique classes present in the dataset.
    Supports RF, DT, and HGB models dynamically.

    :param dataset_type: string describing the dataset type
    :param unique_classes: list or array of unique class labels
    :param model_type: string indicating the algorithm type ('rf', 'dt', 'hgb')
    :return: standardized model name as a string
    """
    # Normalize model type to lowercase for naming consistency
    model_suffix = str(model_type).strip().lower()

    # Strip whitespace and convert to lowercase for consistency
    classes = [str(c).strip() for c in unique_classes if str(c).strip()]

    # If there are more than two classes, return a generic name
    if len(classes) > 2:
        return f"{model_suffix}_{dataset_type}_aggregate"
    
    # Identify the attack class (assuming 'normal' is the benign class)
    attack_classes = [c for c in classes if c.lower() != 'normal']
    attack_name = attack_classes[0].lower() if attack_classes else "normal"
    
    return f"{model_suffix}_{dataset_type}_{attack_name}"


def _save_metadata(model, model_name, metrics, dataset_type, classes, samples, dst_dir):
    """
    Saves the metadata of the trained model (RF, DT, HGB, Isolation Forest, etc.), 
    including its parameters and evaluation metrics, to a CSV file.

    :param model: the trained model object
    :param model_name: standardized name for the trained model
    :param metrics: dictionary containing evaluation metrics of the model
    :param dataset_type: string describing the dataset type (nb15, sat20, ...)
    :param classes: string listing the unique classes in the dataset
    :param samples: integer representing the number of samples in the dataset
    :param dst_dir: path of the destination directory
    :return: None
    """
    # Get model and data params
    params = model.get_params()
    
    # Create a unified dictionary to avoid column misalignment in the CSV
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'dataset_type': dataset_type, 
        'classes': classes, 
        'samples': samples, 
        'train_split': MLConstants.TRAIN_SPLIT,
        
        # Common/Shared parameters
        'random_state': params.get('random_state', 'None'),
        'max_depth': params.get('max_depth', 'None'),
        'n_features_in': int(getattr(model, 'n_features_in_', 0)) if hasattr(model, 'n_features_in_') else 'None',
        'n_classes': len(model.classes_) if hasattr(model, 'classes_') else 'None',
        
        # Specific parameters: Random Forest & Decision Tree
        'n_estimators': params.get('n_estimators', 'None'),
        'max_features': params.get('max_features', 'None'),
        'criterion': params.get('criterion', 'None'),
        'min_samples_split': params.get('min_samples_split', 'None'),
        
        # Specific parameters: HistGradientBoosting
        'learning_rate': params.get('learning_rate', 'None'),
        'max_iter': params.get('max_iter', 'None')
    }

    # Add model metrics
    results.update(metrics)

    # Metrics formatting (None -> 'None' in stringa per pulizia nel CSV)
    results = {k: (v if v is not None else 'None') for k, v in results.items()}
    
    # Save metadata
    models_metadata_file = dst_dir / Naming.MODELS_METADATA
    update_or_append_csv(
        file_path=models_metadata_file,
        data_dict=results,
        match_keys=['model_name', 'dataset_type'],
        id_column='id'
    )


def _save_model(model, model_name, dst_dir):
    """
    Saves the trained model to disk and updates the models registry.

    :param model: the trained model object (RF, DT, HGB, etc.)
    :param model_name: standardized name for the trained model
    :param dst_dir: path of the destination directory
    :return: None
    """
    # Save model
    model_path = dst_dir / f'{model_name}.joblib'
    joblib.dump(model, model_path)

    # Save model path in registry
    model_registry_file = dst_dir / Naming.MODELS_REGISTRY

    update_or_append_csv(
        file_path=model_registry_file, 
        data_dict={'path': str(model_path)}, 
        match_keys=['path'], 
        id_column='id'
    )


def _train_classifier(data, model_type, seed):
    """
    Trains a selected classifier (Random Forest, Decision Tree, or HistGradientBoosting) 
    on the provided dataset and returns the trained model, evaluation metrics, and feature names.

    :param data: DataFrame containing the dataset and the 'split_type' attribute
    :param model_type: str, type of model to train. Options: 'rf' (Random Forest), 
                       'dt' (Decision Tree), 'hgb' (HistGradientBoosting)
    :param seed: random_state seed of the current model
    :return: tuple containing the trained model, evaluation metrics, and feature names
    """
    # Get training and testing data split
    train_set = data[data['split_type'] == 'train']
    test_set = data[data['split_type'] == 'test']

    # Drop columns not necessary for training
    X_train = train_set.drop(columns=MLConstants.X_DROP_LABELS)
    X_test = test_set.drop(columns=MLConstants.X_DROP_LABELS)

    # Select target labels
    y_train = train_set[MLConstants.Y_LABEL]
    y_test = test_set[MLConstants.Y_LABEL]

    # Initialize the requested classifier dynamically
    model_type_lower = model_type.lower()
    
    if model_type_lower == Naming.RANDOM_FOREST:    # Get Random Forest
        model = RandomForestClassifier(random_state=seed, verbose=MLConstants.MODEL_VERBOSE, n_jobs=-1)
    elif model_type_lower == Naming.DECISION_TREE:      # Get Decision Tree
        model = DecisionTreeClassifier(random_state=seed)
    elif model_type_lower == Naming.HIST_GRADIENT_BOOSTING:     # Get HGB
        model = HistGradientBoostingClassifier(random_state=seed, verbose=MLConstants.MODEL_VERBOSE)
    else:
        raise ValueError(
            f"[ERROR] Unsupported model type: '{model_type}'. Choose from {MLConstants.MODEL_TYPE}."
        )

    # Fit the chosen model
    model.fit(X_train, y_train)

    # Get predictions and evaluation metrics
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_scores)

    # Get feature names for later use in feature importance plotting
    feature_names = X_train.columns.tolist()

    return model, metrics, feature_names


# --- Public Functions ---
def model_processing(data, dataset_type, model_type, seed):
    """
    Orchestrates the classification pipeline by creating the destination directory,
    training the selected model (RF, DT, or HGB), and saving the resulting model 
    along with its metadata and plots.

    :param data: DataFrame containing the dataset and the 'class' attribute
    :param dataset_type: string describing the dataset type (nb15, sat20, ...)
    :param model_type: string indicating the algorithm type ('rf', 'dt', 'hgb')
    :param seed: random_state seed of the current model
    :return: None
    """
    print(f"\n--- Starting {model_type} Model Building (random_state={seed}) for {dataset_type} ---")

    # --- Create Directories ---
    # Create the main model category directory
    models_category_dir = create_directory(dir_name=model_type, parent_path=ProjectPaths.RUNS)

    # Create the seed model category sub directory
    seed_dir = create_directory(dir_name=f"{seed}", parent_path=models_category_dir)

    # Create the models sub directory
    models_dir = create_directory(dir_name=ProjectPaths.DIR_MODELS, parent_path=seed_dir)

    # Create the feature importance sub directory
    feature_importance_plots_dir = create_directory(dir_name=ProjectPaths.DIR_FEATURE_IMPORTANCE, parent_path=models_dir)

    # --- Build Current Model ---
    # Train the dynamic classifier (handles RF, DT, and HGB internally)
    model, metrics, feature_names = _train_classifier(data, model_type, seed)

    # --- Save model and metadata ---
    # Get unique classes and standardized model name
    unique_classes = data['class'].unique()
    classes = ", ".join(str(c) for c in unique_classes)
    model_name = _get_standardized_model_name(dataset_type, unique_classes, model_type)

    # Save model
    _save_model(
        model=model, 
        model_name=model_name,
        dst_dir=models_dir
    )

    # Save metadata
    _save_metadata(
        model=model, 
        model_name=model_name, 
        metrics=metrics, 
        dataset_type=dataset_type, 
        classes=classes, 
        samples=data.shape[0],
        dst_dir=models_dir
    )

    # --- Save Plot ---
    # Save feature importance plot if enabled
    if PlotFlags.ENABLE_FEATURE_IMPORTANCE:
        # Note: HistGradientBoosting does not support default impurity-based feature_importances_ natively.
        # We ensure the trained model actually has the attribute before plotting to prevent runtime crashes.
        if hasattr(model, 'feature_importances_'):
            save_feature_importances_plot(
                model=model, 
                model_name=model_name,
                feature_names=feature_names, 
                dst_dir=feature_importance_plots_dir
            )
        else:
            print(f" -> [INFO] Feature importance plot skipped: \
                  {model_type} does not natively support impurity-based feature importances.")

    print(f"--- {model_type} Model Building (random_state={seed}) for {dataset_type} Completed ---\n")


if __name__ == "__main__":
    pass
