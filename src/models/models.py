"""
Model training and management for Random Forest.
"""
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from plotting.plotting import plot_feature_importances
from utils.file_utils import create_csv_from_data, create_directory, update_or_append_csv
from utils.metrics import calculate_metrics
from utils.config import MLConstants, Naming, ProjectPaths


# --- Internal Helper Functions ---
def _save_feature_importance_and_plots(model, feature_names, plots_dir, csv_dir):
    """
    Extracts and saves the feature importance of the trained Random Forest model

    :param model: the trained Random Forest model object
    :param feature_names: list of feature names corresponding to the model's input features
    :param dst_dir: target directory for saving the feature importance CSV
    :param plots_dir: target directory for saving the feature importance plot
    :param csv_dir: target directory for saving the feature importance CSV
    :return: None
    """
    # If the model is into a Pipeline due to the normalizing, 
    # get the pure Random Forest model object
    if isinstance(model, Pipeline):
        rf_model = model.named_steps['rf']
    else:
        rf_model = model

    # Get feature importance
    importance = rf_model.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})

    # Calculate standard deviation of feature importances across all trees in the Random Forest
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
    
    # Save feature importance to CSV
    existing_files = list(csv_dir.glob(f'{Naming.MODEL}_*.csv'))
    feature_importance_name = f'{Naming.MODEL}_{len(existing_files) + 1}_{Naming.FEATURE_IMPORTANCE}'
    create_csv_from_data(importance_df, f"{feature_importance_name}{Naming.EXT}", csv_dir)

    # Plot feature importance and save the plot
    plot_path = plots_dir / f"{feature_importance_name}{Naming.PLOT_EXT}"
    plot_feature_importances(importance, feature_names, std, plot_path)

    print(f"Feature importance saved to {feature_importance_name}")


def _save_model_and_metadata(model, metrics, dataset_type, classes, samples, main_dst_dir, model_dst_dir):
    """
    Simplifies and unifies saving for Random Forest models
    
    :param model: the trained Random Forest model object
    :param metrics: a dictionary containing the calculated evaluation metrics
    :param dataset_type: string describing the dataset type (nb15, sat20, ...)
    :param samples: integer indicating the number of samples used
    :param main_dst_dir: path object where main files will be saved
    :param model_dst_dir: path object where model files will be saved
    :return: None
    """
    # Save model
    existing_files = list(model_dst_dir.glob(f'{Naming.MODEL}_*.joblib'))
    model_name = f'{Naming.MODEL}_{len(existing_files) + 1}'
    model_path = model_dst_dir / f'{model_name}.joblib'
    joblib.dump(model, model_path)

    # Save model path
    update_or_append_csv(main_dst_dir / Naming.MODELS_PATHS, {'path': str(model_path)}, ['path'], id_column='id')

    # If the model is into a Pipeline due to the normalizing, 
    # get the pure Random Forest model object
    if isinstance(model, Pipeline):
        rf_model = model.named_steps['rf']
    else:
        rf_model = model

    # Get model and data params
    params = rf_model.get_params()
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
        'n_features_in': int(rf_model.n_features_in_),
        'criterion': params.get('criterion', 'gini'),
        'max_depth': params.get('max_depth', None),
        'n_classes': len(rf_model.classes_)
    }

    # Add model metrics
    results.update(metrics)

    # Metrics formatting (None -> 'None')
    results = {k: (v if v is not None else 'None') for k, v in results.items()}
    
    # Save metadata
    info_file = main_dst_dir / Naming.MODEL_INFO
    match_keys = ['model_name', 'dataset_type']
    update_or_append_csv(info_file, results, match_keys, id_column='id')
    
    print(f"Model {model_name} saved to {main_dst_dir.name} and metadata processed in {info_file.name}")


def _random_forest(data, mode):
    """
    Trains a Random Forest classifier using a supervised learning approach

    :param data: data to be used for the training of the model
    :param mode: variable to choose the normalized or unnormalized method
    :return: model built and related metrics
    """
    print(f"Training Random Forest...")

    # Get training and testing data
    train_set = data[data['split_type'] == 'train']
    test_set = data[data['split_type'] == 'test']

    # Drop columns not necessary
    X_train = train_set.drop(columns=["label", "class", "split_type"])
    X_test = test_set.drop(columns=["label", "class", "split_type"])

    # Select labels
    y_train = train_set["label"]
    y_test = test_set["label"]

    # Fit model by choosing the normalized or unnormalized way
    if mode == MLConstants.NORMALIZED:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=MLConstants.RANDOM_STATE, verbose=MLConstants.MODEL_VERBOSE))
        ])
    elif mode == MLConstants.UNNORMALIZED:
        model = RandomForestClassifier(random_state=MLConstants.RANDOM_STATE, verbose=MLConstants.MODEL_VERBOSE)

    model.fit(X_train, y_train)

    # Get metrics
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_scores)

    # Get feature names for later use in feature importance plotting
    feature_names = X_train.columns.tolist()

    print("Training process done.")
    return model, metrics, feature_names


# --- Public Functions ---
def model_processing(data, type, mode):
    """
    Orchestrates the Random Forest pipeline by creating the destination directory,
    training the model, and saving the resulting model along with its metadata

    :param data: DataFrame containing the dataset and the 'class' attribute
    :param type: string describing the dataset type (nb15, sat20, ...)
    :param mode: variable to choose the normalized or unnormalized method
    :return: None
    """
    # Create main directory
    model_dir = create_directory(mode, ProjectPaths.MODELS_DIR)
    joblib_dir = create_directory(ProjectPaths.DIR_MODELS, model_dir)
    feature_importance_dir = create_directory(ProjectPaths.DIR_FEATURE_IMPORTANCE, model_dir)
    feature_importance_plots_dir = create_directory(ProjectPaths.DIR_PLOTS, feature_importance_dir)
    feature_importance_csv_dir = create_directory(ProjectPaths.DIR_CSV, feature_importance_dir)

    # Create random forest model
    model, metrics, feature_names = _random_forest(data, mode)

    # Get classes
    unique_classes = data['class'].unique()
    classes = ", ".join(str(c) for c in unique_classes)

    # Save random forest model and metadata
    _save_model_and_metadata(model, metrics, type, classes, data.shape[0], model_dir, joblib_dir)

    # Get feature importance and save it
    _save_feature_importance_and_plots(model, feature_names, feature_importance_plots_dir, feature_importance_csv_dir)


if __name__ == "__main__":
    pass
