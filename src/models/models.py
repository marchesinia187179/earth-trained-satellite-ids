"""
Model training and management for Random Forest.
"""
import joblib

from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from utils.file_utils import create_directory, update_or_append_csv
from utils.paths import (
    DECIMAL_DIGITS, MODEL_VERBOSE, MODELS_DIR, NORMALIZED, RANDOM_STATE,
    RF_INFO_FILENAME, RF_MODEL_PREFIX, TRAIN_SPLIT, UNNORMALIZED
)

# --- Internal Helper Functions ---
def _calculate_metrics(y_test, y_pred, y_scores):
    """
    Internal helper to calculate metrics consistently

    :param y_test: ground truth (correct) target values containing actual classes (0 or 1)
    :param y_pred: estimated targets (0 or 1) returned by a classifier's predict method
    :param y_scores: predicted probabilities or decision scores for the positive class
    :return: a dictionary mapping metric names to their calculated and formatted values
    """

    # Security check for number of classes to avoid metric calculation error
    has_classes = len(y_test.unique()) > 1
    if not has_classes:
        print("Warrning: the test data contains only one class. Only some metrics will be calculated!")

    # Get metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = round(accuracy_score(y_test, y_pred), DECIMAL_DIGITS) if has_classes else None
    precision = round(precision_score(y_test, y_pred), DECIMAL_DIGITS) if has_classes else None
    recall = round(recall_score(y_test, y_pred), DECIMAL_DIGITS)
    f1 = round(f1_score(y_test, y_pred), DECIMAL_DIGITS) if has_classes else None
    roc = round(roc_auc_score(y_test, y_scores), DECIMAL_DIGITS) if has_classes else None
    pr = round(average_precision_score(y_test, y_scores), DECIMAL_DIGITS) if has_classes else None
    tpr = round(tp / (tp + fn) if (tp + fn) > 0 else None, DECIMAL_DIGITS)
    fnr = round(fn / (tp + fn) if (tp + fn) > 0 else None, DECIMAL_DIGITS)
    tnr = round(tn / (fp + tn) if (fp + tn) > 0 else None, DECIMAL_DIGITS)
    fpr = round(fp / (fp + tn) if (fp + tn) > 0 else None, DECIMAL_DIGITS)
    
    return {
        "TP": tp,   # True Positives: Number of actual attacks correctly identified as attacks
        "TN": tn,   # True Negatives: Number of normal traffic instances correctly identified as normal
        "FP": fp,   # False Positives: Number of normal traffic instances incorrectly flagged as attacks
        "FN": fn,   # False Negatives: Number of actual attacks that completely bypassed the model
        "Accuracy": accuracy,   # Accuracy: Ratio of correct predictions (both attacks and normal) over total instances
        "Precision": precision,     # Precision: Ratio of true attacks identified over total predicted attacks (measures false alarms)
        "Recall": recall,   # Recall: Ratio of true attacks identified over total actual attacks (same as TPR)
        "F1-Score": f1,     # F1-Score: Harmonic mean of Precision and Recall (balances false alarms and missed attacks)
        "ROC-AUC": roc,     # ROC-AUC: Ability of the model to distinguish between classes across all possible thresholds
        "PR-AUC": pr,   # PR-AUC: Average precision across all recall levels (highly critical for imbalanced attack data)
        "TPR": tpr,     # True Positive Rate: Ratio of positive predictions over Total Actual Positives
        "FNR": fnr,     # False Negative Rate: Ratio of actual positive predicted as negative over Total Actual Positives
        "TNR": tnr,     # Total Negative Rate: Ratio of negative predictions over Total Actual Negatives
        "FPR": fpr      # False Positive Rate: Ratio of actual negative predicted as positive over Total Actual Negatives
    }


def _save_model_and_metadata(model, metrics, dataset_type, classes, samples, dst_dir):
    """
    Simplifies and unifies saving for Random Forest models
    
    :param model: the trained Random Forest model object
    :param metrics: a dictionary containing the calculated evaluation metrics
    :param dataset_type: string describing the dataset type (nb15, sat20, ...)
    :param samples: integer indicating the number of samples used
    :param dst_dir: path object where files will be saved
    :return: None
    """
    # Save model
    existing_files = list(dst_dir.glob(f'{RF_MODEL_PREFIX}_*.joblib'))
    model_name = f'{RF_MODEL_PREFIX}_{len(existing_files) + 1}'
    joblib.dump(model, dst_dir / f'{model_name}.joblib')

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
        'train_split': TRAIN_SPLIT,
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
    info_file = dst_dir / RF_INFO_FILENAME
    match_keys = ['model_name', 'dataset_type']
    update_or_append_csv(info_file, results, match_keys, id_column='id')
    
    print(f"Model {model_name} saved to {dst_dir.name} and metadata processed in {info_file.name}")


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
    if mode == NORMALIZED:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=RANDOM_STATE, verbose=MODEL_VERBOSE))
        ])
    elif mode == UNNORMALIZED:
        model = RandomForestClassifier(random_state=RANDOM_STATE, verbose=MODEL_VERBOSE)

    model.fit(X_train, y_train)

    # Get metrics
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    metrics = _calculate_metrics(y_test, y_pred, y_scores)

    print("Training process done.")
    return model, metrics


# --- Public Training Functions ---
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
    model_dir = create_directory(mode, MODELS_DIR)

    # Create random forest model
    model, metrics = _random_forest(data, mode)

    # Get classes
    unique_classes = data['class'].unique()
    classes = ", ".join(str(c) for c in unique_classes)

    # Save random forest model and metadata
    _save_model_and_metadata(model, metrics, type, classes, data.shape[0], model_dir)
