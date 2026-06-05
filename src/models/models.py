import pathlib
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from utils.file_utils import append_data_to_csv


def isolation_forest():
    pass


def save_random_forest(model, params, attack_cat, f1, precision, recall):
    save_path = pathlib.Path(__file__).resolve().parent / "random_forest_saved"

    existing_files = list(save_path.glob('random_forest_model*.joblib'))
    id = len(existing_files) + 1

    joblib.dump(model, save_path / f'random_forest_model_{id}.joblib')

    results = {
        'id': id,
        'train_ratio': params['train_ratio'],
        'n_estimators': params['n_estimators'],
        'max_depth': 'None' if params['max_depth'] is None else params['max_depth'],
        'min_samples_split': params['min_samples_split'],
        'max_features': params['max_features'],
        'cross_validation': params['cross_validation'],
        'attack_cat': attack_cat,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    append_data_to_csv(results, save_path / 'random_forest_models_info.csv')


def random_forest(data, train_ratio=0.8, n_estimators=None, max_depth=None, min_samples_split=None, max_features=None, cross_validation=True):

    X = data.drop(columns=["label", "attack_cat"])
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)

    if cross_validation:
        samples = X.shape[0]
        n_est = 300 if samples > 50000 else 100
        param_dist = {
            'n_estimators': [n_est, n_est + 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }

        n_iter_tuning = 10 if samples > 100000 else 5
        rf = RandomForestClassifier(random_state=42)
        model = RandomizedSearchCV(
            estimator=rf, 
            param_distributions=param_dist, 
            n_iter=n_iter_tuning, 
            cv=3, 
            n_jobs=-1, 
            verbose=3
        )

        model.fit(X_train, y_train)
        y_pred = model.best_estimator_.predict(X_test)
        
        params = model.best_params_.copy()
        params['train_ratio'] = train_ratio
        params['cross_validation'] = True

        model = model.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        params = {
            'train_ratio': train_ratio,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'max_features': max_features,
            'cross_validation': False
        }

    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    attack_cat = "_".join(data["attack_cat"].unique())

    save_random_forest(model, params, attack_cat, f1, precision, recall)


def model_processing(data, model_type):

    if model_type not in ["random forest", "isolation forest"]:
        print("Invalid model type")
        return
    
    if model_type == "random forest":

        user_choice = input("Do you want to set the parameters? [y/n] ").lower()
        if user_choice not in ['y', 'n']:
            print("Error: Invalid input. Please enter 'y' or 'n'.")
            sys.exit(1)
        
        if user_choice == 'y':
            user_input = input("Insert train_ratio, n_estimators, max_depth, min_samples_split and max_features: [train_ratio n_estimators max_depth min_samples_split max_features] ").split()
            if len(user_input) != 5:
                print("Error: Invalid input format. Expected: [train_ratio n_estimators max_depth min_samples_split max_features]")
                sys.exit(1)

            train_ratio = float(user_input[0])
            n_estimators = int(user_input[1])
            max_depth = None if user_input[2].lower() == 'none' else int(user_input[2])
            min_samples_split = int(user_input[3])
            max_features = user_input[4]
            
            random_forest(data, train_ratio, n_estimators, max_depth, min_samples_split, max_features, cross_validation=False)
        else:
            random_forest(data)

    elif model_type == "isolation forest":
        isolation_forest(data)