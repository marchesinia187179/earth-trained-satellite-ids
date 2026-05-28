import pathlib
import numpy as np
import joblib
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


def build_random_forest(model_name, data, train_ratio, n_estimators, max_depth, random_state=42, verbose=0):

    X = data.drop(columns=["label", "attack_cat"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=random_state)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, verbose=verbose)
    model.fit(X_train, y_train)

    save_path = pathlib.Path(__file__).resolve().parent / "saved" / f"{model_name}_{train_ratio}_{n_estimators}_{max_depth}.joblib"
    joblib.dump(model, save_path)

    y_pred = model.predict(X_test)
    print(f1_score(y_test, y_pred, average="macro"))
    print(precision_score(y_test, y_pred, average="macro"))
    print(recall_score(y_test, y_pred, average="macro"))

    
