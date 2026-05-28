from models.random_forest.random_forest import build_random_forest
from utils.file_utils import *
from preprocessing.data_preprocessing import *
from preprocessing.file_preprocessing import *
import sys
import pathlib
import pandas as pd


def file_preprocessing_state(data_preprocessed, dataset_type):
    """
    Wrapper function to trigger the file-level preprocessing state.

    :param data_preprocessed: The DataFrame after data preprocessing.
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    """
    file_preprocessing(data_preprocessed, dataset_type)


def data_preprocessing_state(path, dataset_type):
    """
    Wrapper function to load data and trigger the data-level preprocessing state.

    :param path: Path to the raw CSV file.
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    :internal data: Raw DataFrame loaded from CSV.
    :return: Preprocessed DataFrame.
    """
    data = get_data_from_csv(path)
    data_preprocessed = data_preprocessing(data, dataset_type)

    return data_preprocessed


def preprocessing_loop(user_choice):
    while user_choice == 'y':
        user_input = input("Insert the path and the dataset_type of the dataset: [path dataset_type] (dataset_type: nb15, sat20 or ter20) ").split()
        
        # Check if the user provided both path and type
        if len(user_input) != 2:
            print("Error: Invalid input format. Expected: [path dataset_type]")
            sys.exit(1)

        path, dataset_type = user_input

        # Check if the file path exists
        if not pathlib.Path(path).exists():
            print(f"Error: The file path '{path}' does not exist.")
            sys.exit(1)

        # Check if the dataset type is supported
        if dataset_type not in ['nb15', 'sat20', 'ter20']:
            print(f"Error: Unsupported dataset type '{dataset_type}'. Choose from: nb15, sat20, ter20")
            sys.exit(1)

        data_preprocessed = data_preprocessing_state(path, dataset_type)
        file_preprocessing_state(data_preprocessed, dataset_type)

        user_choice = input("Do you want to start a new preprocessing? [y/n] ").lower()
        if user_choice not in ['y', 'n']:
            print("Error: Invalid input. Please enter 'y' or 'n'.")
            sys.exit(1)


def random_forest_state():
    data_path_str = input("Insert the path of the dataset: [path] ")
    data_path = pathlib.Path(data_path_str)

    # Check if the file path exists
    if not data_path.exists():
        print(f"Error: The file path '{data_path_str}' does not exist.")
        sys.exit(1)

    model_name = data_path.stem
    data = get_data_from_csv(data_path)
    data = pd.DataFrame(data)

    # Ensure the dataset contains the necessary 'label' column
    if 'label' not in data.columns:
        print(f"Error: The dataset at '{data_path}' is missing the 'label' column.")
        sys.exit(1)

    user_input = input("Insert the train ratio, the number of estimators and the max depth of the model: [train_ratio, n_estimators, max_depth] ").split()

    # Check if the user provided all three parameters
    if len(user_input) != 3:
        print("Error: Invalid input format. Expected: [train_ratio, n_estimators, max_depth]")
        sys.exit(1)

    try:
        train_ratio = float(user_input[0])
        n_estimators = int(user_input[1])
        max_depth = int(user_input[2])
    except ValueError:
        print("Error: Invalid input for train_ratio, n_estimators, or max_depth. Please enter numeric values.")
        sys.exit(1)
    
    # Basic validation for the parameters
    if not (0 < train_ratio < 1) or n_estimators <= 0 or max_depth <= 0:
        print("Error: train_ratio must be between 0 and 1 (exclusive), n_estimators and max_depth must be positive integers.")
        sys.exit(1)

    build_random_forest(model_name, data, train_ratio, n_estimators, max_depth)


def isolation_forest_state():
    pass


def build_model_loop(user_choice):
    while user_choice == 'y':

        user_input = input("Choose which model do you want to use: [random forest or isolation forest] ").lower()

        if user_input not in ['random forest', 'isolation forest']:
            print("Error: Invalid input. Please enter 'random forest' or 'isolation forest'.")
            sys.exit(1)

        if user_input == 'random forest':
            random_forest_state()
        elif user_input == 'isolation forest':
            isolation_forest_state()

        user_choice = input("Do you want to build a new model? [y/n] ").lower()
        if user_choice not in ['y', 'n']:
            print("Error: Invalid input. Please enter 'y' or 'n'.")
            sys.exit(1)


def main():
    """
    Main entry point of the application. 
    Handles user input, validation, and orchestrates the preprocessing pipeline.

    :internal user_choice: Tracks whether the user wants to continue or exit.
    :internal user_input: Captures the path and dataset type provided by the user.
    """
    user_choice = input("Do you want to start the preprocessing? [y/n] ").lower()

    if user_choice not in ['y', 'n']:
        print("Error: Invalid input. Please enter 'y' or 'n'.")
        sys.exit(1)

    if user_choice == 'y':
        preprocessing_loop(user_choice)

    user_choice = input("Do you want to build a new model? [y/n] ").lower()

    if user_choice not in ['y', 'n']:
        print("Error: Invalid input. Please enter 'y' or 'n'.")
        sys.exit(1)

    if user_choice == 'y':
        build_model_loop(user_choice)

    

    

if __name__ == "__main__":
    main()