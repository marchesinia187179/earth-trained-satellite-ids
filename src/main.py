import joblib

from models.models import model_processing
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


def build_model_loop(user_choice):
    while user_choice == 'y':

        user_input = input("Choose which model do you want to use: [random forest or isolation forest] ").lower()

        if user_input not in ['random forest', 'isolation forest']:
            print("Error: Invalid input. Please enter 'random forest' or 'isolation forest'.")
            sys.exit(1)

        data_path_str = input("Insert the path of the dataset: [path] ")
        data_path = pathlib.Path(data_path_str)
        
        if not data_path.exists():
            print(f"Error: The file path '{data_path_str}' does not exist.")
            sys.exit(1)

        data = get_data_from_csv(data_path)
        data = pd.DataFrame(data)

        model_processing(data, user_input)

        user_choice = input("Do you want to build a new model? [y/n] ").lower()
        if user_choice not in ['y', 'n']:
            print("Error: Invalid input. Please enter 'y' or 'n'.")
            sys.exit(1)


def classification_loop(user_choice):
    while user_choice == 'y':

        user_input = input("Insert the path of the dataset: [path] ")
        data_path = pathlib.Path(user_input)

        if not data_path.exists():
            print(f"Error: The file path '{user_input}' does not exist.")
            sys.exit(1)

        data = get_data_from_csv(data_path)
        data = pd.DataFrame(data)
        
        user_input = input("Insert the path of the model: [path] ")
        model_path = pathlib.Path(user_input)

        if not model_path.exists():
            print(f"Error: The file path '{user_input}' does not exist.")
            sys.exit(1)

        model = joblib.load(model_path)










        


        user_choice = input("Do you want to start a new classification? [y/n] ").lower()
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

    user_choice = input("Do you want to build a model? [y/n] ").lower()

    if user_choice not in ['y', 'n']:
        print("Error: Invalid input. Please enter 'y' or 'n'.")
        sys.exit(1)

    if user_choice == 'y':
        build_model_loop(user_choice)

    user_choice = input("Do you want to start the classification? [y/n] ").lower()

    if user_choice not in ['y', 'n']:
        print("Error: Invalid input. Please enter 'y' or 'n'.")
        sys.exit(1)

    if user_choice == 'y':
        classification_loop(user_choice)

    

    

if __name__ == "__main__":
    main()