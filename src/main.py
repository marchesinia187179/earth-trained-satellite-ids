import joblib

from classification.classification import classification_processing
from models.models import model_processing
from utils.file_utils import get_data_from_csv, get_model_info
from utils.input_utils import get_split_input, validate_path, validate_choice, get_y_n_choice
from preprocessing.data_preprocessing import data_preprocessing
from preprocessing.file_preprocessing import file_preprocessing
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
    print("\n--- Starting Preprocessing Phase ---")
    while user_choice == 'y':
        prompt = "Insert the path and the dataset_type of the dataset: [path dataset_type] (dataset_type: nb15, sat20 or ter20) "
        user_input = get_split_input(prompt, 2)
        
        path = validate_path(user_input[0])
        dataset_type = validate_choice(user_input[1], ['nb15', 'sat20', 'ter20'], "dataset type")

        data_preprocessed = data_preprocessing_state(path, dataset_type)
        file_preprocessing_state(data_preprocessed, dataset_type)

        user_choice = get_y_n_choice("Do you want to start a new preprocessing? [y/n] ")


def build_model_loop(user_choice):
    print("\n--- Starting Model Building Phase ---")
    while user_choice == 'y':
        model_input = input("Choose which model do you want to use: [random forest or isolation forest] ").lower()
        model_type = validate_choice(model_input, ['random forest', 'isolation forest'], "model type")

        path_input = input("Insert the path of the dataset: [path] ")
        data_path = validate_path(path_input)

        data = get_data_from_csv(data_path)

        model_processing(data, model_type)

        user_choice = get_y_n_choice("Do you want to build a new model? [y/n] ")


def classification_loop(user_choice):
    print("\n--- Starting Classification Phase ---")
    while user_choice == 'y':
        path_input = input("Insert the path of the dataset: [path] ")
        data_path = validate_path(path_input)

        data = get_data_from_csv(data_path)
        
        model_path_input = input("Insert the path of the model: [path] ")
        model_path = validate_path(model_path_input)

        model = joblib.load(model_path)
        model_info = get_model_info(model_path)

        classification_processing(data, model, model_path.stem, model_info["attack_cat"], data_path.stem)

        user_choice = get_y_n_choice("Do you want to start a new classification? [y/n] ")


def main():
    """
    Main entry point of the application. 
    Handles user input, validation, and orchestrates the preprocessing pipeline.

    :internal user_choice: Tracks whether the user wants to continue or exit.
    :internal user_input: Captures the path and dataset type provided by the user.
    """
    user_choice = get_y_n_choice("Do you want to start the preprocessing? [y/n] ")
    if user_choice == 'y':
        preprocessing_loop(user_choice)

    user_choice = get_y_n_choice("Do you want to build a model? [y/n] ")
    if user_choice == 'y':
        build_model_loop(user_choice)

    user_choice = get_y_n_choice("Do you want to start the classification? [y/n] ")
    if user_choice == 'y':
        classification_loop(user_choice)


if __name__ == "__main__":
    main()