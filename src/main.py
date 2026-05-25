from utils.file_utils import *
from preprocessing.data_preprocessing import *
from preprocessing.file_preprocessing import *
import sys
import pathlib


def file_preprocessing_state(data_preprocessed, dataset_type):
    file_preprocessing(data_preprocessed, dataset_type)


def data_preprocessing_state(path, dataset_type):
    data = get_data_from_csv(path)
    data_preprocessed = data_preprocessing(data, dataset_type)

    return data_preprocessed


def main():
    user_choice = input("Do you want to start the preprocessing? [y/n] ").lower()

    if user_choice not in ['y', 'n']:
        print("Error: Invalid input. Please enter 'y' or 'n'.")
        sys.exit(1)

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

if __name__ == "__main__":
    main()