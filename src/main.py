from utils.file_utils import *
from preprocessing.data_preprocessing import *
from preprocessing.file_preprocessing import *


def file_preprocessing_state(data_preprocessed, type):
    print(f"File preprocessing for {type} started...")
    file_preprocessing(data_preprocessed, type)
    print(f"File preprocessing for {type} completed successfully!")


def data_preprocessing_state(path, type):
    print(f"Data preprocessing for {type} started...")
    data = get_data_from_csv(path)
    data_preprocessed = data_preprocessing(data, type)
    print(f"Data preprocessing for {type} completed successfully!")

    return data_preprocessed


def main():
    print("Do you want to start the preprocessing? [y/n]")
    input = input()

    while input == 'y':
        print("Insert the path and the type of the dataset: [path type] (type: nb15, sat20 or ter20)")
        path, type = input().split()

        data_preprocessed = data_preprocessing_state(path, type)
        file_preprocessing_state(data_preprocessed, type)

        print(f"Preprocessing for {type} completed successfully!")
        print("Do you want to start a new preprocessing? [y/n]")
        input = input()


if __name__ == "__main__":
    main()