import pandas as pd
import pathlib
import sys

def get_data_from_csv(file_path):
    """
    Reads a CSV file and returns a pandas DataFrame.

    :param file_path: Path to the CSV file to be read.
    :return: A pandas DataFrame containing the loaded data.
    """
    try:
        return pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error: Could not read file at {file_path}. Details: {e}")
        sys.exit(1)


def create_csv_from_data(data, file_name, file_path):
    """
    Creates a CSV file from a DataFrame or raw data in the specified directory.

    :param data: Data to save (DataFrame, dict, or list).
    :param file_name: Output filename (without or with .csv).
    :param file_path: Directory path where the file will be saved.
    :return: The full Path object to the created CSV file.
    """
    df = pd.DataFrame(data)
    if not str(file_name).endswith('.csv'):
        file_name = f"{file_name}.csv"
    
    full_path = pathlib.Path(file_path) / file_name
    df.to_csv(full_path, index=False, encoding='utf-8')
    
    return full_path


def create_directory(dir_name, parent_path=None):
    """
    Creates a directory given its name and an optional parent path.

    :param dir_name: Name of the directory to create.
    :param parent_path: Optional parent directory. Defaults to Current Working Directory.
    :return: The Path object of the created directory.
    """
    if parent_path:
        path = pathlib.Path(parent_path) / dir_name
    else:
        path = pathlib.Path.cwd() / dir_name
    path.mkdir(parents=True, exist_ok=True)

    return path
