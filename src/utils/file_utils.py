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


def append_data_to_csv(results_dict, file_path):
    new_entry = pd.DataFrame([results_dict])
    if not file_path.exists():
        new_entry.to_csv(file_path, index=False)
    else:
        new_entry.to_csv(file_path, mode='a', header=False, index=False)
    

def get_model_info(file_path):
    try:
        # Estraiamo l'ID numerico dalla fine del nome del file (es. random_forest_model_10 -> 10)
        model_id = int(str(file_path.stem).split('_')[-1])
    except (ValueError, IndexError):
        print(f"Error: Could not extract ID from model filename '{file_path.name}'. Expected format: name_ID.joblib")
        sys.exit(1)

    # Cerchiamo l'unico file che contiene 'info.csv' nella stessa cartella
    info_files = list(pathlib.Path(file_path).resolve().parent.glob('*info.csv'))
    
    if not info_files:
        print(f"Error: No info CSV file found in {file_path.parent}")
        sys.exit(1)

    # Poiché c'è un solo file, prendiamo il primo della lista
    df = get_data_from_csv(info_files[0])

    # Cerchiamo la riga che corrisponde all'ID del modello salvato nel CSV
    model_info = df[df['id'] == model_id]

    if model_info.empty:
        print(f"Error: Model ID {model_id} not found in {info_files[0]}")
        sys.exit(1)

    # Restituiamo la riga come Series per permettere l'accesso diretto via chiave (es. model_info['attack_cat'])
    return model_info.iloc[0]
