"""
Utility functions for file handling and CSV operations.
"""
import pandas as pd
import pathlib
from datetime import datetime
import sys
from utils.paths import CLASSES_DIR_NAME, CLASSIFICATION_SUFFIX, GENERAL_FILE_INFO_PATH, RANDOM_STATE, ROOT_DIR


def update_or_append_csv(file_path, data_dict, match_keys, id_column='id'):
    """
    Aggiorna una riga esistente in un file CSV se i campi in match_keys corrispondono,
    altrimenti genera un nuovo ID incrementale (sulla colonna specificata) e appende i dati.
    Mantiene sempre id_column come prima colonna.
    """
    if file_path.exists():
        try:
            # Carica il file trattando 'None' come nullo
            df = pd.read_csv(file_path, na_values='None')
            
            if not df.empty:
                # Costruisce la maschera booleana per verificare i duplicati
                mask = True
                for key in match_keys:
                    mask &= (df[key] == data_dict[key])
                
                if mask.any():
                    # Aggiorna la riga preservando l'ID originale
                    idx = df.index[mask][0]
                    data_dict[id_column] = df.at[idx, id_column]
                    
                    for key, value in data_dict.items():
                        df.at[idx, key] = value
                        
                    # --- MODIFICA: Forza l'ID come prima colonna nell'aggiornamento ---
                    cols = [id_column] + [c for c in df.columns if c != id_column]
                    df = df[cols]
                    # ------------------------------------------------------------------
                    
                    df.to_csv(file_path, index=False, na_rep='None')
                    print(f"Updated existing entry in: {file_path.name}")
                    return
                
                # Genera un nuovo ID incrementale basato sul massimo esistente
                max_id = df[id_column].max()
                data_dict[id_column] = int(max_id + 1) if pd.notnull(max_id) else 1
            else:
                data_dict[id_column] = 1
        except Exception as e:
            print(f"Warning: Could not parse {file_path.name} for update ({e}). Appending as new.")
            data_dict[id_column] = 1
    else:
        data_dict[id_column] = 1

    # Accoda la nuova riga
    new_entry = pd.DataFrame([data_dict])
    
    # --- MODIFICA: Forza l'ID come prima colonna prima di salvare la nuova riga ---
    ordered_cols = [id_column] + [c for c in new_entry.columns if c != id_column]
    new_entry = new_entry[ordered_cols]
    # ------------------------------------------------------------------------------

    if not file_path.exists():
        new_entry.to_csv(file_path, index=False, na_rep='None')
    else:
        # --- MODIFICA SICUREZZA: Allinea le colonne per l'append (mode='a' ignora gli header) ---
        if 'df' in locals() and not df.empty:
            existing_cols = [id_column] + [c for c in df.columns if c != id_column]
            # Assicurati che le colonne di new_entry matchino esattamente quelle del file esistente
            for col in existing_cols:
                if col not in new_entry.columns:
                    new_entry[col] = None
            new_entry = new_entry[existing_cols]
        # ----------------------------------------------------------------------------------------
        
        new_entry.to_csv(file_path, mode='a', header=False, index=False, na_rep='None')
        
    print(f"Data appended/created in: {file_path.name}")


def get_data_from_csv(file_path):
    """
    Reads a CSV file and returns a pandas DataFrame.

    :param file_path: Path to the CSV file to be read.
    :return: A pandas DataFrame containing the loaded data.
    """
    try:
        print(f"Loading CSV: {file_path.name}")
        return pd.read_csv(file_path, low_memory=False, na_values='None')
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
    df.to_csv(full_path, index=False, encoding='utf-8', na_rep='None')
    print(f"Created file: {full_path.name}")

    file_info = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': full_path.name,
        'relative_path': str(full_path.relative_to(ROOT_DIR)),
        'rows': df.shape[0],
        'columns': df.shape[1],
    }

    print(f"  - Data shape: {df.shape}")
    if 'class' in df.columns:
        counts = df['class'].value_counts()
        file_info['class_distribution'] = str(counts.to_dict())
        print(f"  - Class distribution (class):\n{counts.to_string()}")
    else:
        file_info['class_distribution'] = 'None'
    if 'class' in df.columns and 'split_type' in df.columns:
        pivot = df.groupby(['class', 'split_type']).size().unstack(fill_value=0)
        file_info['train_test_distribution'] = str(pivot.to_dict())
        print(f"  - Train/Test division per class:\n{pivot.to_string()}")
    else:
        file_info['train_test_distribution'] = 'None'
    print("-" * 30)
    
    match_keys = ['relative_path']
    update_or_append_csv(GENERAL_FILE_INFO_PATH, file_info, match_keys, id_column='id')
    
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
    print(f"Directory ready: {path.relative_to(path.parents[1])}")

    return path


def append_data_to_csv(results_dict, file_path):
    """
    Appends a single row of data (as a dictionary) to an existing CSV or creates a new one.
    Assicura che, se l'ID è presente, venga sempre posizionato come prima colonna.
    """
    new_entry = pd.DataFrame([results_dict])
    
    # --- MODIFICA: Verifica se esiste una chiave 'id' (o affini) e la forza al primo posto ---
    if 'id' in new_entry.columns:
        ordered_cols = ['id'] + [c for c in new_entry.columns if c != 'id']
        new_entry = new_entry[ordered_cols]
    # ------------------------------------------------------------------------------------------

    if not file_path.exists():
        new_entry.to_csv(file_path, index=False, na_rep='None')
    else:
        new_entry.to_csv(file_path, mode='a', header=False, index=False, na_rep='None')
    

def get_model_info(file_path):
    """
    Retrieves metadata for a specific model from its associated 'info.csv' file.

    :param file_path: Path to the saved model (.joblib).
    :return: A pandas Series containing the model's training metadata.
    """
    try:
        # Search for the unique file containing 'info.csv' in the same folder
        info_files = list(pathlib.Path(file_path).resolve().parent.glob('*info.csv'))
        
        if not info_files:
            print(f"Error: No info CSV file found in {file_path.parent}")
            sys.exit(1)

        # Since there is only one file, take the first one
        df = get_data_from_csv(info_files[0])

        # Search for the row corresponding to the model name saved in the CSV
        model_info = df[df['model_name'] == file_path.stem]

        if model_info.empty:
            print(f"Error: Model '{file_path.stem}' not found in {info_files[0]}")
            sys.exit(1)

        # Return the row as a Series to allow direct access via key
        return model_info.iloc[0]
    except Exception as e:
        print(f"Error retrieving model info: {e}")
        sys.exit(1)



def concat_and_shuffle(data_list):
    return pd.concat(data_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def group_by_model_and_save(data, dst_dir):
    """
    Groups the data by model and save it in CSV file

    :param data: data to be grouped
    :param dst_dir: directory path where model-specific results will be saved
    """
    # Get data
    data_df = get_data_from_csv(data)

    # Group and save data by model_name
    for model_name, group in data_df.groupby('model_name'):
        create_csv_from_data(group, f"{model_name}{CLASSIFICATION_SUFFIX}", dst_dir)


def group_by_classes_and_save(data, dst_dir):
    """
    Groups the data by dataset type/classes and saves it into CSV file

    :param data: data to be grouped
    :param dst_dir: directory path where type/classes-specific results will be saved
    """
    # Get data
    data_df = get_data_from_csv(data)

    # Group and save data by classes for each dataset_type
    for dataset_type, group in data_df.groupby('dataset_type'):
        curr_dataset_dir = create_directory(dataset_type, dst_dir)
        
        for classes_name, classes_group in group.groupby('classes'):
            curr_classes_dir = create_directory(CLASSES_DIR_NAME, curr_dataset_dir)
            create_csv_from_data(classes_group, f"{classes_name}{CLASSIFICATION_SUFFIX}", curr_classes_dir)

