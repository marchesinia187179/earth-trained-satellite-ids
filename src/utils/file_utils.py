import pandas as pd
import pathlib

def get_data_from_csv(file_path):
    """Legge un file CSV e restituisce un DataFrame."""
    return pd.read_csv(file_path, low_memory=False)


def create_csv_from_data(data, file_name, file_path):
    """Crea un file CSV a partire da un DataFrame o dati nella directory specificata."""
    df = pd.DataFrame(data)
    # Assicuriamoci che il nome del file abbia l'estensione corretta
    if not str(file_name).endswith('.csv'):
        file_name = f"{file_name}.csv"
    
    full_path = pathlib.Path(file_path) / file_name
    df.to_csv(full_path, index=False, encoding='utf-8')
    
    return full_path


def create_directory(dir_name, parent_path=None):
    """Crea una directory dato il nome e il path genitore."""
    if parent_path:
        path = pathlib.Path(parent_path) / dir_name
    else:
        path = pathlib.Path.cwd() / dir_name
    path.mkdir(parents=True, exist_ok=True)

    return path
