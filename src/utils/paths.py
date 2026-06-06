import pathlib

# Root del progetto (rispetto a src/utils/paths.py)
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

# Cartelle principali
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"

# Sottocartelle specifiche
MODELS_SAVED_DIR = SRC_DIR / "models" / "random_forest_saved"
RESULTS_DIR = SRC_DIR / "classification"

# File specifici (opzionale)
RF_INFO_CSV = MODELS_SAVED_DIR / "random_forest_models_info.csv"