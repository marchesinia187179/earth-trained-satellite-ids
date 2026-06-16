import pathlib

# Root del progetto (rispetto a src/utils/paths.py)
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

# Cartelle principali
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"

# Sottocartelle specifiche
RF_MODELS_SAVED_DIR = SRC_DIR / "models" / "random_forest_saved"
IF_MODELS_SAVED_DIR = SRC_DIR / "models" / "isolation_forest_saved"
RESULTS_DIR = SRC_DIR / "classification"

# File specifici (opzionale)
RF_INFO_CSV = RF_MODELS_SAVED_DIR / "random_forest_models_info.csv"
IF_INFO_CSV = IF_MODELS_SAVED_DIR / "isolation_forest_models_info.csv"

# File per i risultati della classificazione
RF_CLASSIFICATION_RESULTS_CSV = RESULTS_DIR / "random_forest_classification_results.csv"
IF_CLASSIFICATION_RESULTS_CSV = RESULTS_DIR / "isolation_forest_classification_results.csv"