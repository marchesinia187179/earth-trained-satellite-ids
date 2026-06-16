"""
Centralized management of project paths.
"""
import pathlib

# Project Root (relative to src/utils/paths.py)
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

# Main folders
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"

# Specific subfolders
RF_MODELS_SAVED_DIR = SRC_DIR / "models" / "random_forest_saved"
IF_MODELS_SAVED_DIR = SRC_DIR / "models" / "isolation_forest_saved"
RESULTS_DIR = SRC_DIR / "classification"

# Specific files (optional)
RF_INFO_CSV = RF_MODELS_SAVED_DIR / "random_forest_models_info.csv"
IF_INFO_CSV = IF_MODELS_SAVED_DIR / "isolation_forest_models_info.csv"

# Classification results files
RF_CLASSIFICATION_RESULTS_CSV = RESULTS_DIR / "random_forest_classification_results.csv"
IF_CLASSIFICATION_RESULTS_CSV = RESULTS_DIR / "isolation_forest_classification_results.csv"