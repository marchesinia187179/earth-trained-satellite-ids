"""
Centralized management of project paths.
"""
import pathlib

# Project Root (relative to src/utils/paths.py)
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

# Main folders
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"
INDEPENDENT_DIR = DATA_DIR / "independent_datasets"
DEPENDENT_DIR = DATA_DIR / "dependent_datasets"

# Specific subfolders for models
RF_MODELS_SAVED_DIR = SRC_DIR / "models" / "random_forest_saved"
IF_MODELS_SAVED_DIR = SRC_DIR / "models" / "isolation_forest_saved"

# Specific subfolders for classification results
CLASSIFICATION_DIR = SRC_DIR / "classification"

# Mode-specific subdirectories for models
INDEPENDENT_RF_DIR = RF_MODELS_SAVED_DIR / "independent"
DEPENDENT_RF_DIR = RF_MODELS_SAVED_DIR / "dependent"
INDEPENDENT_IF_DIR = IF_MODELS_SAVED_DIR / "independent"
DEPENDENT_IF_DIR = IF_MODELS_SAVED_DIR / "dependent"

# Mode-specific subdirectories for classification results
INDEPENDENT_RESULTS_DIR = CLASSIFICATION_DIR / "independent"
DEPENDENT_RESULTS_DIR = CLASSIFICATION_DIR / "dependent"