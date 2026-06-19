"""
Centralized management of project paths.
"""
import pathlib

# Project Root (relative to src/utils/paths.py)
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

# --- Main folders ---
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"
INDEPENDENT_DIR = DATA_DIR / "independent_datasets"
DEPENDENT_DIR = DATA_DIR / "dependent_datasets"

# --- Raw dataset paths ---
NB15_RAW_PATH = DATA_DIR / "nb15.csv"
SAT20_RAW_PATH = DATA_DIR / "sat20.csv"
TER20_RAW_PATH = DATA_DIR / "ter20.csv"

# --- Models ---
RF_MODELS_SAVED_DIR = SRC_DIR / "models" / "random_forest_saved"
IF_MODELS_SAVED_DIR = SRC_DIR / "models" / "isolation_forest_saved"

# --- Classification ---
CLASSIFICATION_DIR = SRC_DIR / "classification"
INDEPENDENT_RESULTS_DIR = CLASSIFICATION_DIR / "independent"
DEPENDENT_RESULTS_DIR = CLASSIFICATION_DIR / "dependent"

# --- Structured dataset directory names ---
JOINT_DIR_NAME = "joint_preprocessed"

# --- Helper to ensure directories exist ---
def setup_project_directories():
    """
    Ensures that all necessary directories exist before running the pipeline.
    Call this at the very start of main().
    """
    directories_to_create = [
        INDEPENDENT_DIR,
        DEPENDENT_DIR,
        RF_MODELS_SAVED_DIR / "independent",
        RF_MODELS_SAVED_DIR / "dependent",
        IF_MODELS_SAVED_DIR / "independent",
        IF_MODELS_SAVED_DIR / "dependent",
        INDEPENDENT_RESULTS_DIR,
        DEPENDENT_RESULTS_DIR
    ]
    
    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)

# --- Filenames and Constants ---
# (Keeping your original constants below for compatibility)
RF_INFO_FILENAME = "random_forest_models_info.csv"
IF_INFO_FILENAME = "isolation_forest_models_info.csv"
RF_RESULTS_FILENAME = "random_forest_classification_results.csv"
IF_RESULTS_FILENAME = "isolation_forest_classification_results.csv"

# Preprocessed suffixes
PREPROCESSED_DIR_SUFFIX = "_preprocessed"