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

# --- Directory and file naming constants ---
ATTACK_CAT_DIR_NAME = "attack_cat"
NORMAL_ATTACK_DIR_NAME = "normal_attack"
JOINT_DIR_NAME = "joint_preprocessed"

# Specific preprocessed filenames (stem only, .csv added by create_csv_from_data)
NB15_SCALED_FILE_STEM = "nb15_preprocessed_scaled"
NB15_FILE_STEM = "nb15_preprocessed"  # Aggiungi questo se mancante per coerenza
ATTACKS_FILE_STEM = "Attacks"
NORMAL_FILE_STEM = "Normal" # For attack_cat/Normal.csv
JOINT_NORMAL_SAT20_FILE_STEM = "Normal_sat20"
JOINT_NORMAL_TER20_FILE_STEM = "Normal_ter20"

# Preprocessed dataset directory names
NB15_PREPROCESSED_DIR_NAME = "nb15_preprocessed"
SAT20_PREPROCESSED_DIR_NAME = "sat20_preprocessed"
TER20_PREPROCESSED_DIR_NAME = "ter20_preprocessed"

# General suffix for preprocessed dataset directories and main files
PREPROCESSED_DIR_SUFFIX = "_preprocessed" # e.g., nb15_preprocessed
PREPROCESSED_MAIN_FILE_SUFFIX = "_preprocessed"

# --- Models ---
RF_MODELS_SAVED_DIR = SRC_DIR / "models" / "random_forest_saved"
IF_MODELS_SAVED_DIR = SRC_DIR / "models" / "isolation_forest_saved"

# Mode-specific subdirectories for models
INDEPENDENT_RF_DIR = RF_MODELS_SAVED_DIR / "independent"
DEPENDENT_RF_DIR = RF_MODELS_SAVED_DIR / "dependent"
INDEPENDENT_IF_DIR = IF_MODELS_SAVED_DIR / "independent"
DEPENDENT_IF_DIR = IF_MODELS_SAVED_DIR / "dependent"

# --- Classification ---
CLASSIFICATION_DIR = SRC_DIR / "classification"
INDEPENDENT_RESULTS_DIR = CLASSIFICATION_DIR / "independent"
DEPENDENT_RESULTS_DIR = CLASSIFICATION_DIR / "dependent"

# --- Plots ---
PLOTS_DIR = CLASSIFICATION_DIR / "plots"
INDEPENDENT_PLOTS_DIR = PLOTS_DIR / "independent"
DEPENDENT_PLOTS_DIR = PLOTS_DIR / "dependent"

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
        DEPENDENT_RESULTS_DIR,
        PLOTS_DIR,
        INDEPENDENT_PLOTS_DIR,
        DEPENDENT_PLOTS_DIR
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

# General file info
GENERAL_FILE_INFO_FILENAME = "general_file_info.csv"
GENERAL_FILE_INFO_PATH = DATA_DIR / GENERAL_FILE_INFO_FILENAME