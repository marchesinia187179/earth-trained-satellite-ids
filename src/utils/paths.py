"""
Centralized management of project paths.
"""
import pathlib

# --- Root folder ---
# It depends from where `path.py` is saved
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent


# --- Main folders ---
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"


# --- Sub folders ---
# data/
UNNORMALIZED_DATASETS_DIR = DATA_DIR / "unnormalized"
NORMALIZED_DATASETS_DIR = DATA_DIR / "normalized"

# src/
CLASSIFICATION_DIR = SRC_DIR / "classification"
MODELS_DIR = SRC_DIR / "models"
PLOTTING_DIR = SRC_DIR / "plotting"
PREPROCESSING_DIR = SRC_DIR / "preprocessing"
UTILS_DIR = SRC_DIR / "utils"


# --- Sub Sub folders --- 
# data/unnormalized/


# data/normalized/


# src/classification/
UNNORMALIZED_CLASSIFICATION_DIR = CLASSIFICATION_DIR / "unnormalized"
NORMALIZED_CLASSIFICATION_DIR = CLASSIFICATION_DIR / "normalized"

# src/models/
UNNORMALIZED_MODELS_DIR = MODELS_DIR / "unnormalized"
NORMALIZED_MODELS_DIR = MODELS_DIR / "normalized"

# src/plotting/
UNNORMALIZED_PLOTS_DIR = PLOTTING_DIR / "unnormalized"
NORMALIZED_PLOTS_DIR = PLOTTING_DIR / "normalized"

# src/preprocessing/
# Nothing for the moment

# src/utils/
# Nothing for the moment








# --- Raw dataset paths ---
NB15_RAW_PATH = DATA_DIR / "nb15.csv"
SAT20_RAW_PATH = DATA_DIR / "sat20.csv"
TER20_RAW_PATH = DATA_DIR / "ter20.csv"














# Project Root (relative to src/utils/paths.py)









# --- Directory and file naming constants ---
ATTACK_CAT_DIR_NAME = "attack_cat"
NORMAL_ATTACK_DIR_NAME = "normal_attack"
JOINT_DIR_NAME = "joint_preprocessed"

# Specific preprocessed filenames (stem only, .csv added by create_csv_from_data)
NB15_SCALED_FILE_STEM = "nb15_preprocessed_scaled"
NB15_FILE_STEM = "nb15_preprocessed"
NB15_ATTACKS_SCALED_FILE_STEM = "nb15_attacks_preprocessed_scaled"
SAT20_FILE_STEM = "sat20_preprocessed"
SAT20_ATTACKS_SCALED_FILE_STEM = "sat20_attacks_preprocessed_scaled"
TER20_FILE_STEM = "ter20_preprocessed"
TER20_ATTACKS_SCALED_FILE_STEM = "ter20_attacks_preprocessed_scaled"
NORMAL_FILE_STEM = "Normal"
JOINT_NORMAL_SAT20_FILE_STEM = "Normal_sat20"
JOINT_NORMAL_TER20_FILE_STEM = "Normal_ter20"

# Preprocessed dataset directory names
NB15_PREPROCESSED_DIR_NAME = "nb15_preprocessed"
SAT20_PREPROCESSED_DIR_NAME = "sat20_preprocessed"
TER20_PREPROCESSED_DIR_NAME = "ter20_preprocessed"
JOINT_DIR_NAME = "joint_preprocessed"

# General suffix for preprocessed dataset directories and main files
PREPROCESSED_DIR_SUFFIX = "_preprocessed"
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

BY_MODEL_DIR_NAME = "by_model"
BY_DATASET_DIR_NAME = "by_dataset"
TESTING_DATASET_DIR_NAME = "testing_datasets"
DATASET_RESULTS_SUFFIX = "_classification_results.csv"
TESTING_DATASET_RESULTS_SUFFIX = "_classification_results.csv"
INDEPENDENT_RESULTS_MODELS_DIR = INDEPENDENT_RESULTS_DIR / BY_MODEL_DIR_NAME
DEPENDENT_RESULTS_MODELS_DIR = DEPENDENT_RESULTS_DIR / BY_MODEL_DIR_NAME
INDEPENDENT_RESULTS_DATASETS_DIR = INDEPENDENT_RESULTS_DIR / BY_DATASET_DIR_NAME
DEPENDENT_RESULTS_DATASETS_DIR = DEPENDENT_RESULTS_DIR / BY_DATASET_DIR_NAME

# --- Helper to ensure directories exist ---
def setup_project_directories():
    """
    Ensures that all necessary directories exist before running the pipeline.
    Call this at the very start of main().
    """
    directories_to_create = [
        RF_MODELS_SAVED_DIR / "independent",
        RF_MODELS_SAVED_DIR / "dependent",
        IF_MODELS_SAVED_DIR / "independent",
        IF_MODELS_SAVED_DIR / "dependent",
        INDEPENDENT_RESULTS_DIR,
        DEPENDENT_RESULTS_DIR,
        INDEPENDENT_RESULTS_MODELS_DIR,
        DEPENDENT_RESULTS_MODELS_DIR,
        INDEPENDENT_RESULTS_DATASETS_DIR,
        DEPENDENT_RESULTS_DATASETS_DIR
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