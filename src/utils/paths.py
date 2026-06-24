"""
Centralized management of project paths.
"""
import pathlib

NB15_PREFIX = "nb15"
STIN_PREFIX = "stin"
SAT20_PREFIX = "sat20"
TER20_PREFIX = "ter20"

NB15_STIN_PREFIX = "nb15_stin"
NB15_SAT20_PREFIX = "nb15_sat20"
NB15_TER20_PREFIX = "nb15_ter20"

RF_MODEL_PREFIX = "rf_model_"

PREPROCESSED_SUFFIX = "_prep"
PREPROCESSED_AGGREGATE_SUFFIX = "_prep_aggr"
PREPROCESSED_AGGREGATE_SCALED_SUFFIX = "_prep_aggr_scaled"

UNNORMALIZED = "unnorm"
NORMALIZED = "norm"

DATA_FILE_TYPE = ".csv"

# ------ FOLDERS ------
# --- Root folder ---
# It depends from where `path.py` is saved
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

# --- Main folders ---
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"

# --- Sub folders ---
# data/
NB15_PREPROCESSED_DIR = DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}"
SAT20_PREPROCESSED_DIR = DATA_DIR / f"{SAT20_PREFIX}{PREPROCESSED_SUFFIX}"
TER20_PREPROCESSED_DIR = DATA_DIR / f"{TER20_PREFIX}{PREPROCESSED_SUFFIX}"

# src/
CLASSIFICATION_DIR = SRC_DIR / "classification"
MODELS_DIR = SRC_DIR / "models"
PLOTTING_DIR = SRC_DIR / "plotting"
PREPROCESSING_DIR = SRC_DIR / "preprocessing"
UTILS_DIR = SRC_DIR / "utils"

# --- Sub Sub folders --- 
# data/{dataset type}{preprocessed suffix}/
CLASS_DIR_NAME = "class"
NORMAL_ANOMALY_DIR_NAME = "normal_anomaly"

# src/classification/
UNNORMALIZED_CLASSIFICATION_DIR = CLASSIFICATION_DIR / UNNORMALIZED
NORMALIZED_CLASSIFICATION_DIR = CLASSIFICATION_DIR / NORMALIZED

# src/models/
UNNORMALIZED_MODELS_DIR = MODELS_DIR / UNNORMALIZED
NORMALIZED_MODELS_DIR = MODELS_DIR / NORMALIZED

# src/plotting/
UNNORMALIZED_PLOTS_DIR = PLOTTING_DIR / UNNORMALIZED
NORMALIZED_PLOTS_DIR = PLOTTING_DIR / NORMALIZED


# ------ FILES ------
# --- Raw dataset ---
# data/
NB15_RAW_PATH = DATA_DIR / f"{NB15_PREFIX}{DATA_FILE_TYPE}"
SAT20_RAW_PATH = DATA_DIR / f"{SAT20_PREFIX}{DATA_FILE_TYPE}"
TER20_RAW_PATH = DATA_DIR / f"{TER20_PREFIX}{DATA_FILE_TYPE}"



# ------ MODELS ------















# ------ CLASSIFICATION ------

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


# General file info
GENERAL_FILE_INFO_FILENAME = "general_file_info.csv"
GENERAL_FILE_INFO_PATH = DATA_DIR / GENERAL_FILE_INFO_FILENAME