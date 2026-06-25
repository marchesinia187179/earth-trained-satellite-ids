"""
Centralized management of project paths.
"""
import pathlib

RANDOM_STATE = 42
NORMAL_ANOMALY_RATIO = 10
TRAIN_SPLIT = 0.8
MODEL_VERBOSE = 2

NB15_PREFIX = "nb15"
SAT20_PREFIX = "sat20"
TER20_PREFIX = "ter20"

HYBRID_PREFIX = "hybrid"
NB15_SAT20_PREFIX = "nb15_sat20"
NB15_TER20_PREFIX = "nb15_ter20"

RF_MODEL_PREFIX = "rf_model_"

PREPROCESSED_SUFFIX = "_prep"
PREPROCESSED_SCALED_SUFFIX = "_prep_scaled"

UNNORMALIZED = "unnormalized"
NORMALIZED = "normalized"

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
SINGLE_CLASSES_DIR_NAME = "single_classes"
NORMAL_ANOMALY_DIR_NAME = "normal_anomaly"
SCALED_DIR_NAME = "scaled"

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





GENERAL_FILE_INFO_FILENAME = "general_file_info.csv"
GENERAL_FILE_INFO_PATH = DATA_DIR / GENERAL_FILE_INFO_FILENAME









"""

# ------ CLASSIFICATION ------

BY_MODEL_DIR_NAME = "by_model"
BY_DATASET_DIR_NAME = "by_dataset"
TESTING_DATASET_DIR_NAME = "testing_datasets"
DATASET_RESULTS_SUFFIX = "_classification_results.csv"
TESTING_DATASET_RESULTS_SUFFIX = "_classification_results.csv"


# --- Helper to ensure directories exist ---
def setup_project_directories():
    
    Ensures that all necessary directories exist before running the pipeline.
    Call this at the very start of main().
    
    directories_to_create = [
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



"""