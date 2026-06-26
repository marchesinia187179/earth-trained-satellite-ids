"""
Centralized management of project paths.
"""
import pathlib

RANDOM_STATE = 42
NORMAL_ANOMALY_RATIO = 10
TRAIN_SPLIT = 0.8
MODEL_VERBOSE = 1
DECIMAL_DIGITS = 4

NB15_PREFIX = "nb15"
SAT20_PREFIX = "sat20"
TER20_PREFIX = "ter20"

HYBRID_PREFIX = "hybrid"
NB15_SAT20_PREFIX = "nb15_sat20"
NB15_TER20_PREFIX = "nb15_ter20"

RF_MODEL_PREFIX = "model"
RF_INFO_FILENAME = "models_info"
CLASSIFICATIONS_FILENAME = "classifications"
CLASSIFICATION_SUFFIX = "_classification"

PREPROCESSED_SUFFIX = "_prep"
PREPROCESSED_SCALED_SUFFIX = "_prep_scaled"
NORMAL_ANOMALY_SUFFIX = "_normal_anomaly"

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
CLASSIFICATIONS_DIR = SRC_DIR / "classifications"
MODELS_DIR = SRC_DIR / "models"
PLOTTING_DIR = SRC_DIR / "plotting"
PREPROCESSING_DIR = SRC_DIR / "preprocessing"
UTILS_DIR = SRC_DIR / "utils"

# --- Sub Sub folders --- 
# data/{dataset type}{preprocessed suffix}/
SINGLE_CLASSES_DIR_NAME = "single_classes"
NORMAL_ANOMALY_DIR_NAME = "normal_anomaly"
SCALED_DIR_NAME = "scaled"
BY_DATASET_DIR_NAME = "by_dataset"
BY_MODEL_DIR_NAME = "by_model"
CLASSES_DIR_NAME = "classes"

# src/classification/
UNNORMALIZED_CLASSIFICATION_DIR = CLASSIFICATIONS_DIR / UNNORMALIZED
NORMALIZED_CLASSIFICATION_DIR = CLASSIFICATIONS_DIR / NORMALIZED

# src/models/
UNNORMALIZED_MODELS_DIR = MODELS_DIR / UNNORMALIZED
NORMALIZED_MODELS_DIR = MODELS_DIR / NORMALIZED
MODELS_PATHS_FILENAME = "models_paths"

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







# --- Datasets ---
DATASETS = [
    {'type': NB15_PREFIX, 'path': NB15_RAW_PATH},
    {'type': SAT20_PREFIX, 'path': SAT20_RAW_PATH},
    {'type': TER20_PREFIX, 'path': TER20_RAW_PATH}
]

# --- Routine Model Configuration ---
# Defines the standard set of models to be built during a routine phase
ROUTINE_MODELS = [
    # --- NB15 dataset ---
    # Aggregate
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{SCALED_DIR_NAME}/{NB15_PREFIX}{PREPROCESSED_SCALED_SUFFIX}{DATA_FILE_TYPE}")},
     
    # Single classes
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{NORMAL_ANOMALY_DIR_NAME}/Normal_DoS{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{NORMAL_ANOMALY_DIR_NAME}/Normal_Exploits{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{NORMAL_ANOMALY_DIR_NAME}/Normal_Fuzzers{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{NORMAL_ANOMALY_DIR_NAME}/Normal_Generic{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{NORMAL_ANOMALY_DIR_NAME}/Normal_Reconnaissance{DATA_FILE_TYPE}")},

    # --- Hybrid dataset ---
    # Aggregate
    {'dataset_type': (f"{HYBRID_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{SCALED_DIR_NAME}/{HYBRID_PREFIX}{PREPROCESSED_SCALED_SUFFIX}{DATA_FILE_TYPE}")},

    # Sub dataset
    {'dataset_type': (f"{NB15_SAT20_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{SCALED_DIR_NAME}/{NB15_SAT20_PREFIX}{PREPROCESSED_SCALED_SUFFIX}{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_TER20_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{SCALED_DIR_NAME}/{NB15_TER20_PREFIX}{PREPROCESSED_SCALED_SUFFIX}{DATA_FILE_TYPE}")}
]



# --- Routine Classification Configuration ---
# Defines the standard set of classifications to do during a routine phase
CLASSIFICATIONS = [
    # --- NB15 dataset ---
    # Aggregate
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{SCALED_DIR_NAME}/{NB15_PREFIX}{PREPROCESSED_SCALED_SUFFIX}{DATA_FILE_TYPE}")},
     
    # Normal_anomaly
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{NORMAL_ANOMALY_DIR_NAME}/Normal_DoS{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{NORMAL_ANOMALY_DIR_NAME}/Normal_Exploits{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{NORMAL_ANOMALY_DIR_NAME}/Normal_Fuzzers{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{NORMAL_ANOMALY_DIR_NAME}/Normal_Generic{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_PREFIX}"), 'path': (DATA_DIR / f"{NB15_PREFIX}{PREPROCESSED_SUFFIX}/{NORMAL_ANOMALY_DIR_NAME}/Normal_Reconnaissance{DATA_FILE_TYPE}")},

    # --- Hybrid dataset ---
    # Aggregate
    {'dataset_type': (f"{HYBRID_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{SCALED_DIR_NAME}/{HYBRID_PREFIX}{PREPROCESSED_SCALED_SUFFIX}{DATA_FILE_TYPE}")},

    # Sub dataset
    {'dataset_type': (f"{NB15_SAT20_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{SCALED_DIR_NAME}/{NB15_SAT20_PREFIX}{PREPROCESSED_SCALED_SUFFIX}{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_TER20_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{SCALED_DIR_NAME}/{NB15_TER20_PREFIX}{PREPROCESSED_SCALED_SUFFIX}{DATA_FILE_TYPE}")},

    # Normal_anomaly
    {'dataset_type': (f"{NB15_SAT20_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{NB15_SAT20_PREFIX}{NORMAL_ANOMALY_SUFFIX}/Normal_Syn_DDoS{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_SAT20_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{NB15_SAT20_PREFIX}{NORMAL_ANOMALY_SUFFIX}/Normal_UDP_DDoS{DATA_FILE_TYPE}")},

    {'dataset_type': (f"{NB15_TER20_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{NB15_TER20_PREFIX}{NORMAL_ANOMALY_SUFFIX}/Normal_Botnet{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_TER20_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{NB15_TER20_PREFIX}{NORMAL_ANOMALY_SUFFIX}/Normal_DDoS{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_TER20_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{NB15_TER20_PREFIX}{NORMAL_ANOMALY_SUFFIX}/Normal_Syn_DDoS{DATA_FILE_TYPE}")},
    {'dataset_type': (f"{NB15_TER20_PREFIX}"), 'path': (DATA_DIR / f"{HYBRID_PREFIX}{PREPROCESSED_SUFFIX}/{NB15_TER20_PREFIX}{NORMAL_ANOMALY_SUFFIX}/Normal_UDP_DDoS{DATA_FILE_TYPE}")}
]