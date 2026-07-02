"""
Centralized management of project paths, ML constants, and routine configurations.
"""

import pathlib

class MLConstants:
    """ Hyperparameters and standard metrics for the ML pipeline """

    RANDOM_STATE = 42
    NORMAL_ANOMALY_RATIO = 10
    TRAIN_SPLIT = 0.8
    MODEL_VERBOSE = 0
    DECIMAL_DIGITS = 4
    PLOTTING_METRICS = ['F1-Score', 'Precision', 'Recall']


class Naming:
    """ Standardized prefixes, suffixes, and extensions for dataset and model files """

    EXT = ".csv"
    PLOT_EXT = ".png"
    
    # Dataset Prefixes
    NB15 = "nb15"
    SAT20 = "sat20"
    TER20 = "ter20"
    HYBRID = "hybrid"
    NB15_SAT20 = "nb15_sat20"
    NB15_TER20 = "nb15_ter20"

    # Suffixes
    PREP = "_prep"
    PREP_SCALED = "_prep_scaled"
    CLASSIFICATION = "_classification"

    # Common File Names
    MODEL_INFO = f"models_info{EXT}"
    CLASSIFICATIONS = f"classifications{EXT}"
    MODELS_PATHS = f"models_registry{EXT}"
    MODEL = "model"     # It doesn't have an extension because it can be .pkl, .joblib, etc. depending on the model type
    FEATURE_IMPORTANCE = f"feature_importance"  # It doesn't have an extension because it can be .csv or .png depending on the context
    FEATURE_IMPORTANCE_BY_PERMUTATION = f"feature_importance_by_permutation{EXT}"


class ProjectPaths:
    """ Absolute Pathlib structures for project directories and core files """

    # --- Main Folders ---
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    DATA = ROOT / "data"
    SRC = ROOT / "src"
    MODELS = ROOT / "models"
    RESULTS = ROOT / "results"

    # --- DATA Subfolders ---
    RAW_DATA_DIR = DATA / "raw"
    PREP_DATA_DIR = DATA / "preprocessed"
    METADATA_DIR = PREP_DATA_DIR / "metadata"

    # --- RESULTS Subfolders ---
    RESULTS_PLOT_DIR = RESULTS / "plots"
    RESULTS_CSV_DIR = RESULTS / "csv"

    # --- SRC Subforlders ---
    CLASSIFICATIONS_DIR = SRC / "classifications"
    PLOTTING_DIR = SRC / "plotting"

    # --- Folder Name Constants (For dynamic path building) --- 
    DIR_SINGLE_CLASSES = "single_classes"
    DIR_NORMAL_ANOMALY = "normal_anomaly"
    DIR_SCALED = "scaled"
    DIR_BY_DATASET = "by_dataset"
    DIR_BY_MODEL = "by_model"
    DIR_CLASSES = "classes"
    DIR_FEATURE_IMPORTANCE = "feature_importance"
    DIR_MODELS = "models"
    DIR_PLOTS = "plots"
    DIR_CSV = "csv"
    DIR_DATASETS = "datasets"
    DIR_CLASSIFICATIONS = "classifications"

    # --- Pipeline Essential Files ---
    DATASETS_FOR_MODEL_BUILDING = METADATA_DIR / f"model_paths{Naming.EXT}"
    DATASETS_FOR_CLASSIFICATIONS = METADATA_DIR / f"classification_paths{Naming.EXT}"
    DATASETS_INFO = METADATA_DIR / f"datasets_info{Naming.EXT}"
    DATASETS_FEATURES_MEAN = METADATA_DIR / f"feature_mean{Naming.EXT}"
    DATASETS_FEATURES_VAR = METADATA_DIR / f"feature_variance{Naming.EXT}"
    MODELS_INFO = RESULTS_CSV_DIR / f"models_info{Naming.EXT}"

    # --- Raw Datasets ---
    NB15_RAW = RAW_DATA_DIR / f"{Naming.NB15}{Naming.EXT}"
    SAT20_RAW = RAW_DATA_DIR / f"{Naming.SAT20}{Naming.EXT}"
    TER20_RAW = RAW_DATA_DIR / f"{Naming.TER20}{Naming.EXT}"


class RoutineConfig:
    """ Pre-defined configurations for training and evaluation routines """

    BASE_DATASETS = [
        {'type': Naming.NB15, 'path': ProjectPaths.NB15_RAW},
        {'type': Naming.SAT20, 'path': ProjectPaths.SAT20_RAW},
        {'type': Naming.TER20, 'path': ProjectPaths.TER20_RAW}
    ]

    # Defines the standard set of models to be built during a routine phase
    DATASETS_TARGETS_FOR_MODEL_BUILDING = [
        # --- NB15 dataset ---
        {'dataset_type': Naming.NB15, 'filename': f"{Naming.NB15}{Naming.PREP_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_DoS{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Exploits{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Fuzzers{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Generic{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Reconnaissance{Naming.EXT}"}
    ]

    # Defines the standard set of classifications to do during a routine phase
    DATASETS_TARGETS_FOR_CLASSIFICATIONS = [
        # --- NB15 dataset ---
        {'dataset_type': Naming.NB15, 'filename': f"{Naming.NB15}{Naming.PREP_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_DoS{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Exploits{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Fuzzers{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Generic{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Reconnaissance{Naming.EXT}"},

        # --- Hybrid dataset ---
        {'dataset_type': Naming.HYBRID, 'filename': f"{Naming.HYBRID}{Naming.PREP_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15_SAT20, 'filename': f"{Naming.NB15_SAT20}{Naming.PREP_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15_TER20, 'filename': f"{Naming.NB15_TER20}{Naming.PREP_SCALED}{Naming.EXT}"},

        # --- Specific Normal/Anomaly hybrid sub-datasets ---
        {'dataset_type': Naming.NB15_SAT20, 'filename': f"Normal_Syn_DDoS{Naming.EXT}"},
        {'dataset_type': Naming.NB15_SAT20, 'filename': f"Normal_UDP_DDoS{Naming.EXT}"},
        {'dataset_type': Naming.NB15_TER20, 'filename': f"Normal_Botnet{Naming.EXT}"},
        {'dataset_type': Naming.NB15_TER20, 'filename': f"Normal_DDoS{Naming.EXT}"},
        {'dataset_type': Naming.NB15_TER20, 'filename': f"Normal_Syn_DDoS{Naming.EXT}"},
        {'dataset_type': Naming.NB15_TER20, 'filename': f"Normal_UDP_DDoS{Naming.EXT}"}
    ]


class PlotConfig:
    """ Custom matrix layout and axis sorting for thesis visualizations """

    # Preferred vertical order (Trained models row alignment)
    HEATMAP_ROW_ORDER = [
        # --- NB15 dataset ---
        "RF (Aggregate nb15)",
        "RF (DoS nb15)",
        "RF (Exploits nb15)",
        "RF (Fuzzers nb15)",
        "RF (Generic nb15)",
        "RF (Reconnaissance nb15)",

        # --- Hybrid dataset ---
        "RF (Aggregate hybrid)",
        "RF (Aggregate nb15_sat20)",
        "RF (Aggregate nb15_ter20)",

        # --- Specific Normal/Anomaly hybrid sub-datasets ---
        "RF (Syn_DDoS nb15_sat20)",
        "RF (UDP_DDoS nb15_sat20)",
        "RF (Botnet nb15_ter20)",
        "RF (DDoS nb15_ter20)",
        "RF (Syn_DDoS nb15_ter20)",
        "RF (UDP_DDoS nb15_ter20)"
    ]
    
    # Preferred horizontal order (Test datasets column alignment)
    HEATMAP_COLUMN_ORDER = [
        # --- NB15 dataset ---
        "Aggregate nb15",
        "DoS nb15",
        "Exploits nb15",
        "Fuzzers nb15",
        "Generic nb15",
        "Reconnaissance nb15",

        # --- Hybrid dataset ---
        "Aggregate hybrid",
        "Aggregate nb15_sat20",
        "Aggregate nb15_ter20",

        # --- Specific Normal/Anomaly hybrid sub-datasets ---
        "Syn_DDoS nb15_sat20",
        "UDP_DDoS nb15_sat20",
        "Botnet nb15_ter20",
        "DDoS nb15_ter20",
        "Syn_DDoS nb15_ter20",
        "UDP_DDoS nb15_ter20"
    ]


if __name__ == "__main__":
    pass
