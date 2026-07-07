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
    SHAP_MAX_SAMPLES = 500
    X_DROP_LABELS = ["label", "class", "split_type"]
    Y_LABEL = "label"
    PCA_COMPONENTS = 2


class Naming:
    """ Standardized prefixes, suffixes, and extensions for dataset and model files """

    EXT = ".csv"
    PLOT_EXT = ".png"
    
    # Dataset Prefixes
    NB15 = "nb15"
    SAT20 = "sat20"
    TER20 = "ter20"
    HYBRID = "hybrid"
    NB15_STIN = "nb15_stin"
    NB15_SAT20 = "nb15_sat20"
    NB15_TER20 = "nb15_ter20"

    # Suffixes
    PREP = "_prep"
    PREP_SCALED = "_prep_scaled"
    CLASSIFICATION = "_classification"
    AGGR = "_aggr"
    AGGR_SCALED = "_aggr_scaled"

    # Common File Names
    MODEL_INFO = f"models_info{EXT}"
    CLASSIFICATIONS = f"classifications{EXT}"
    MODEL = "model"     # It doesn't have an extension because it can be .pkl, .joblib, etc. depending on the model type
    FEATURE_IMPORTANCE = f"feature_importance"  # It doesn't have an extension because it can be .csv or .png depending on the context
    FEATURE_IMPORTANCE_BY_PERMUTATION = f"feature_importance_by_permutation{EXT}"


class ProjectPaths:
    """ Absolute Pathlib structures for project directories and core files """

    # --- Main Folders ---
    ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
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
    
    # Specific CSV Subfolders
    CLASSIFICATIONS_CSV_DIR = RESULTS_CSV_DIR / "classifications"
    FEATURE_IMPORTANCE_CSV_DIR = RESULTS_CSV_DIR / "feature_importance"
    CLASSIFICATIONS_BY_MODEL_DIR = CLASSIFICATIONS_CSV_DIR / "by_model"
    CLASSIFICATIONS_BY_DATASET_DIR = CLASSIFICATIONS_CSV_DIR / "by_dataset"

    # Specific PLOTS Subfolders
    PERFORMANCE_PLOTS_DIR = RESULTS_PLOT_DIR / "performance"
    FEAT_IMP_PLOTS_DIR = RESULTS_PLOT_DIR / "feature_importance"
    PCA_PLOTS_INDEPENDENT_DIR = RESULTS_PLOT_DIR / "pca_independent_domain"
    PCA_PLOTS_CROSS_DOMAIN_DIR = RESULTS_PLOT_DIR / "pca_cross_domain"
    PROB_PLOTS_DIR = RESULTS_PLOT_DIR / "probabilities"
    SHAP_PLOTS_DIR = RESULTS_PLOT_DIR / "shap"

    # --- SRC Subfolders ---
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
    DIR_PERFORMANCE = "performance"

    # --- Pipeline Essential Files ---
    DATASETS_FOR_MODEL_BUILDING = METADATA_DIR / f"model_paths{Naming.EXT}"
    DATASETS_FOR_CLASSIFICATIONS = METADATA_DIR / f"classification_paths{Naming.EXT}"
    DATASETS_INFO = METADATA_DIR / f"datasets_info{Naming.EXT}"
    DATASETS_FEATURES_MEAN = METADATA_DIR / f"feature_mean{Naming.EXT}"
    DATASETS_FEATURES_VAR = METADATA_DIR / f"feature_variance{Naming.EXT}"
    MODELS_INFO = RESULTS_CSV_DIR / f"models_info{Naming.EXT}"
    MODELS_REGISTRY = MODELS / f"models_registry{Naming.EXT}"

    # --- Raw Datasets ---
    NB15_RAW = RAW_DATA_DIR / f"{Naming.NB15}{Naming.EXT}"
    SAT20_RAW = RAW_DATA_DIR / f"{Naming.SAT20}{Naming.EXT}"
    TER20_RAW = RAW_DATA_DIR / f"{Naming.TER20}{Naming.EXT}"


class RoutineConfig:
    """ Pre-defined configurations for training and evaluation routines """

    BASE_DATASETS = [
        {'dataset_type': Naming.NB15, 'path': ProjectPaths.NB15_RAW},
        {'dataset_type': Naming.SAT20, 'path': ProjectPaths.SAT20_RAW},
        {'dataset_type': Naming.TER20, 'path': ProjectPaths.TER20_RAW}
    ]

    # Defines the standard set of models to be built during a routine phase
    DATASETS_TARGETS_FOR_MODEL_BUILDING = [
        # --- NB15 dataset ---
        {'dataset_type': Naming.NB15, 'filename': f"{Naming.NB15}{Naming.AGGR_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_DoS{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Exploits{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Fuzzers{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Generic{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Reconnaissance{Naming.EXT}"},
        {'dataset_type': Naming.HYBRID, 'filename': f"{Naming.HYBRID}{Naming.AGGR_SCALED}{Naming.EXT}"}
    ]

    # Defines the standard set of classifications to do during a routine phase
    DATASETS_TARGETS_FOR_CLASSIFICATIONS = [
        # --- NB15 dataset ---
        {'dataset_type': Naming.NB15, 'filename': f"{Naming.NB15}{Naming.AGGR_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_DoS{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Exploits{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Fuzzers{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Generic{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Reconnaissance{Naming.EXT}"},

        # --- Hybrid dataset ---
        {'dataset_type': Naming.HYBRID, 'filename': f"{Naming.HYBRID}{Naming.AGGR_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15_STIN, 'filename': f"{Naming.NB15_STIN}{Naming.AGGR_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15_SAT20, 'filename': f"{Naming.NB15_SAT20}{Naming.AGGR_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15_TER20, 'filename': f"{Naming.NB15_TER20}{Naming.AGGR_SCALED}{Naming.EXT}"},

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
        "RF (Aggregate hybrid)"
        # "RF (Aggregate nb15_stin)",
        # "RF (Aggregate nb15_sat20)",
        # "RF (Aggregate nb15_ter20)",

        # --- Specific Normal/Anomaly hybrid sub-datasets ---
        # "RF (Syn_DDoS nb15_sat20)",
        # "RF (UDP_DDoS nb15_sat20)",
        # "RF (Botnet nb15_ter20)",
        # "RF (DDoS nb15_ter20)",
        # "RF (Syn_DDoS nb15_ter20)",
        # "RF (UDP_DDoS nb15_ter20)"
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
        "Aggregate nb15_stin",
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


class PlotFlags:
    """Master Switch flags for granular control of plot generation during pipeline execution"""
    
    # Heavy-weight plots (computationally expensive)
    ENABLE_SHAP_PLOTS = True              # Very heavy - SHAP explainability plots
    
    # Light-weight plots (standard visualizations)
    ENABLE_HEATMAP_PLOTS = True            # Performance matrix heatmaps
    ENABLE_FEATURE_IMPORTANCE = True       # Feature importance bar charts
    ENABLE_PROBABILITY_PLOTS = True        # Probability distribution histograms
    ENABLE_PCA_INDEPENDENT_PLOTS = True    # PCA 2D scatter plots for independent domain
    ENABLE_PCA_CROSS_DOMAIN_PLOTS = True   # PCA 2D scatter plots for cross domain
    ENABLE_PERFORMANCE_PLOTS = True        # Performance matrix heatmaps


if __name__ == "__main__":
    pass
