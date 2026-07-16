"""
Centralized management of project paths, ML constants, and routine configurations.
"""
import pathlib


class MLConstants:
    """
    Hyperparameters, directory configurations, and standard evaluation metrics 
    for the Machine Learning classification pipeline.
    """

    # --- Experimental Setup & Reproducibility ---
    SEEDS = [0, 1, 7, 42, 101, 123, 999, 1337, 2026, 12345]     # Seeds used for random states across iterations
    MAIN_SEED = 127001                                          # Main seed for primary model initializations
    MODEL_TYPE = ['rf', 'dt', 'hgb']                            # Supported model families (Random Forest, Decision Tree, HistGradientBoosting)

    # --- Dataset & Splitting Parameters ---
    NORMAL_ANOMALY_RATIO = 10                                   # Ratio of normal traffic to anomaly instances
    TRAIN_SPLIT = 0.8                                           # Percentage of data used for training (80%)
    X_DROP_LABELS = ['label', 'class', 'split_type']            # Non-feature columns to drop before training/evaluation
    Y_LABEL = 'label'                                           # Ground-truth target column name

    # --- Model Training & Formatting ---
    MODEL_VERBOSE = 0                                           # Verbosity level of classifier fitting (0 = silent)
    DECIMAL_DIGITS = 4                                          # Precision decimal places for metrics and rounding

    # --- Visualization & Explainability ---
    PLOTTING_METRICS = ['TNR', 'TPR']                           # Metrics to display in evaluation plots
    SHAP_MAX_SAMPLES = 500                                      # Maximum sample size used for SHAP explainer calculations
    PCA_COMPONENTS = 2                                          # Number of target dimensions for Principal Component Analysis
    KDE_TOP_FEATURES = ['pkts_per_sec', 'total_bytes', 'dst_win_byt']  # Top features selected for KDE distribution analysis

    # --- Injection Parameters ---
    INJECTION_RATIO = 299       # Must be 1 digit less than the denominator that we want (e.g. 1/3 than INJ_RATIO = 2)

    # --- Welch t-test Parameters ---
    WELCH_TTEST_FEATURE = 'F1-Score'
    WELCH_TTEST_ALFA_VALUE = 0.05


class Naming:
    """
    Standardized prefixes, suffixes, and file extensions used for dataset 
    management, model serialization, and visualization output naming.
    """

    # --- File Extensions ---
    EXT = ".csv"                                # Standard extension for tabular data files
    PLOT_EXT = ".png"                           # Standard extension for exported visualization plots

    # --- Model Family Prefixes ---
    RANDOM_FOREST = "rf"                        # Prefix for Random Forest models
    DECISION_TREE = "dt"                        # Prefix for Decision Tree models
    HIST_GRADIENT_BOOSTING = "hgb"              # Prefix for HistGradientBoosting models
    
    # --- Dataset Type Prefixes ---
    NB15 = "nb15"                               # UNSW-NB15 baseline dataset prefix
    SAT20 = "sat20"                             # Satellite dataset prefix
    TER20 = "ter20"                             # Terrestrial dataset prefix
    HYBRID = "hybrid"                           # Combined/Hybrid dataset prefix
    INJECTION = "injection"                     # Injection-based dataset prefix
    NB15_STIN = "nb15_stin"                     # UNSW-NB15 with spatial-temporal injection prefix
    NB15_SAT20 = "nb15_sat20"                   # Mixed UNSW-NB15 and Satellite dataset prefix
    NB15_TER20 = "nb15_ter20"                   # Mixed UNSW-NB15 and Terrestrial dataset prefix

    # --- Suffixes ---
    AGGR = "_aggr"                              # Suffix for aggregated raw results
    AGGR_SCALED = "_aggr_scaled"                # Suffix for aggregated and normalized/scaled results

    # --- Standardized Filenames ---
    CLASSIFICATIONS = f"classifications{EXT}"   # Target file for model prediction logs and outputs
    MODELS_REGISTRY = f"models_registry{EXT}"   # Target file cataloging trained models and configurations
    MODELS_METADATA = f"models_metadata{EXT}"   # Target file containing training times, hyperparameters, and environment specs
    WELCH_TTEST_FEATURE_MEANS = f"{MLConstants.WELCH_TTEST_FEATURE}_means{EXT}"
    WELCH_TTEST = f"welch_ttest{EXT}"


class ProjectPaths:
    """
    Absolute Pathlib structures and constants defining project directories, 
    dynamic subfolder names, and core files for the ML pipeline.
    """

    # --- Core Project Directories ---
    ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
    DATA = ROOT / "data"                                    # Directory for raw, preprocessed, and metadata datasets
    SRC = ROOT / "src"                                      # Directory containing source code and scripts
    RUNS = ROOT / "runs"                                    # Directory storing experiment outputs, serialized models, and metrics

    # --- Data & Preprocessing Subfolders ---
    RAW_DATA_DIR = DATA / "raw"                             # Directory storing original, unmodified datasets
    PREP_DATA_DIR = DATA / "preprocessed"                   # Directory storing engineered and cleaned datasets
    METADATA_DIR = PREP_DATA_DIR / "metadata"               # Directory containing statistical summaries and pipeline file paths
    METADATA_PLOT_DIR = METADATA_DIR / "plots"              # Directory storing dataset analysis visualizations

    # --- Exploratory Data Analysis (EDA) Plot Directories ---
    PCA_PLOTS_DIR = METADATA_PLOT_DIR / "pca"               # Output directory for 2D/3D PCA projection scatter plots
    KDE_PLOTS_DIR = METADATA_PLOT_DIR / "kde"               # Output directory for feature Kernel Density Estimate distributions

    # --- Folder Name Constants (For dynamic run-time path building) --- 
    DIR_SINGLE_CLASSES = "single_classes"                   # Subfolder for individual traffic class processing
    DIR_NORMAL_ANOMALY = "normal_anomaly"                   # Subfolder for binary (normal vs anomaly) classification
    DIR_BY_DATASET = "by_dataset"                           # Subfolder grouping outputs by input dataset name
    DIR_BY_MODEL = "by_model"                               # Subfolder grouping outputs by specific model family
    DIR_FEATURE_IMPORTANCE = "feature_importance"           # Subfolder containing raw feature importance values
    DIR_MODELS = "models"                                   # Subfolder containing serialized model files (e.g., joblib/pickle)
    DIR_PLOTS = "plots"                                     # General subfolder name for plots within run directories
    DIR_RESULTS = "results"                                 # General subfolder name for tabular evaluation results
    DIR_WELCH_TTEST = "welch_ttest"                         # General subfolder name for welch t-test

    # --- Run-Specific Visualization Subfolders ---
    DIR_PERFORMANCE_PLOTS = "performance"                   # Subfolder storing performance charts
    DIR_FEAT_IMP_PLOTS = "feature_importance"               # Subfolder storing generated feature importance bar charts
    DIR_PROB_PLOTS = "probabilities"                        # Subfolder storing predicted probability distribution plots
    DIR_PR_CURVE_PLOTS = "pr_curves"                        # Subfolder storing Precision-Recall curve plots
    DIR_THRESHOLD_PLOTS = "threshold_metrics"               # Subfolder storing metric performance charts across decision thresholds
    DIR_SHAP_PLOTS = "shap"                                 # Subfolder storing SHAP-based local and global explainability plots

    # --- Pipeline Essential Metadata & Configuration Files ---
    DATASETS_FOR_MODEL_BUILDING = METADATA_DIR / f"data_model_building_paths{Naming.EXT}"   # Paths config file for model training datasets
    DATASETS_FOR_CLASSIFICATIONS = METADATA_DIR / f"data_classification_paths{Naming.EXT}"  # Paths config file for evaluation datasets
    DATASETS_INFO = METADATA_DIR / f"datasets_info{Naming.EXT}"                             # General datasets information and metrics log
    DATASETS_FEATURES_MEAN = METADATA_DIR / f"feature_mean{Naming.EXT}"                     # CSV tracking feature means for distribution analysis
    DATASETS_FEATURES_VAR = METADATA_DIR / f"feature_variance{Naming.EXT}"                  # CSV tracking feature variances for distribution analysis

    # --- Raw Input Datasets ---
    NB15_RAW = RAW_DATA_DIR / f"{Naming.NB15}{Naming.EXT}"      # Path to raw UNSW-NB15 baseline dataset
    SAT20_RAW = RAW_DATA_DIR / f"{Naming.SAT20}{Naming.EXT}"    # Path to raw Satellite dataset
    TER20_RAW = RAW_DATA_DIR / f"{Naming.TER20}{Naming.EXT}"    # Path to raw Terrestrial dataset


class RoutineConfig:
    """ Pre-defined configurations for pipeline training and evaluation routines """

    # --- Base Raw Datasets ---
    BASE_DATASETS = [
        {'dataset_type': Naming.NB15, 'path': ProjectPaths.NB15_RAW},
        {'dataset_type': Naming.SAT20, 'path': ProjectPaths.SAT20_RAW},
        {'dataset_type': Naming.TER20, 'path': ProjectPaths.TER20_RAW}
    ]

    # --- Target Datasets for Training / Model Fitting ---
    # Defines the standard set of models to be built during a routine phase
    DATASETS_TARGETS_FOR_MODEL_BUILDING = [
        # --- NB15 dataset ---
        {'dataset_type': Naming.NB15, 'filename': f"{Naming.NB15}{Naming.AGGR_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_DoS{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Exploits{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Fuzzers{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Generic{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Reconnaissance{Naming.EXT}"},

        # --- Hybrid dataset ---
        {'dataset_type': Naming.INJECTION, 'filename': f"{Naming.INJECTION}{Naming.AGGR_SCALED}{Naming.EXT}"}
    ]

    # --- Target Datasets for Prediction / Classification ---
    # Defines the standard set of classifications to perform during a routine phase
    DATASETS_TARGETS_FOR_CLASSIFICATIONS = [
        # --- NB15 dataset ---
        {'dataset_type': Naming.NB15, 'filename': f"{Naming.NB15}{Naming.AGGR_SCALED}{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_DoS{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Exploits{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Fuzzers{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Generic{Naming.EXT}"},
        {'dataset_type': Naming.NB15, 'filename': f"Normal_Reconnaissance{Naming.EXT}"},

        # --- Hybrid dataset ---
        {'dataset_type': Naming.INJECTION, 'filename': f"{Naming.INJECTION}{Naming.AGGR_SCALED}{Naming.EXT}"},
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
    """ Custom matrix layout and axis sorting configurations for thesis visualizations """

    # Preferred vertical order (Trained models row alignment)
    HEATMAP_ROW_ORDER = [
        # --- NB15 dataset ---
        "(Aggregate nb15)",
        "(DoS nb15)",
        "(Exploits nb15)",
        "(Fuzzers nb15)",
        "(Generic nb15)",
        "(Reconnaissance nb15)",

        # --- Hybrid dataset ---
        "(Aggregate injection)",
        # "(Aggregate nb15_stin)",
        # "(Aggregate nb15_sat20)",
        # "(Aggregate nb15_ter20)",

        # --- Specific Normal/Anomaly hybrid sub-datasets ---
        # "(Syn_DDoS nb15_sat20)",
        # "(UDP_DDoS nb15_sat20)",
        # "(Botnet nb15_ter20)",
        # "(DDoS nb15_ter20)",
        # "(Syn_DDoS nb15_ter20)",
        # "(UDP_DDoS nb15_ter20)"
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
        "Aggregate injection",
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
    ENABLE_SHAP_PLOTS = True            # Very heavy - SHAP explainability plots
    
    # Light-weight plots (standard visualizations)
    ENABLE_FEATURE_IMPORTANCE = True    # Feature importance bar charts
    ENABLE_PROBABILITY_PLOTS = True     # Probability distribution histograms
    ENABLE_PCA_PLOTS = True             # PCA 2D scatter plots
    ENABLE_PERFORMANCE_PLOTS = True     # Performance matrix heatmaps
    ENABLE_PR_CURVE_PLOTS = True        # Precision-Recall (PR) Curve (Cross-Domain)
    ENABLE_THRESHOLD_PLOTS = True       # F1-Score vs Threshold
    ENABLE_KDE_PLOTS = True             # KDE Plot about important features


if __name__ == "__main__":
    pass
