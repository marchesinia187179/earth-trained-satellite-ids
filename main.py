"""
Main entry point for the Satellite IDS project.
"""

from pathlib import Path
from src.classification import classification_processing
from src.models import model_processing
from src.plotting import plot_pca, plotting_processing
from src.utils.file_utils import (
    create_directory, get_data_from_csv, group_by_classes_and_save, 
    group_by_model_and_save, group_datasets_paths_for_filename_list,
    init_project_environment
)
from src.utils.config import MLConstants, Naming, ProjectPaths, RoutineConfig, PlotConfig, PlotFlags
from src.data_preprocessing import data_preprocessing
from src.file_preprocessing import hybrid_dataset_file_preprocessing, single_dataset_file_preprocessing


# --- Internal Helper Functions ---
def _run_all_phases():
    """ Executes all phases of the pipeline in a single run """
    print(f"\n--- Running Full Pipeline ---")

    _preprocessing()
    _model_building()
    _classifications()
        
    print(f"\n--- Full Pipeline Completed ---")




def _classifications():
    """ Evaluates saved models on specific testing datasets """
    print("\n--- Starting Classification Phase ---")
   
    # Do classification process for each classification task
    datasets = get_data_from_csv(ProjectPaths.DATASETS_FOR_CLASSIFICATIONS)
    for d in datasets.to_dict('records'):
        dataset_type = d['dataset_type']
        dataset_path = Path(d['path'])

        data = get_data_from_csv(dataset_path)

        # Generate one PCA plot for each loaded test dataset and avoid filename collisions
        if PlotFlags.ENABLE_PCA_PLOTS:
            dataset_stem = dataset_path.stem.lower().replace(' ', '_')
            pca_output_path = ProjectPaths.PCA_PLOTS_DIR / f"{dataset_type.lower()}_{dataset_stem}{Naming.PLOT_EXT}"
            X = data.drop(columns=["label", "class", "split_type"])
            y = data["label"]
            plot_pca(X, y, pca_output_path)
        else:
            print(f"⏭️  Skipping PCA plot for {dataset_type} (ENABLE_PCA_PLOTS=False)")

        models_paths = get_data_from_csv(ProjectPaths.MODELS / Naming.MODELS_REGISTRY)['path']
        for model_path in models_paths:
            classification_processing(Path(model_path), data, dataset_type, dataset_stem if PlotFlags.ENABLE_PCA_PLOTS else dataset_path.stem.lower().replace(' ', '_'))

    # Group classifications by model and by dataset type and save them in separate directories
    parent_path = ProjectPaths.RESULTS_CSV_DIR / ProjectPaths.DIR_CLASSIFICATIONS
    group_by_model_dir = create_directory(ProjectPaths.DIR_BY_MODEL, parent_path)
    group_by_classes_dir = create_directory(ProjectPaths.DIR_BY_DATASET, parent_path)
    src_file = ProjectPaths.RESULTS_CSV_DIR / ProjectPaths.DIR_CLASSIFICATIONS / Naming.CLASSIFICATIONS
    group_by_model_and_save(src_file, group_by_model_dir)
    group_by_classes_and_save(src_file, group_by_classes_dir)

    # --- Generate Performance Plots (F1-Score, Precision, Recall) ---
    # This phase consolidates classification results and generates global performance heatmaps
    if PlotFlags.ENABLE_PERFORMANCE_PLOTS:
        print("\n--- Generating Global Performance Plots ---")
        models = get_data_from_csv(ProjectPaths.RESULTS_CSV_DIR / Naming.MODEL_INFO)
        data = get_data_from_csv(ProjectPaths.RESULTS_CSV_DIR / ProjectPaths.DIR_CLASSIFICATIONS / Naming.CLASSIFICATIONS)
        plotting_processing(models, data, MLConstants.PLOTTING_METRICS, PlotConfig.HEATMAP_ROW_ORDER, PlotConfig.HEATMAP_COLUMN_ORDER)
        print("Global performance plots completed.")
    else:
        print("⏭️  Skipping global performance plots (ENABLE_PERFORMANCE_PLOTS=False)")

    print("\n--- Routine Classification Completed ---")


def _model_building():
    """ Executes a predefined model building routine """
    print("\n--- Starting Model Building Phase ---")

    # Start Model Processing for each model building dataset
    datasets = get_data_from_csv(ProjectPaths.DATASETS_FOR_MODEL_BUILDING)
    for d in datasets.to_dict('records'):
        dataset_type = d['dataset_type']
        dataset_path = d['path']

        data = get_data_from_csv(Path(dataset_path))
        model_processing(data, dataset_type)

    print("\n--- Routine Model Building Completed ---")


def _preprocessing():
    """ Executes a predefined preprocessing routine for nb15, sat20 ter20 and hybrid datasets """
    print("\n--- Starting Preprocessing Phase ---")

    # Initialize variables for hybrid dataset
    nb15_normal_data = None
    sat20_anomaly_data = None
    ter20_anomaly_data = None
    
    # Do preprocessing for each dataset
    for d in RoutineConfig.BASE_DATASETS:
        # Get dataset params
        dataset_type = d['type']
        dataset_path = d['path']

        # Security check for existing path
        if not dataset_path.exists():
            print(f"Warning: Dataset file not found at {dataset_path}. Skipping {dataset_type}.")
            continue
            
        print(f"\n[ROUTINE] Processing {dataset_type}...")
        
        # Do data preprocessing
        data = get_data_from_csv(dataset_path)
        data_prep = data_preprocessing(data, dataset_type)

        # Do file preprocessing for a single dataset
        single_dataset_file_preprocessing(data_prep, dataset_type)

        # Set variables for hybrid dataset
        if dataset_type == Naming.NB15:
            nb15_normal_data = data_prep[data_prep['label'] == 0]
        elif dataset_type == Naming.SAT20:
            sat20_anomaly_data = data_prep[data_prep['label'] == 1]
        elif dataset_type == Naming.TER20:
            ter20_anomaly_data = data_prep[data_prep['label'] == 1]
    
    # Do file preprocessing for a hybrid dataset
    hybrid_dataset_file_preprocessing(nb15_normal_data, sat20_anomaly_data, ter20_anomaly_data)

    # Group datasets paths for model building; save them in a csv file
    group_datasets_paths_for_filename_list(
        ProjectPaths.DATASETS_INFO, ProjectPaths.DATASETS_FOR_MODEL_BUILDING, RoutineConfig.DATASETS_TARGETS_FOR_MODEL_BUILDING
    )
    
    # Group datasets paths for classifications; save them in a csv file
    group_datasets_paths_for_filename_list(
        ProjectPaths.DATASETS_INFO, ProjectPaths.DATASETS_FOR_CLASSIFICATIONS, RoutineConfig.DATASETS_TARGETS_FOR_CLASSIFICATIONS
    )

    print("\n--- Routine Preprocessing Phase Completed ---")


# --- Public Functions ---
def main():
    """ Main entry point of the application with a main dashboard menu """
    init_project_environment()

    while True:
        # Print main dashboard menu
        print("\n" + "="*60)
        print("      SATELLITE IDS - MAIN DASHBOARD")
        print("="*60)
        print("1. Run PREPROCESSING")
        print("2. Run MODEL BUILDING")
        print("3. Run CLASSIFICATIONS + PERFORMANCE PLOTS")
        print("4. Run ALL!")
        print("5. Exit application")
        print("="*60)
        
        # Ask user's response
        main_choice = input("Select execution mode (1, 2, 3, 4 or 5): ")
        
        # Do user's choice
        if main_choice == '1':  # Preprocessing case
            _preprocessing()
        elif main_choice == '2':    # Model building case
            _model_building()
        elif main_choice == '3':    # Classifications case (now includes performance plots)
            _classifications()
        elif main_choice == '4':    # All case
            _run_all_phases()
        elif main_choice == '5':    # Exit case
            print("\nExiting application. Goodbye!")
            break
        else:
            print("\nInvalid option! Please choose 1, 2, 3, 4 or 5.")


if __name__ == "__main__":
    main()
