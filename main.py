"""
Main entry point for the Satellite IDS project.
"""

from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.classification import classification_processing
from src.models import model_processing
from src.plotting import plot_heatmap_for_metrics, save_pca_plot
from src.utils.file_utils import (
    create_directory, get_data_from_csv, group_by_classes_and_save, 
    group_by_model_and_save, group_datasets_paths_for_filename_list,
    init_project_environment
)
from src.utils.config import MLConstants, Naming, ProjectPaths, RoutineConfig, PlotFlags
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

    # --- Create directories for classification results ---
    # Create the main classifications directory
    classifications_dir = create_directory(ProjectPaths.DIR_CLASSIFICATIONS, ProjectPaths.RESULTS_CSV_DIR)

    # Create subdirectories for grouping classifications by model and by dataset type
    group_by_model_dir = create_directory(ProjectPaths.DIR_BY_MODEL, classifications_dir)
    group_by_classes_dir = create_directory(ProjectPaths.DIR_BY_DATASET, classifications_dir)
   
    # Define the paths for the classifications CSV file and the models info CSV file
    classifications_file = classifications_dir / Naming.CLASSIFICATIONS
    models_info_file = ProjectPaths.RESULTS_CSV_DIR / Naming.MODEL_INFO

    # --- Perform classifications for each dataset and model ---
    # Get datasets for classification and iterate through them
    datasets = get_data_from_csv(ProjectPaths.DATASETS_FOR_CLASSIFICATIONS)

    # --- Setup PCA Cross Domain ---
    # Save PCA CROSS DOMAIN plot if enabled
    if PlotFlags.ENABLE_PCA_CROSS_DOMAIN_PLOTS:
        baseline_record = datasets[datasets['path'].apply(lambda p: Path(p).stem) == f"{Naming.NB15}{Naming.AGGR_SCALED}"].to_dict('records')[0]
        dataset_path = Path(baseline_record['path'])
        dataset_type = baseline_record['dataset_type']
        
        data = get_data_from_csv(dataset_path)
        train_data = data[data['split_type'] == 'train']

        X_train_base = train_data.drop(columns=MLConstants.X_DROP_LABELS)
        y_train_base = train_data[MLConstants.Y_LABEL]
        
        scaler = StandardScaler()
        pca = PCA(n_components=MLConstants.PCA_COMPONENTS, random_state=MLConstants.RANDOM_STATE)

        scaler.fit(X_train_base)
        X_train_scaled_for_fit = scaler.transform(X_train_base)
        pca.fit(X_train_scaled_for_fit)
        
        save_pca_plot(
            X=X_train_base,
            y=y_train_base,
            dataset_type=dataset_type,
            dataset_name=f"{dataset_path.stem}_train_baseline",
            precomputed_scaler=scaler, 
            precomputed_pca=pca
        )
        
    # Iterate through each dataset and classify using the saved models
    for d in datasets.to_dict('records'):
        dataset_type = d['dataset_type']
        dataset_path = Path(d['path'])

        # Get the data from the dataset CSV
        data = get_data_from_csv(dataset_path)

        # Get the paths of the saved models from the models registry
        models_paths = get_data_from_csv(ProjectPaths.MODELS_REGISTRY)['path']

        # Iterate through each model and perform classification
        for model_path in models_paths:
            classification_processing(
                model_path=Path(model_path), 
                data=data, 
                dataset_type=dataset_type, 
                dataset_name=dataset_path.stem)

        # --- Save Plots Based on Flags ---
        # Save PCA INDEPENDENT plot if enabled
        test_data = data[data['split_type'] == 'test']
        X_test = test_data.drop(columns=MLConstants.X_DROP_LABELS)
        y_test = test_data[MLConstants.Y_LABEL]
        
        if PlotFlags.ENABLE_PCA_INDEPENDENT_PLOTS:
            save_pca_plot(
                X=X_test, 
                y=y_test, 
                dataset_type=dataset_type, 
                dataset_name=dataset_path.stem
            )

        # Save PCA CROSS DOMAIN plot if enabled
        if PlotFlags.ENABLE_PCA_CROSS_DOMAIN_PLOTS:
            save_pca_plot(
                X=X_test,
                y=y_test,
                dataset_type=dataset_type,
                dataset_name=f"{dataset_path.stem}_test",
                precomputed_scaler=scaler, 
                precomputed_pca=pca
            )

    # --- Group and Save Classification Results ---
    # Group classifications by model and save them in the corresponding directory
    group_by_model_and_save(classifications_file, group_by_model_dir)

    # Group classifications by dataset type and save them in the corresponding directory
    group_by_classes_and_save(classifications_file, group_by_classes_dir)

    # --- Generate Performance Plots (F1-Score, Precision, Recall) ---
    # If the flag is enabled, generate performance plots based on the classification results
    if PlotFlags.ENABLE_PERFORMANCE_PLOTS:
        # Get the data for models and classifications from the respective CSV files
        models = get_data_from_csv(models_info_file)
        data = get_data_from_csv(classifications_file)

        # Generate performance plots using the plotting_processing function
        plot_heatmap_for_metrics(models, data)

    print("\n--- Routine Classification Completed ---")


def _model_building():
    """ Executes a predefined model building routine """
    print("\n--- Starting Model Building Phase ---")

    # Get dataset for random forest models
    datasets = get_data_from_csv(ProjectPaths.DATASETS_FOR_RANDOM_FOREST)

    # Iterate through each dataset and build random forest models
    for d in datasets.to_dict('records'):
        dataset_type = d['dataset_type']
        dataset_path = d['path']

        data = get_data_from_csv(Path(dataset_path))
        model_processing(data, dataset_type, Naming.RANDOM_FOREST)

    # Get dataset for isolation forest models
    datasets = get_data_from_csv(ProjectPaths.DATASETS_FOR_ISOLATION_FOREST)

    # Iterate through each dataset and build isolation forest models
    for d in datasets.to_dict('records'):
        dataset_type = d['dataset_type']
        dataset_path = d['path']

        data = get_data_from_csv(Path(dataset_path))
        model_processing(data, dataset_type, Naming.ISOLATION_FOREST)

    print("\n--- Routine Model Building Completed ---")


def _preprocessing():
    """ Executes a predefined preprocessing routine for nb15, sat20 ter20 and hybrid datasets """
    print("\n--- Starting Preprocessing Phase ---")

    # Initialize variables for hybrid dataset
    nb15_normal_data = None
    nb15_anomaly_data = None
    sat20_anomaly_data = None
    ter20_anomaly_data = None
    
    # Do preprocessing for each dataset
    for d in RoutineConfig.BASE_DATASETS:
        # Get dataset params
        dataset_type = d['dataset_type']
        dataset_path = d['path']
        
        # Do data preprocessing
        data = get_data_from_csv(dataset_path)
        data_prep = data_preprocessing(data, dataset_type)

        # Do file preprocessing for a single dataset
        single_dataset_file_preprocessing(data_prep, dataset_type)

        # Set variables for hybrid dataset
        if dataset_type == Naming.NB15:
            nb15_normal_data = data_prep[data_prep['label'] == 0]
            nb15_anomaly_data = data_prep[data_prep['label'] == 1]
        elif dataset_type == Naming.SAT20:
            sat20_anomaly_data = data_prep[data_prep['label'] == 1]
        elif dataset_type == Naming.TER20:
            ter20_anomaly_data = data_prep[data_prep['label'] == 1]
    
    # Do file preprocessing for a hybrid dataset
    hybrid_dataset_file_preprocessing(nb15_normal_data, nb15_anomaly_data, sat20_anomaly_data, ter20_anomaly_data)

    # Group datasets paths for model building; save them in a csv file
    jobs = [
        (ProjectPaths.DATASETS_FOR_RANDOM_FOREST, RoutineConfig.DATASETS_TARGETS_FOR_RANDOM_FOREST), 
        (ProjectPaths.DATASETS_FOR_ISOLATION_FOREST, RoutineConfig.DATASETS_TARGETS_FOR_ISOLATION_FOREST)
    ]
    
    for dst_path, filename_list in jobs:
        group_datasets_paths_for_filename_list(
            src_path=ProjectPaths.DATASETS_INFO, 
            dst_path=dst_path, 
            filename_list=filename_list
        )

    # Group datasets paths for classifications; save them in a csv file
    group_datasets_paths_for_filename_list(
        src_path=ProjectPaths.DATASETS_INFO, 
        dst_path=ProjectPaths.DATASETS_FOR_CLASSIFICATIONS, 
        filename_list=RoutineConfig.DATASETS_TARGETS_FOR_CLASSIFICATIONS
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
