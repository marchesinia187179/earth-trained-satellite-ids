"""
Main entry point for the Satellite IDS project.
"""
import pandas as pd

from pathlib import Path
from itertools import product
from src.classification import classification_processing
from src.models import model_processing
from src.plotting import (
    save_all_kde_plots, save_all_pca_plots, 
    save_heatmap_for_metrics_plot, save_aggregated_metrics_heatmap
)
from src.utils.file_utils import (
    aggregate_welch_ttest_feature_scores_by_seed, create_directory, get_data_from_csv, 
    group_by_classes_and_save, group_by_model_and_save, group_datasets_paths_for_filename_list,
    init_project_environment, validate_path, create_csv_from_data
)
from src.utils.config import MLConstants, Naming, ProjectPaths, RoutineConfig, PlotFlags
from src.utils.metrics import calculate_welch_ttest_from_summary
from src.data_preprocessing import data_preprocessing
from src.file_preprocessing import hybrid_dataset_file_preprocessing, single_dataset_file_preprocessing


# --- Internal Helper Functions ---
def _run_all_phases():
    """ Executes all phases of the pipeline in a single run """
    print(f"\n--- Running Full Pipeline ---")

    # --- Preprocessing Phase ---
    _preprocessing()

    # --- Model Building Phase ---
    for model_type, seed in product(MLConstants.MODEL_TYPE, MLConstants.SEEDS):
        _model_building(model_type, seed)

    # --- Classifications Phase ---
    for model_type, seed in product(MLConstants.MODEL_TYPE, MLConstants.SEEDS):
        _classifications(model_type, seed)

    # --- Welch t-test Phase ---
    for model_type in MLConstants.MODEL_TYPE:
        _welch_ttest(model_type)
        
    print(f"\n--- Full Pipeline Completed ---")


def _welch_ttest(model_type):
    """ Executes the Welch's t-test statistical analysis across all seeds for a specific model type """
    print(f"\n--- Starting Welch's T-Test for {model_type} Models Phase ---")

    # --- Create directory for Welch's t-test results ---
    welch_ttest_dir = create_directory(
        dir_name=ProjectPaths.DIR_WELCH_TTEST, 
        parent_path=ProjectPaths.RUNS / model_type
    )

    # --- Collect classification CSV paths ---
    # Retrieve the classification results for each seed
    csv_paths = []
    for seed in MLConstants.SEEDS:
        classifications_file = validate_path(
            path=ProjectPaths.RUNS / model_type / f"{seed}" / ProjectPaths.DIR_RESULTS / Naming.CLASSIFICATIONS, 
            is_directory=False
        )
        csv_paths.append(classifications_file)

    # --- Aggregate metrics and calculate statistical significance ---
    # Aggregate the classification scores by seed
    data = aggregate_welch_ttest_feature_scores_by_seed(csv_paths=csv_paths)

    # Save the aggregated feature means
    create_csv_from_data(
        data=data, 
        file_name=Naming.WELCH_TTEST_FEATURE_MEANS, 
        file_path=welch_ttest_dir
    )
    
    # Compute Welch's t-test against the reference model
    results_list = calculate_welch_ttest_from_summary(data=data)

    # Save the final statistical results
    create_csv_from_data(
        data=results_list, 
        file_name=Naming.WELCH_TTEST, 
        file_path=welch_ttest_dir
    )

    print(f"\n--- Routine Welch's T-Test for {model_type} Models Completed ---")


def _aggregated_heatmaps(model_type):
    """ Executes the generation of aggregated mean/variance heatmaps across all seeds """
    print(f"\n--- Starting Aggregated Heatmaps Phase for {model_type} Models ---")

    # --- Create directory for the aggregated heatmaps ---
    agg_heatmap_dir = create_directory(
        dir_name=ProjectPaths.DIR_AGGR_HEATMAP, 
        parent_path=ProjectPaths.RUNS / model_type
    )

    # --- Collect classification and metadata CSVs across all seeds ---
    all_data = []
    all_models_metadata = []
    
    for seed in MLConstants.SEEDS:
        # Get classifications file and validate it as a file
        classifications_file = validate_path(
            path=ProjectPaths.RUNS / model_type / f"{seed}" / ProjectPaths.DIR_RESULTS / Naming.CLASSIFICATIONS, 
            is_directory=False
        )
        
        # Read classifications and append the active seed to the dataframe
        data = get_data_from_csv(classifications_file)
        data['seed'] = seed
        all_data.append(data)

        # Get models metadata file and validate it as a file
        models_metadata_file = validate_path(
            path=ProjectPaths.RUNS / model_type / f"{seed}" / ProjectPaths.DIR_MODELS / Naming.MODELS_METADATA, 
            is_directory=False
        )
        
        # Read models metadata and append to the list
        metadata = get_data_from_csv(models_metadata_file)
        all_models_metadata.append(metadata)

    # --- Combine data and remove redundant metadata entries ---
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_metadata = pd.concat(all_models_metadata, ignore_index=True).drop_duplicates(subset=['model_name'])

    # --- Generate Aggregated Performance Plots ---
    save_aggregated_metrics_heatmap(
        model_type=model_type,
        models=combined_metadata,
        data=combined_data,
        dst_dir=agg_heatmap_dir
    )

    print(f"\n--- Aggregated Heatmaps Phase for {model_type} Models Completed ---")


def _classifications(model_type, seed):
    """ Evaluates saved models on specific testing datasets """
    print(f"\n--- Starting Classification for {model_type} Model Building (random_state={seed}) Phase ---")

    # --- Get helper dir and file paths ---
    # Get main directory and validate it as a directory
    model_seed_dir = validate_path(
        path=ProjectPaths.RUNS / model_type / f"{seed}", 
        is_directory=True
    )
    
    # Get models registry file and validate it as a file
    models_registry_file = validate_path(
        path=model_seed_dir / ProjectPaths.DIR_MODELS / Naming.MODELS_REGISTRY, 
        is_directory=False
    )

    # Get models metadata file and validate it as a file
    models_metadata_file = validate_path(
        path=model_seed_dir / ProjectPaths.DIR_MODELS / Naming.MODELS_METADATA, 
        is_directory=False
    )

    # --- Create directories for classification results ---
    # Create the main results directory
    results = create_directory(dir_name=ProjectPaths.DIR_RESULTS, parent_path=model_seed_dir)

    # Create subdirectories for grouping classifications by model and by dataset type
    group_by_model_dir = create_directory(dir_name=ProjectPaths.DIR_BY_MODEL, parent_path=results)
    group_by_classes_dir = create_directory(dir_name=ProjectPaths.DIR_BY_DATASET, parent_path=results)

    # Create the main plots directory
    plots_dir = create_directory(dir_name=ProjectPaths.DIR_PLOTS, parent_path=results)

    # Define the paths for the classifications CSV file and the models info CSV file
    classifications_file = results / Naming.CLASSIFICATIONS

    # --- Perform classifications for each dataset and model ---
    # Get datasets for classification and iterate through them
    datasets = get_data_from_csv(ProjectPaths.DATASETS_FOR_CLASSIFICATIONS)
        
    # Iterate through each dataset and classify using the saved models
    for d in datasets.to_dict('records'):
        dataset_type = d['dataset_type']
        dataset_path = Path(d['path'])

        # Get the data from the dataset CSV
        data = get_data_from_csv(dataset_path)

        # Get the paths of the saved models from the models registry
        models_paths = get_data_from_csv(models_registry_file)['path']

        # Iterate through each model and perform classification
        for model_path in models_paths:
            classification_processing(
                model_path=Path(model_path), 
                data=data, 
                dataset_type=dataset_type, 
                dataset_name=dataset_path.stem,
                classifications_file=classifications_file,
                plots_dir=plots_dir
            )

    # --- Group and Save Classification Results ---
    # Group classifications by model and save them in the corresponding directory
    group_by_model_and_save(data=classifications_file, dst_dir=group_by_model_dir)

    # Group classifications by dataset type and save them in the corresponding directory
    group_by_classes_and_save(data=classifications_file, dst_dir=group_by_classes_dir)

    # --- Generate Performance Plots (F1-Score, Precision, Recall) ---
    # If the flag is enabled, generate performance plots based on the classification results
    if PlotFlags.ENABLE_PERFORMANCE_PLOTS:
        save_heatmap_for_metrics_plot(
            model_type=model_type,
            models=get_data_from_csv(models_metadata_file), 
            data=get_data_from_csv(classifications_file),
            dst_dir=plots_dir
        )

    print(f"\n--- Routine Classification for {model_type} Model Building (random_state={seed}) Completed ---")


def _model_building(model_type, seed):
    """ Executes a predefined model building routine """
    print(f"\n--- Starting {model_type} Model Building (random_state={seed}) Phase ---")

    # Get dataset for random forest models
    datasets = get_data_from_csv(ProjectPaths.DATASETS_FOR_MODEL_BUILDING)

    # Iterate through each dataset and build models
    for d in datasets.to_dict('records'):
        model_processing(
            data=get_data_from_csv(Path(d['path'])), 
            dataset_type=d['dataset_type'], 
            model_type=model_type,
            seed=seed
        )

    print(f"\n--- Routine {model_type} Model Building (random_state={seed}) Completed ---")


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
    jobs = [(ProjectPaths.DATASETS_FOR_MODEL_BUILDING, RoutineConfig.DATASETS_TARGETS_FOR_MODEL_BUILDING)]
    
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

    # --- Save Plots Based on Flags ---
    # Save PCA INDEPENDENT plot if enabled
    if PlotFlags.ENABLE_PCA_PLOTS:
        save_all_pca_plots(
            config_csv_path=ProjectPaths.DATASETS_FOR_CLASSIFICATIONS
        )

    # Save Feature KDE distributions plot if enabled
    if PlotFlags.ENABLE_KDE_PLOTS:
        save_all_kde_plots(
            config_csv_path=ProjectPaths.DATASETS_FOR_CLASSIFICATIONS,
            features_to_plot=MLConstants.KDE_TOP_FEATURES
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
        print("3. Run CLASSIFICATIONS")
        print("4. Run WELCH T-TEST")
        print("5. Run ALL!")
        print("6. Exit application")
        print("="*60)
        
        # Ask user's response
        main_choice = input("Select execution mode (1, 2, 3, 4, 5 or 6): ")
        
        # Do user's choice
        if main_choice == '1':  # Preprocessing case
            _preprocessing()
        elif main_choice == '2':    # Model building case
            for model_type, seed in product(MLConstants.MODEL_TYPE, MLConstants.SEEDS):
                _model_building(model_type, seed)
        elif main_choice == '3':    # Classifications case
            # for model_type, seed in product(MLConstants.MODEL_TYPE, MLConstants.SEEDS):
            #     _classifications(model_type, seed)

            for model_type in MLConstants.MODEL_TYPE:
                _aggregated_heatmaps(model_type)
        elif main_choice == '4':    # Welch t-test case
            for model_type in MLConstants.MODEL_TYPE:
                _welch_ttest(model_type)
        elif main_choice == '5':    # All case
            _run_all_phases()
        elif main_choice == '6':    # Exit case
            print("\nExiting application. Goodbye!")
            break
        else:
            print("\nInvalid option! Please choose 1, 2, 3, 4, 5 or 6.")


if __name__ == "__main__":
    main()
