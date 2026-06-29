"""
Main entry point for the Satellite IDS project.
"""
from pathlib import Path

from classifications.classification import classification_processing
from models.models import model_processing
from plotting.plotting import plotting_processing
from utils.file_utils import (
    create_directory, get_data_from_csv, group_by_classes_and_save, 
    group_by_model_and_save, group_datasets_paths_for_filename_list)
from utils.input_utils import validate_choice
from utils.paths import (
    BY_DATASET_DIR_NAME, BY_MODEL_DIR_NAME, CLASSIFICATIONS_DIR, CLASSIFICATIONS_FILE, 
    DATASETS, DATASETS_FOR_CLASSIFICATIONS_PATH, DATASETS_FOR_MODEL_BUILDING_PATH, DATASETS_INFO_PATH, 
    FILENAMES_DATASETS_FOR_CLASSIFICATIONS, FILENAMES_DATASETS_FOR_MODEL_BUILDING, MODELS_DIR, MODELS_PATHS, 
    NB15_PREFIX, NORMALIZED, PLOTTING_METRICS, SAT20_PREFIX, TER20_PREFIX, UNNORMALIZED)
from preprocessing.data_preprocessing import data_preprocessing
from preprocessing.file_preprocessing import hybrid_dataset_file_preprocessing, single_dataset_file_preprocessing


def _run_all_phases():
    """ Executes all phases of the pipeline in a single run for both normalized and unnormalized modes """
    print("\n=== FULL AUTOMATED PIPELINE (NON-STOP: BOTH MODES) ===")

    for mode in [NORMALIZED, UNNORMALIZED]:
        print(f"\n--- Running Full Pipeline for {mode.upper()} Mode ---")
        _preprocessing()
        _model_building()
        _classifications()
        _plotting()
        
        print(f"\n--- Full Pipeline Completed for {mode.upper()} Mode ---")

    print("\n=== FULL PIPELINE AUTOMATICALLY COMPLETED FOR BOTH MODES ===")


def _plotting():
    """ Show classifications on TNR and TPR plots """
    print("\n--- Plotting Phase ---")
    
    # Ask to user which mode he wants to start (normalized or unnormalized)
    mode_input = input(f"Choose routine pipeline mode: [{NORMALIZED} or {UNNORMALIZED}] ").lower()
    mode = validate_choice(mode_input, [NORMALIZED, UNNORMALIZED], "mode")

    data = get_data_from_csv(CLASSIFICATIONS_DIR / mode / CLASSIFICATIONS_FILE)
    plotting_processing(data, mode, PLOTTING_METRICS)

    print(f"\n--- Routine Plotting Phase Completed for {mode.upper()} Mode ---")


def _classifications():
    """ Evaluates saved models on specific testing datasets """
    print("\n--- Starting Classification Phase ---")
    
    # Ask to user which mode he wants to start (normalized or unnormalized)
    mode_input = input(f"Choose routine pipeline mode: [{NORMALIZED} or {UNNORMALIZED}] ").lower()
    mode = validate_choice(mode_input, [NORMALIZED, UNNORMALIZED], "mode")

    # Do classification process for each classification task
    datasets = get_data_from_csv(DATASETS_FOR_CLASSIFICATIONS_PATH)
    for d in datasets.to_dict('records'):
        dataset_type = d['dataset_type']
        dataset_path = d['path']

        data = get_data_from_csv(Path(dataset_path))
        models_paths = get_data_from_csv(MODELS_DIR / mode / MODELS_PATHS)['path']     # Get all models paths for the selected mode
        for model_path in models_paths:
            classification_processing(Path(model_path), data, dataset_type, mode)

    # Group classifications by model and by dataset type and save them in separate directories
    curr_classifications_dir = CLASSIFICATIONS_DIR / mode
    group_by_model_dir = create_directory(BY_MODEL_DIR_NAME, curr_classifications_dir)
    group_by_classes_dir = create_directory(BY_DATASET_DIR_NAME, curr_classifications_dir)
    group_by_model_and_save(curr_classifications_dir / CLASSIFICATIONS_FILE, group_by_model_dir)
    group_by_classes_and_save(curr_classifications_dir / CLASSIFICATIONS_FILE, group_by_classes_dir)

    print("\n--- Routine Classification Completed ---")


def _model_building():
    """ Executes a predefined model building routine """
    print("\n--- Starting Model Building Phase ---")
    
    # Ask to user which mode he wants to start (normalized or unnormalized)
    mode_input = input(f"Choose routine pipeline mode: [{NORMALIZED} or {UNNORMALIZED}] ").lower()
    mode = validate_choice(mode_input, [NORMALIZED, UNNORMALIZED], "mode")

    # Start Model Processing for each model building dataset
    datasets = get_data_from_csv(DATASETS_FOR_MODEL_BUILDING_PATH)
    for d in datasets.to_dict('records'):
        dataset_type = d['dataset_type']
        dataset_path = d['path']

        data = get_data_from_csv(Path(dataset_path))
        model_processing(data, dataset_type, mode)

    print("\n--- Routine Model Building Completed ---")


def _preprocessing():
    """ Executes a predefined preprocessing routine for nb15, sat20 ter20 and hybrid datasets """
    print("\n--- Starting Preprocessing Phase ---")

    # Initialize variables for hybrid dataset
    nb15_normal_data = None
    sat20_anomaly_data = None
    ter20_anomaly_data = None
    
    # Do preprocessing for each dataset
    for d in DATASETS:
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
        if dataset_type == NB15_PREFIX:
            nb15_normal_data = data_prep[data_prep['label'] == 0]
        elif dataset_type == SAT20_PREFIX:
            sat20_anomaly_data = data_prep[data_prep['label'] == 1]
        elif dataset_type == TER20_PREFIX:
            ter20_anomaly_data = data_prep[data_prep['label'] == 1]
    
    # Do file preprocessing for a hybrid dataset
    hybrid_dataset_file_preprocessing(nb15_normal_data, sat20_anomaly_data, ter20_anomaly_data)

    # Group datasets paths for model building and save them in a csv file
    group_datasets_paths_for_filename_list(
        DATASETS_INFO_PATH, DATASETS_FOR_MODEL_BUILDING_PATH, FILENAMES_DATASETS_FOR_MODEL_BUILDING
    )

    # Group datasets paths for classifications and save them in a csv file
    group_datasets_paths_for_filename_list(
        DATASETS_INFO_PATH, DATASETS_FOR_CLASSIFICATIONS_PATH, FILENAMES_DATASETS_FOR_CLASSIFICATIONS
    )

    print("\n--- Routine Preprocessing Phase Completed ---")


def main():
    """ Main entry point of the application with a main dashboard menu """
    # setup_project_directories()

    while True:
        # Print main dashboard menu
        print("\n" + "="*60)
        print("      SATELLITE IDS - MAIN DASHBOARD")
        print("="*60)
        print("1. Run PREPROCESSING")
        print("2. Run MODEL BUILDING")
        print("3. Run CLASSIFICATIONS")
        print("4. Run PLOTTING")
        print("5. Run ALL!")
        print("6. Exit application")
        print("="*60)
        
        # Ask user's response
        main_choice = input("Select execution mode (1, 2, 3, 4, 5 or 6): ")
        
        # Do user's choice
        if main_choice == '1':  # Preprocessing case
            _preprocessing()
        elif main_choice == '2':    # Model building case
            _model_building()
        elif main_choice == '3':    # Classifications case
            _classifications()
        elif main_choice == '4':    # Plotting case
            _plotting()
        elif main_choice == '5':    # All case
            _run_all_phases()
        elif main_choice == '6':    # Exit case
            print("\nExiting application. Goodbye!")
            break
        else:
            print("\nInvalid option! Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()