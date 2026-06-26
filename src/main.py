"""
Main entry point for the Satellite IDS project.
"""
import joblib

from classifications.classification import classification_processing
from models.models import model_processing
from utils.file_utils import get_data_from_csv
# from utils.input_utils import get_split_input, validate_path, validate_choice, get_y_n_choice, get_numeric_input
from utils.input_utils import validate_choice, validate_path
from utils.paths import CLASSIFICATIONS, DATASETS, MODELS_PATH_FILE, NB15_PREFIX, NB15_RAW_PATH, NORMALIZED, ROUTINE_MODELS, SAT20_PREFIX, SAT20_RAW_PATH, TER20_PREFIX, TER20_RAW_PATH, UNNORMALIZED # setup_project_directories
from preprocessing.data_preprocessing import data_preprocessing
from preprocessing.file_preprocessing import hybrid_dataset_file_preprocessing, single_dataset_file_preprocessing
# from models.models import model_processing, run_routine_models
# from classification.classification import classification_processing, run_routine_classifications
# from view.plotting import generate_custom_recall_heatmap
from pathlib import Path



"""


# --- Runtime Loops ---

        




def classification_loop():
    Interactive loop for evaluating saved models on specific testing datasets
    print("\n--- Starting Classification Phase ---")
    mode_input = input("Choose classification mode: [independent or dependent] ").lower()
    mode = validate_choice(mode_input, ['independent', 'dependent'], "mode")

    user_choice = 'y'
    while user_choice == 'y':
        path_input = input("Insert the path of the dataset for testing: [path] ")
        data_path = validate_path(path_input)

        dataset_type_input = input("Insert the dataset_type of the testing dataset: (nb15, sat20 or ter20) ").lower()
        dataset_type = validate_choice(dataset_type_input, ['nb15', 'sat20', 'ter20'], "dataset type")

        data = get_data_from_csv(data_path)
        
        models_to_test = []
        while True:
            model_path_input = input("Insert the path of the model: [path] ")
            model_path = validate_path(model_path_input)

            model = joblib.load(model_path)
            models_to_test.append({
                'model_obj': model,
                'model_name': model_path.stem
            })
            
            if get_y_n_choice("Do you want to add another model to test on this dataset? [y/n] ") == 'n':
                break

        classification_processing(data, mode, models_to_test, dataset_type, data_path.stem)
        user_choice = get_y_n_choice("Do you want to start a new classification session (new dataset)? [y/n] ")


def run_guided_routine_pipeline():
    Executes the pipeline in Routine mode but guides the user step-by-step
    print("\n=== AUTOMATED ROUTINE PIPELINE (GUIDED STEP-BY-STEP) ===")
    
    # SOLVED: Asked once here to maintain pipeline consistency across all steps
    mode_input = input("Choose routine pipeline mode: [independent or dependent] ").lower()
    mode = validate_choice(mode_input, ['independent', 'dependent'], "mode")

    if get_y_n_choice("Do you want to execute the routine PREPROCESSING phase? [y/n] ") == 'y':
        run_routine_preprocessing(mode)
    
    if get_y_n_choice("Do you want to execute the routine MODEL BUILDING phase? [y/n] ") == 'y':
        print(f"\n--- Running Routine Model Building ({mode.upper()} Mode) ---")
        run_routine_models(mode)

    if get_y_n_choice("Do you want to execute the routine CLASSIFICATION phase? [y/n] ") == 'y':
        print(f"\n--- Running Routine Classification ({mode.upper()} Mode) ---")
        run_routine_classifications(mode)

    if get_y_n_choice("Do you want to execute the routine PLOTTING phase? [y/n] ") == 'y':
        print(f"\n--- Running Routine Plotting ({mode.upper()} Mode) ---")
        generate_custom_recall_heatmap(mode)

    print("\nRoutine pipeline session concluded.")


def run_full_automated_pipeline():
    
    Executes all routines non-stop (Pre-processing -> Training -> Testing) 
    for BOTH independent and dependent modes without any interruptions.
    
    print("\n=== FULL AUTOMATED ROUTINE PIPELINE (NON-STOP: BOTH MODES) ===")
    
    modes_to_run = ['independent', 'dependent']
    
    for mode in modes_to_run:
        print("\n" + "*"*50)
        print(f"      STARTING PIPELINE FOR: {mode.upper()} MODE")
        print("*"*50)
        
        print(f"\n--- 1. Executing Non-Stop Routine Preprocessing ({mode}) ---")
        run_routine_preprocessing(mode)
        
        print(f"\n--- 2. Executing Non-Stop Routine Model Building ({mode}) ---")
        run_routine_models(mode)
        
        print(f"\n--- 3. Executing Non-Stop Routine Classifications ({mode}) ---")
        run_routine_classifications(mode)

        print(f"\n--- 4. Executing Non-Stop Routine Plotting ({mode}) ---")
        generate_custom_recall_heatmap(mode)
        
    print("\n=== FULL PIPELINE AUTOMATICALLY COMPLETED FOR BOTH MODES ===")


def run_manual_pipeline():
    Secondary menu for manual, step-by-step interactive operations
    while True:
        print("\n=== MANUAL STEP-BY-STEP FLOW ===")
        print("1. Run Interactive Preprocessing")
        print("2. Build/Train Interactive Model")
        print("3. Start Classification / Test Model")
        print("4. Return to main menu")
        
        choice = input("Select an option (1-4): ")
        
        if choice == '1':
            preprocessing_loop()
        elif choice == '2':
            build_model_loop()
        elif choice == '3':
            classification_loop()
        elif choice == '4':
            print("Returning to main menu...")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 4.")


"""



def _classifications():
    """ Interactive loop for evaluating saved models on specific testing datasets """
    print("\n--- Starting Classification Phase ---")
    
    # Ask to user which mode he wants to start (normalize or unnormalize)
    mode_input = input(f"Choose routine pipeline mode: [{NORMALIZED} or {UNNORMALIZED}] ").lower()
    mode = validate_choice(mode_input, [NORMALIZED, UNNORMALIZED], "mode")

    # Do classification process for each classification task
    for classification in CLASSIFICATIONS:
        dataset_type = classification['dataset_type']
        data = get_data_from_csv(classification['path'])

        models_paths = get_data_from_csv(MODELS_PATH_FILE)['path']
        for model_path in models_paths:
            model = joblib.load(model_path)

            classification_processing(model, data, dataset_type, mode)

    print("\n--- Routine Classification Completed ---")


def _model_building():
    """ Executes a predefined model building routine """
    print("\n--- Starting Model Building Phase ---")
    
    # Ask to user which mode he wants to start (normalize or unnormalize)
    mode_input = input(f"Choose routine pipeline mode: [{NORMALIZED} or {UNNORMALIZED}] ").lower()
    mode = validate_choice(mode_input, [NORMALIZED, UNNORMALIZED], "mode")

    # Start Model Processing
    for routine_model in ROUTINE_MODELS:
        dataset_path = routine_model['path']
        data = get_data_from_csv(dataset_path)
        dataset_type = routine_model['dataset_type']
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
            # TODO
            return
        elif main_choice == '5':    # All case
            # TODO
            return
        elif main_choice == '6':    # Exit case
            print("\nExiting application. Goodbye!")
            break
        else:
            print("\nInvalid option! Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()