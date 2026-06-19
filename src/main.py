"""
Main entry point for the Satellite IDS project.
"""
import joblib  # Keep joblib for the interactive classification_loop

from utils.file_utils import get_data_from_csv
from utils.input_utils import get_split_input, validate_path, validate_choice, get_y_n_choice, get_numeric_input
from utils.paths import INDEPENDENT_DIR, DEPENDENT_DIR, NB15_RAW_PATH, SAT20_RAW_PATH, TER20_RAW_PATH, setup_project_directories
from preprocessing.data_preprocessing import data_preprocessing
from preprocessing.file_preprocessing import file_preprocessing, create_joint_datasets
from models.models import model_processing, run_routine_models
from classification.classification import classification_processing, run_routine_classifications

# --- State Wrappers ---
def file_preprocessing_state(data_preprocessed, dataset_type, base_dir, normal_attack_ratio=None, replacing_mode=None):
    """
    Wrapper function to trigger the file-level preprocessing state.
    """
    file_preprocessing(data_preprocessed, dataset_type, base_dir, normal_attack_ratio, replacing_mode)


def data_preprocessing_state(path, dataset_type, dependent=True, scaler_stats=None, train_split=0.8):
    """
    Wrapper function to load data and trigger the data-level preprocessing state.
    """
    data = get_data_from_csv(path)
    data_preprocessed, scaler_stats = data_preprocessing(data, dataset_type, dependent, scaler_stats, train_split)
    return data_preprocessed, scaler_stats


# --- Runtime Loops ---
def preprocessing_loop():
    """
    Interactive loop for running the data and file preprocessing phase.
    """
    print("\n--- Starting Preprocessing Phase ---")
    mode_input = input("Choose preprocessing mode: [independent or dependent] ").lower()
    mode = validate_choice(mode_input, ['independent', 'dependent'], "mode")

    train_split = get_numeric_input("Insert the train split percentage (e.g. 80): ", type_func=float, min_val=1, max_val=99) / 100

    if mode == 'dependent':
        print("\n[DEPENDENT MODE] First dataset will be used for Fit (scaler calculation).")
        prompt = "Insert the path and the dataset_type for FIT: [path dataset_type] "
        user_input = get_split_input(prompt, 2)
        path = validate_path(user_input[0])
        dataset_type = validate_choice(user_input[1], ['nb15', 'sat20', 'ter20'], "dataset type")

        data_pre, scaler_stats = data_preprocessing_state(path, dataset_type, dependent=True, scaler_stats=None, train_split=train_split)
        file_preprocessing_state(data_pre, dataset_type, DEPENDENT_DIR)

        while get_y_n_choice("Do you want to add a dataset for TRANSFORM using calculated stats? [y/n] ") == 'y':
            prompt = "Insert the path and the dataset_type for TRANSFORM: [path dataset_type] "
            user_input = get_split_input(prompt, 2)
            path = validate_path(user_input[0])
            dataset_type = validate_choice(user_input[1], ['nb15', 'sat20', 'ter20'], "dataset type")

            data_pre, _ = data_preprocessing_state(path, dataset_type, dependent=True, scaler_stats=scaler_stats, train_split=train_split)
            file_preprocessing_state(data_pre, dataset_type, DEPENDENT_DIR)
    
    else: 
        user_choice = 'y'
        while user_choice == 'y':
            prompt = "Insert path and dataset_type for independent preprocessing: [path dataset_type] "
            user_input = get_split_input(prompt, 2)
            path = validate_path(user_input[0])
            dataset_type = validate_choice(user_input[1], ['nb15', 'sat20', 'ter20'], "dataset type")

            data_pre, _ = data_preprocessing_state(path, dataset_type, dependent=False, train_split=train_split)
            file_preprocessing_state(data_pre, dataset_type, INDEPENDENT_DIR)
            user_choice = get_y_n_choice("Do you want to process another independent dataset? [y/n] ")


def run_routine_preprocessing():
    """
    Executes a predefined preprocessing routine for nb15, sat20, and ter20.
    """
    print("\n--- Starting Routine Preprocessing Phase ---")

    mode_input = input("Choose routine preprocessing mode: [independent or dependent] ").lower()
    mode = validate_choice(mode_input, ['independent', 'dependent'], "mode")
    
    datasets = [
        {'type': 'nb15', 'path': NB15_RAW_PATH},
        {'type': 'sat20', 'path': SAT20_RAW_PATH},
        {'type': 'ter20', 'path': TER20_RAW_PATH}
    ]
    
    nb15_ratio = 10.0
    nb15_replacing = False
    train_split = 0.8
    scaler_stats = None 
    
    current_base_dir = DEPENDENT_DIR if mode == 'dependent' else INDEPENDENT_DIR

    for i, ds in enumerate(datasets):
        if not ds['path'].exists():
            print(f"Warning: Dataset file not found at {ds['path']}. Skipping {ds['type']}.")
            continue
            
        print(f"\n[ROUTINE] Processing {ds['type']} ({mode.capitalize()} Mode)...")

        if mode == 'dependent':
            data_preprocessed, new_scaler_stats = data_preprocessing_state(ds['path'], ds['type'], dependent=True, scaler_stats=scaler_stats, train_split=train_split)
            if scaler_stats is None: 
                scaler_stats = new_scaler_stats
        else: 
            data_preprocessed, _ = data_preprocessing_state(ds['path'], ds['type'], dependent=False, train_split=train_split)
        
        if ds['type'] == 'nb15':
            file_preprocessing_state(data_preprocessed, ds['type'], current_base_dir, normal_attack_ratio=nb15_ratio, replacing_mode=nb15_replacing)
        else:
            file_preprocessing_state(data_preprocessed, ds['type'], current_base_dir)

    create_joint_datasets(current_base_dir, ratio=nb15_ratio, replacing_mode=nb15_replacing)
    print("\n--- Routine Preprocessing Phase Completed ---")


def build_model_loop():
    """
    Interactive loop for selecting, training, and saving machine learning models.
    """
    print("\n--- Starting Model Building Phase ---")
    mode_input = input("Choose model building mode: [independent or dependent] ").lower()
    mode = validate_choice(mode_input, ['independent', 'dependent'], "mode")

    routine_choice = get_y_n_choice(f"Do you want to run the routine model building for {mode} mode? [y/n] ")
    if routine_choice == 'y':
        run_routine_models(mode)
        if get_y_n_choice("Do you want to build additional models manually? [y/n] ") == 'n':
            return

    user_choice = 'y'
    while user_choice == 'y':
        model_input = input("Choose which model do you want to use: [random forest or isolation forest] ").lower()
        model_type = validate_choice(model_input, ['random forest', 'isolation forest'], "model type")

        dataset_type_input = input("Insert the dataset_type of the training dataset: (nb15, sat20, ter20, nb15+sat20, nb15+ter20) ").lower()
        dataset_type = validate_choice(dataset_type_input, ['nb15', 'sat20', 'ter20', 'nb15+sat20', 'nb15+ter20'], "dataset type")

        path_input = input("Insert the path of the dataset: [path] ")
        data_path = validate_path(path_input)

        data = get_data_from_csv(data_path)
        model_processing(data, mode, model_type, dataset_type, data_path.stem)

        user_choice = get_y_n_choice("Do you want to build a new model? [y/n] ")


def classification_loop():
    """
    Interactive loop for evaluating saved models on specific testing datasets.
    """
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
    """
    Executes the pipeline in Routine mode but guides the user step-by-step.
    """
    print("\n=== AUTOMATED ROUTINE PIPELINE (GUIDED STEP-BY-STEP) ===")
    
    if get_y_n_choice("Do you want to execute the routine PREPROCESSING phase? [y/n] ") == 'y':
        run_routine_preprocessing(mode=None)
    else:
        print("Routine preprocessing skipped.")
    
    if get_y_n_choice("Do you want to execute the routine MODEL BUILDING phase? [y/n] ") == 'y':
        mode_input = input("Choose routine model building mode: [independent or dependent] ").lower()
        mode = validate_choice(mode_input, ['independent', 'dependent'], "mode")
        run_routine_models(mode)
    else:
        print("Routine model building skipped.")

    if get_y_n_choice("Do you want to execute the routine CLASSIFICATION phase? [y/n] ") == 'y':
        mode_input = input("Choose routine classification mode: [independent or dependent] ").lower()
        mode = validate_choice(mode_input, ['independent', 'dependent'], "mode")
        run_routine_classifications(mode)
    else:
        print("Routine classification skipped.")

    print("\nRoutine pipeline session concluded.")


def run_full_automated_pipeline():
    """
    Executes all routines non-stop (Pre-processing -> Training -> Testing) 
    using a user-selected mode without further interruptions.
    """
    print("\n=== FULL AUTOMATED ROUTINE PIPELINE (NON-STOP) ===")
    
    # 1. Ask mode ONCE here
    mode_input = input("Choose FULL pipeline mode: [independent or dependent] ").lower()
    mode = validate_choice(mode_input, ['independent', 'dependent'], "mode")
    
    print(f"\n--- 1. Executing Non-Stop Routine Preprocessing ({mode}) ---")
    run_routine_preprocessing(mode=mode)
    
    print(f"\n--- 2. Executing Non-Stop Routine Model Building ({mode}) ---")
    run_routine_models(mode)
    
    print(f"\n--- 3. Executing Non-Stop Routine Classifications ({mode}) ---")

    run_routine_classifications(mode) 
    
    print("\n=== FULL PIPELINE AUTOMATICALLY COMPLETED ===")


def run_manual_pipeline():
    """
    Secondary menu for manual, step-by-step interactive operations.
    """
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


def main():
    """
    Main entry point of the application with a main dashboard menu.
    """
    setup_project_directories()

    while True:
        print("\n" + "="*60)
        print("      SATELLITE IDS - MAIN DASHBOARD")
        print("="*60)
        print("1. Run AUTOMATED ROUTINES (Step-by-step guided choice)")
        print("2. Run FULL AUTOMATED ROUTINES non-stop (Pre -> Train -> Test)")
        print("3. Manage the pipeline STEP-BY-STEP manually (Interactive)")
        print("4. Exit application")
        print("="*60)
        
        main_choice = input("Select execution mode (1, 2, 3, or 4): ")
        
        if main_choice == '1':
            run_guided_routine_pipeline()
        elif main_choice == '2':
            run_full_automated_pipeline()
        elif main_choice == '3':
            run_manual_pipeline()
        elif main_choice == '4':
            print("\nExiting application. Goodbye!")
            break
        else:
            print("\nInvalid option! Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()