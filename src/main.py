"""
Main entry point for the Satellite IDS project.
"""
import pathlib
import joblib # Keep joblib for the interactive classification_loop

from classification.classification import classification_processing
from models.models import model_processing
from utils.file_utils import get_data_from_csv
from utils.input_utils import get_split_input, validate_path, validate_choice, get_y_n_choice
from preprocessing.data_preprocessing import data_preprocessing
from preprocessing.file_preprocessing import file_preprocessing, create_joint_datasets
from classification.classification import run_routine_classifications # Import the new routine function
from utils.paths import INDEPENDENT_DIR, DEPENDENT_DIR, NB15_RAW_PATH, SAT20_RAW_PATH, TER20_RAW_PATH
from utils.paths import INDEPENDENT_DIR, DEPENDENT_DIR


# --- State Wrappers ---
def file_preprocessing_state(data_preprocessed, dataset_type, base_dir, normal_attack_ratio=None, replacing_mode=None):
    """
    Wrapper function to trigger the file-level preprocessing state.

    :param data_preprocessed: The DataFrame after data preprocessing.
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    :param base_dir: Path where results should be stored.
    :param normal_attack_ratio: Optional ratio for NB15 balancing.
    :param replacing_mode: Optional boolean for sampling with replacement.
    """
    file_preprocessing(data_preprocessed, dataset_type, base_dir, normal_attack_ratio, replacing_mode)


def data_preprocessing_state(path, dataset_type, dependent=True, scaler_stats=None):
    """
    Wrapper function to load data and trigger the data-level preprocessing state.

    :param path: Path to the raw CSV file.
    :param dataset_type: Type of the dataset (nb15, sat20, ter20).
    :param dependent: Boolean for normalization type.
    :param scaler_stats: Precomputed normalization parameters.
    :return: Tuple (Preprocessed DataFrame, scaler_stats).
    """
    data = get_data_from_csv(path)
    data_preprocessed, scaler_stats = data_preprocessing(data, dataset_type, dependent, scaler_stats)

    return data_preprocessed, scaler_stats


# --- Runtime Loops ---
def preprocessing_loop():
    """
    Interactive loop for running the data and file preprocessing phase.
    """
    print("\n--- Starting Preprocessing Phase ---")
    mode_input = input("Choose preprocessing mode: [independent or dependent] ").lower()
    mode = validate_choice(mode_input, ['independent', 'dependent'], "mode")

    if mode == 'dependent':
        print("\n[DEPENDENT MODE] First dataset will be used for Fit (scaler calculation).")
        prompt = "Insert the path and the dataset_type for FIT: [path dataset_type] "
        user_input = get_split_input(prompt, 2)
        path = validate_path(user_input[0])
        dataset_type = validate_choice(user_input[1], ['nb15', 'sat20', 'ter20'], "dataset type")

        # Fit phase
        data_pre, scaler_stats = data_preprocessing_state(path, dataset_type, dependent=True, scaler_stats=None)
        file_preprocessing_state(data_pre, dataset_type, DEPENDENT_DIR)

        # Transform phase for subsequent datasets
        while get_y_n_choice("Do you want to add a dataset for TRANSFORM using calculated stats? [y/n] ") == 'y':
            prompt = "Insert the path and the dataset_type for TRANSFORM: [path dataset_type] "
            user_input = get_split_input(prompt, 2)
            path = validate_path(user_input[0])
            dataset_type = validate_choice(user_input[1], ['nb15', 'sat20', 'ter20'], "dataset type")

            data_pre, _ = data_preprocessing_state(path, dataset_type, dependent=True, scaler_stats=scaler_stats)
            file_preprocessing_state(data_pre, dataset_type, DEPENDENT_DIR)
    
    else: # Independent mode
        user_choice = 'y'
        while user_choice == 'y':
            prompt = "Insert path and dataset_type for independent preprocessing: [path dataset_type] "
            user_input = get_split_input(prompt, 2)
            path = validate_path(user_input[0])
            dataset_type = validate_choice(user_input[1], ['nb15', 'sat20', 'ter20'], "dataset type")

            data_pre, _ = data_preprocessing_state(path, dataset_type, dependent=False)
            file_preprocessing_state(data_pre, dataset_type, INDEPENDENT_DIR)
            user_choice = get_y_n_choice("Do you want to process another independent dataset? [y/n] ")


def run_routine_preprocessing():
    """
    Executes a predefined preprocessing routine for nb15, sat20, and ter20.
    Uses a fixed ratio of 10 and no replacement for NB15.
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
    
    scaler_stats = None # Will store scaler stats if in dependent mode

    for i, ds in enumerate(datasets):
        if not ds['path'].exists():
            print(f"Warning: Dataset file not found at {ds['path']}. Skipping {ds['type']}.")
            continue
            
        print(f"\n[ROUTINE] Processing {ds['type']} ({mode.capitalize()} Mode)...")
        
        current_base_dir = DEPENDENT_DIR if mode == 'dependent' else INDEPENDENT_DIR

        if mode == 'dependent':
            data_preprocessed, new_scaler_stats = data_preprocessing_state(ds['path'], ds['type'], dependent=True, scaler_stats=scaler_stats)
            if scaler_stats is None: # If it's the first dataset, capture the scaler_stats
                scaler_stats = new_scaler_stats
        else: # Independent mode
            data_preprocessed, _ = data_preprocessing_state(ds['path'], ds['type'], dependent=False)
        
        if ds['type'] == 'nb15':
            file_preprocessing_state(data_preprocessed, ds['type'], current_base_dir, normal_attack_ratio=nb15_ratio, replacing_mode=nb15_replacing)
        else:
            file_preprocessing_state(data_preprocessed, ds['type'], current_base_dir)

    # Final Joint Preprocessing Phase
    create_joint_datasets(current_base_dir, ratio=nb15_ratio, replacing_mode=nb15_replacing)

    print("\n--- Routine Preprocessing Phase Completed ---")


def build_model_loop():
    """
    Interactive loop for selecting, training, and saving machine learning models.
    """
    print("\n--- Starting Model Building Phase ---")
    mode_input = input("Choose model building mode: [independent or dependent] ").lower()
    mode = validate_choice(mode_input, ['independent', 'dependent'], "mode")

    user_choice = 'y'
    while user_choice == 'y':
        model_input = input("Choose which model do you want to use: [random forest or isolation forest] ").lower()
        model_type = validate_choice(model_input, ['random forest', 'isolation forest'], "model type")

        dataset_type_input = input("Insert the dataset_type of the training dataset: (nb15, sat20 or ter20) ").lower()
        dataset_type = validate_choice(dataset_type_input, ['nb15', 'sat20', 'ter20'], "dataset type")

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


def main():
    """
    Main entry point of the application. 
    Handles user input, validation, and orchestrates the preprocessing pipeline.

    :internal user_choice: Tracks whether the user wants to continue or exit.
    :internal user_input: Captures the path and dataset type provided by the user.
    """
    # Routine Preprocessing state
    user_choice = get_y_n_choice("Do you want to run routine preprocessing? (NB15, SAT20, TER20) [y/n] ")
    if user_choice == 'y':
        run_routine_preprocessing()

    user_choice = get_y_n_choice("Do you want to start interactive preprocessing? [y/n] ")
    if user_choice == 'y':
        preprocessing_loop()

    user_choice = get_y_n_choice("Do you want to build a model? [y/n] ")
    if user_choice == 'y':
        build_model_loop()

    # New routine classification state
    user_choice = get_y_n_choice("Do you want to run routine classifications? [y/n] ")
    if user_choice == 'y':
        print("\n--- Starting Routine Classification Phase ---")
        run_routine_classifications()
        print("--- Routine Classification Phase Completed ---")

    user_choice = get_y_n_choice("Do you want to start the classification? [y/n] ")
    if user_choice == 'y':
        classification_loop()


if __name__ == "__main__":
    main()