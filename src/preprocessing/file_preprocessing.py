import pathlib
import pandas as pd
import sys

from utils.file_utils import create_directory, create_csv_from_data, get_data_from_csv
from utils.input_utils import get_numeric_input, get_y_n_bool
from utils.paths import (
    DATA_DIR, ATTACK_CAT_DIR_NAME,
    NORMAL_ATTACK_DIR_NAME, JOINT_DIR_NAME,
    NORMAL_FILE_STEM, ATTACKS_FILE_STEM,
    NB15_SCALED_FILE_STEM, NB15_PREPROCESSED_DIR_NAME,
    JOINT_NORMAL_SAT20_FILE_STEM, JOINT_NORMAL_TER20_FILE_STEM,
    PREPROCESSED_DIR_SUFFIX, PREPROCESSED_MAIN_FILE_SUFFIX
)

def split_by_attack_cat(data, dest_path):
    """
    Splits the dataset into multiple CSV files based on the attack category.

    :param data: The pandas DataFrame to split.
    :param dest_path: Path where the 'attack_cat' directory will be created.
    :return: The Path object to the directory containing the split CSV files.
    """
    attack_cat_dir_path = create_directory(ATTACK_CAT_DIR_NAME, dest_path)
    if 'attack_cat' not in data.columns:
        print("Error: 'attack_cat' column not found during split.")
        sys.exit(1)
        
    attack_cat_list = data['attack_cat'].unique()

    for attack_cat in attack_cat_list:
        attack_cat_data = data[data['attack_cat'] == attack_cat].copy() # Use .copy() to avoid SettingWithCopyWarning
        create_csv_from_data(attack_cat_data, attack_cat, attack_cat_dir_path)
    print("Dataset split by attack category.")
    
    return attack_cat_dir_path


def merge_normal_attack(source_path, dest_path, normal_attack_ratio, replacing_mode):
    """
    Creates merged datasets containing both 'Normal' and a specific 'Attack' category, balanced by a ratio.

    :param source_path: Path to the directory containing split category CSVs.
    :param dest_path: Destination path for the merged files.
    :param normal_attack_ratio: The numeric ratio (Normal/Attack) for balancing.
    :param replacing_mode: Boolean indicating if sampling should use replacement.
    :internal attack_cat_list: List of Path objects for each category file.
    :return: The Path object to the 'normal_attack' directory.
    """
    normal_attack_dir_path = create_directory(NORMAL_ATTACK_DIR_NAME, dest_path)
    attack_cat_list = list(source_path.iterdir())
    
    normal_cat_path = next((attack_cat for attack_cat in attack_cat_list if attack_cat.stem == NORMAL_FILE_STEM), None)
    if normal_cat_path is None:
        print("Error: Normal category not found!")
        return normal_attack_dir_path
    
    normal_data = pd.DataFrame(get_data_from_csv(normal_cat_path))
    normal_samples = normal_data.shape[0]

    for attack_cat_path in attack_cat_list: # Iterate through all files in attack_cat_dir
        if attack_cat_path.stem == NORMAL_FILE_STEM: continue # Skip the Normal file itself

        attack_data = get_data_from_csv(attack_cat_path)
        attack_data = pd.DataFrame(attack_data)
        attack_samples = attack_data.shape[0]

        if not replacing_mode and normal_samples < attack_samples * normal_attack_ratio:
            current_attack_data = attack_data.sample(n=int(normal_samples / normal_attack_ratio), replace=False, random_state=42)
            current_normal_data = normal_data
        else:
            current_attack_data = attack_data
            current_normal_data = normal_data.sample(n=int(attack_samples * normal_attack_ratio), replace=replacing_mode, random_state=42)

        df = pd.concat([current_normal_data, current_attack_data])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        create_csv_from_data(df, f"Normal_{attack_cat_path.stem}", normal_attack_dir_path)

    print("Normal and attack samples merged.")

    return normal_attack_dir_path


def merge_attacks(source_path, dest_path):
    """
    Merges all attack-only categories into a single 'Attacks' dataset.

    :param source_path: Path to the directory containing split category CSVs.
    :param dest_path: Destination path for the merged file.
    :internal dfs: List used to collect DataFrames from individual attack files.
    :return: The Path object to the generated 'Attacks.csv' file.
    """
    attack_cat_list = source_path.iterdir()

    dfs = []

    for attack_cat_path in attack_cat_list:
        if attack_cat_path.stem == 'Normal': continue
        attack_data = get_data_from_csv(attack_cat_path)
        dfs.append(pd.DataFrame(attack_data))

    if not dfs:
        print("Error: No attack categories found to merge.")
        sys.exit(1)

    df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Attack categories merged into a single file.")
    return create_csv_from_data(df, ATTACKS_FILE_STEM, dest_path)


def create_scaled_dataset(source_path, dest_path, ratio, replacing_mode):
    """
    Creates a balanced NB15 dataset with equi-probability for all attack categories.
    Each attack category contributes the same number of samples to the attack pool.

    :param source_path: Path to the directory containing split category CSVs.
    :param dest_path: Destination path for the scaled file.
    :param ratio: The numeric ratio (Normal/Attack) for balancing.
    :param replacing_mode: Boolean indicating if sampling should use replacement.
    """
    attack_cat_list = list(source_path.iterdir()) # Get all files in the attack_cat directory
    normal_path = next((p for p in attack_cat_list if p.stem == NORMAL_FILE_STEM), None)
    if not normal_path:
        print("Error: Normal category not found for scaled dataset.")
        return

    normal_df = get_data_from_csv(normal_path)
    attack_paths = [p for p in attack_cat_list if p.stem != 'Normal']
    num_cats = len(attack_paths)
    
    # Load attack data frames
    attack_dfs = [get_data_from_csv(p) for p in attack_paths]
    
    if not replacing_mode:
        # Find the smallest attack category to ensure equi-probability without replacement
        min_atk_samples = min(len(df) for df in attack_dfs)
        if len(normal_df) < (min_atk_samples * num_cats) * ratio:
            # Normal data is the bottleneck
            atk_per_cat = int(len(normal_df) / ratio) // num_cats
            normal_to_sample = int(atk_per_cat * num_cats * ratio)
        else:
            # Attack categories are the bottleneck
            atk_per_cat = min_atk_samples
            normal_to_sample = int(atk_per_cat * num_cats * ratio)
    else:
        # With replacement, use the available Normal data as the primary baseline
        atk_per_cat = int(len(normal_df) / ratio) // num_cats
        normal_to_sample = len(normal_df)

    final_dfs = [normal_df.sample(n=normal_to_sample, replace=replacing_mode, random_state=42)]
    for df in attack_dfs:
        final_dfs.append(df.sample(n=atk_per_cat, replace=replacing_mode, random_state=42))

    df_final = pd.concat(final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    create_csv_from_data(df_final, NB15_SCALED_FILE_STEM, dest_path)
    print("Equi-probable scaled dataset 'nb15_preprocessed_scaled' created.")


def create_joint_datasets(base_dest_dir, ratio, replacing_mode):
    """
    Creates joint datasets combining NB15 Normal samples with SAT20 and TER20 attacks.
    Maintains a 10:1 Normal-to-Attack ratio with equi-probability for attack categories.

    :param base_dest_dir: The base directory (independent or dependent).
    :param ratio: The numeric ratio (Normal/Attack).
    :param replacing_mode: Boolean for sampling with replacement.
    """
    print("\n--- Starting Joint Preprocessing Phase ---")
    joint_dir_path = create_directory(JOINT_DIR_NAME, base_dest_dir)
    
    nb15_normal_path = base_dest_dir / NB15_PREPROCESSED_DIR_NAME / ATTACK_CAT_DIR_NAME / f"{NORMAL_FILE_STEM}.csv"
    if not nb15_normal_path.exists():
        print(f"Error: NB15 Normal file not found at {nb15_normal_path}. Skipping joint phase.")
        return

    normal_df = get_data_from_csv(nb15_normal_path)
    
    for ds_type, joint_file_stem in [('sat20', JOINT_NORMAL_SAT20_FILE_STEM), ('ter20', JOINT_NORMAL_TER20_FILE_STEM)]:
        atk_source_dir = base_dest_dir / f"{ds_type}{PREPROCESSED_DIR_SUFFIX}" / ATTACK_CAT_DIR_NAME
        if not atk_source_dir.exists():
            print(f"Warning: Attack source for {ds_type} not found at {atk_source_dir}. Skipping.")
            continue

        # Filter out 'Normal' if it exists in SAT20/TER20 to get only attacks
        attack_files = [p for p in atk_source_dir.iterdir() if p.stem != NORMAL_FILE_STEM and p.suffix == '.csv']
        if not attack_files:
            continue

        num_cats = len(attack_files)
        total_atk_needed = int(len(normal_df) / ratio)
        atk_per_cat = total_atk_needed // num_cats

        attack_dfs = []
        for p in attack_files:
            df = get_data_from_csv(p)
            attack_dfs.append(df.sample(n=atk_per_cat, replace=replacing_mode, random_state=42))

        df_joint = pd.concat([normal_df] + attack_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        create_csv_from_data(df_joint, joint_file_stem, joint_dir_path)


def file_preprocessing(data, dataset_type, base_dest_dir, normal_attack_ratio=None, replacing_mode=None):
    """
    Orchestrates the file-level preprocessing including splitting, merging attacks, and balancing classes.

    :param data: The preprocessed pandas DataFrame.
    :param dataset_type: The type of dataset being processed (e.g., 'nb15').
    :param base_dest_dir: The base directory where to save the preprocessed data.
    :param normal_attack_ratio: Optional numeric ratio (Normal/Attack) for balancing (NB15 only).
    :param replacing_mode: Optional boolean for sampling with replacement (NB15 only).
    :internal project_root: Resolves the project root path based on file location.
    """
    if dataset_type == 'nb15':
        if normal_attack_ratio is None:
            ratio_prompt = "Insert the normal-attack ratio: [> 0] (e.g. 10, 10 normal data for 1 attack data) "
            normal_attack_ratio = get_numeric_input(ratio_prompt, type_func=float, min_val=0)
        if replacing_mode is None:
            replacing_mode = get_y_n_bool("Do you want to use replacing mode? [y/n] ")
    
    print(f"Running file-level preprocessing for {dataset_type}...")

    main_dir_path = create_directory(f'{dataset_type}{PREPROCESSED_DIR_SUFFIX}', base_dest_dir)

    create_csv_from_data(data, f'{dataset_type}{PREPROCESSED_MAIN_FILE_SUFFIX}', main_dir_path)

    attack_cat_dir_path = split_by_attack_cat(data, main_dir_path)

    if dataset_type == 'nb15':
        merge_attacks(attack_cat_dir_path, main_dir_path)
        merge_normal_attack(attack_cat_dir_path, main_dir_path, normal_attack_ratio, replacing_mode)
        create_scaled_dataset(attack_cat_dir_path, main_dir_path, normal_attack_ratio, replacing_mode)

    print(f"File-level preprocessing for {dataset_type} done.")