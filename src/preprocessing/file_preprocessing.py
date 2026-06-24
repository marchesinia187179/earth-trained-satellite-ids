import pandas as pd
import sys

from utils.file_utils import create_directory, create_csv_from_data, get_data_from_csv
from utils.paths import (
    CLASS_DIR_NAME, DATA_DIR, NB15_PREFIX, NORMAL_ANOMALY_DIR_NAME, PREPROCESSED_SUFFIX, RANDOM_STATE, STIN_PREFIX
)


def safe_stratified_sample(data, n_samples):
    """
    Takes `n_samples` from `data` preserving the `split_type ratio` to avoid the Data Leakage.

    :param data: pool of data to get the correct data sample
    :param n_samples: number of samples
    :param random_state: status of the `random_state` param for the sample
    :return: the correct sample of `data`
    """
    # Security Fallback if the split type colums doesn't exist
    if 'split_type' not in data.columns:
        return data.sample(n=n_samples, random_state=RANDOM_STATE)
    
    # Get the train and test data
    data_train = data[data['split_type'] == 'train']
    data_test = data[data['split_type'] == 'test']
    
    total_len = len(data)
    if total_len == 0:
        return data
    
    # Calculate the real proportions of the data
    train_ratio = len(data_train) / total_len
    
    # Calculate the exact number of samples to get from each split
    n_train = int(n_samples * train_ratio)
    n_test = n_samples - n_train
    
    # Get the correct train and test sample
    sampled_parts = []
    
    if n_train > 0 and not data_train.empty:
        sampled_parts.append(data_train.sample(n=n_train, random_state=RANDOM_STATE))
    if n_test > 0 and not data_test.empty:
        sampled_parts.append(data_test.sample(n=n_test, random_state=RANDOM_STATE))
        
    if not sampled_parts:
        return pd.DataFrame(columns=data.columns)
        
    # Concat the two samples and shuffle them
    df = pd.concat(sampled_parts)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return df


def split_by_class(data, type, dst_dir):
    """
    Splits the `data` by class and
    for each `data_c` creates an own csv file into in `dst_dir`
    
    :param data: data to be split by class
    :param type: type of the data
    :param dst_dir: destination directory path
    :return: None
    """
    # Security check if the feature exists
    if 'class' not in data.columns:
        print(f"Error: 'class' column not found during split. Skipping {type}.")
        return

    # Create an own csv file for each class
    for data_c in data['class'].unique():
        class_data = data[data['class'] == data_c].copy()
        create_csv_from_data(class_data, data_c, dst_dir)

    print("Dataset split by class.")


def merge_normal_anomaly(data, type, dst_dir, normal_anomaly_ratio):
    """
    Merges `normal class` with `anomaly class` of `data` and
    for each `normal_anomaly` combo creates a own csv file into in `dst_dir`

    :param data: pool of data to get normal and anomaly data
    :param type: type of the data
    :param dst_dir: destination directory path
    :return: None 
    """
    # Security check if the Normal class exists
    class_list = data['class'].unique()
    if not "Normal" in class_list:
        print(f"Error: Normal class not found! Skipping {type}.")
        return
    
    # Get the Normal data
    normal_data = data[data['class'] == "Normal"]
    normal_samples = normal_data.shape[0]

    for c in class_list:
        if c == "Normal": continue

        # Get the anomaly data
        anomaly_data = data[data['class'] == c]
        anomaly_samples = anomaly_data.shape[0]

        # Get the correct data samples based on the ratio given
        if normal_samples < anomaly_samples * normal_anomaly_ratio:
            n_anomaly_target = int(normal_samples / normal_anomaly_ratio)
            curr_anomaly_data = safe_stratified_sample(anomaly_data, n_anomaly_target)
            curr_normal_data = normal_data
        else:
            n_normal_target = int(anomaly_samples * normal_anomaly_ratio)
            curr_normal_data = safe_stratified_sample(normal_data, n_normal_target)
            curr_anomaly_data = anomaly_data

        df = pd.concat([curr_normal_data, curr_anomaly_data])
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        create_csv_from_data(df, f"Normal_{c}", dst_dir)

    print("Normal and anomaly samples merged.")













def merge_attacks_scaled(source_path, dest_path, dataset_type):
    attack_cat_list = source_path.iterdir()
    dfs = []

    for attack_cat_path in attack_cat_list:
        if attack_cat_path.name.startswith('.') or attack_cat_path.stem == 'Normal': 
            continue
        
        attack_data = get_data_from_csv(attack_cat_path)
        df_cat = pd.DataFrame(attack_data)
        
        if not df_cat.empty:
            dfs.append(df_cat)

    if not dfs:
        print("Error: No attack categories found to merge.")
        sys.exit(1)

    # --- LOGICA DI EQUIPROBABILITÀ SEPARATA PER TRAIN E TEST ---
    # 1. Trova il numero minimo di campioni train e test tra tutte le categorie
    min_train = min(len(df[df['split_type'] == 'train']) for df in dfs)
    min_test = min(len(df[df['split_type'] == 'test']) for df in dfs)
    
    print(f"Balancing split sub-distributions across all attack categories:")
    print(f"  -> Each class will have exactly {min_train} 'train' samples.")
    print(f"  -> Each class will have exactly {min_test} 'test' samples.")

    balanced_dfs = []
    for df in dfs:
        # 2. Isola i due split per la classe corrente
        df_train = df[df['split_type'] == 'train']
        df_test = df[df['split_type'] == 'test']
        
        # 3. Campiona separatamente per garantire l'equiprobabilità interna
        sampled_train = df_train.sample(n=min_train, random_state=RANDOM_STATE)
        sampled_test = df_test.sample(n=min_test, random_state=RANDOM_STATE)
        
        # 4. Ricombina i due split bilanciati per la classe corrente
        balanced_class_df = pd.concat([sampled_train, sampled_test])
        balanced_dfs.append(balanced_class_df)

    # 5. Concatena tutte le classi bilanciate e rimescola le righe
    df_final = pd.concat(balanced_dfs).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Salvataggio del file
    if dataset_type == 'nb15':
        create_csv_from_data(df_final, NB15_ATTACKS_SCALED_FILE_STEM, dest_path)
    elif dataset_type == 'sat20':
        create_csv_from_data(df_final, SAT20_ATTACKS_SCALED_FILE_STEM, dest_path)
    elif dataset_type == 'ter20':
        create_csv_from_data(df_final, TER20_ATTACKS_SCALED_FILE_STEM, dest_path)
    
    print(f"Attack categories merged successfully. Total samples: {len(df_final)} "
          f"(Train per class: {min_train} | Test per class: {min_test})")
    return dest_path













def create_scaled_dataset(source_path, dest_path, ratio, replacing_mode):
    attack_cat_list = list(source_path.iterdir())
    normal_path = next((p for p in attack_cat_list if p.stem == NORMAL_FILE_STEM), None)
    if not normal_path:
        print("Error: Normal category not found for scaled dataset.")
        return

    normal_df = get_data_from_csv(normal_path)
    attack_paths = [p for p in attack_cat_list if p.stem != 'Normal']
    
    # Controllo di sicurezza inserito
    if not attack_paths:
        return
        
    num_cats = len(attack_paths)
    attack_dfs = [get_data_from_csv(p) for p in attack_paths]
    
    if not replacing_mode:
        min_atk_samples = min(len(df) for df in attack_dfs)
        if len(normal_df) < (min_atk_samples * num_cats) * ratio:
            atk_per_cat = int(len(normal_df) / ratio) // num_cats
            normal_to_sample = int(atk_per_cat * num_cats * ratio)
        else:
            atk_per_cat = min_atk_samples
            normal_to_sample = int(atk_per_cat * num_cats * ratio)
    else:
        atk_per_cat = int(len(normal_df) / ratio) // num_cats
        normal_to_sample = len(normal_df)

    final_dfs = [safe_stratified_sample(normal_df, normal_to_sample, replacing_mode)]
    for df in attack_dfs:
        final_dfs.append(safe_stratified_sample(df, atk_per_cat, replacing_mode))

    df_final = pd.concat(final_dfs).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    create_csv_from_data(df_final, NB15_SCALED_FILE_STEM, dest_path)
    print("Equi-probable scaled dataset 'nb15_preprocessed_scaled' created.")


def create_joint_datasets(base_dest_dir, ratio, replacing_mode):
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

        attack_files = [p for p in atk_source_dir.iterdir() if p.stem != NORMAL_FILE_STEM and p.suffix == '.csv']
        if not attack_files:
            continue

        num_cats = len(attack_files)
        total_atk_needed = int(len(normal_df) / ratio)
        atk_per_cat = total_atk_needed // num_cats

        attack_dfs = []
        for p in attack_files:
            df = get_data_from_csv(p)
            attack_dfs.append(safe_stratified_sample(df, atk_per_cat, replacing_mode))

        df_joint = pd.concat([normal_df] + attack_dfs).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        create_csv_from_data(df_joint, joint_file_stem, joint_dir_path)

        sub_dir_path = create_directory(f"{ds_type}_joint", joint_dir_path)
        
        for p in attack_files:
            df = get_data_from_csv(p)
            sampled_atk = safe_stratified_sample(df, total_atk_needed, replacing_mode)
            df_single_joint = pd.concat([normal_df, sampled_atk]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
            create_csv_from_data(df_single_joint, f"Normal_{p.stem}", sub_dir_path)









def single_dataset_file_preprocessing(data, type):
    """
    Creates the main directories and files of the data

    :param data: data preprocessed
    :param type: name of the dataset type
    """
    print(f"Running file-level preprocessing for {type}...")

    dataset_prep_dir = create_directory(f"{type}{PREPROCESSED_SUFFIX}", DATA_DIR)
    class_dir = create_directory(CLASS_DIR_NAME, dataset_prep_dir)

    create_csv_from_data(data, f'{type}{PREPROCESSED_SUFFIX}', dataset_prep_dir)
    split_by_class(data, type, class_dir)

    if type == NB15_PREFIX:
        # TODO: create the nb15_prep_scaled.csv
        # create_scaled_dataset(attack_cat_dir_path, dataset_prep_dir, normal_attack_ratio, replacing_mode)

        normal_anomaly_dir = create_directory(NORMAL_ANOMALY_DIR_NAME, dataset_prep_dir)
        merge_normal_anomaly(data, type, normal_anomaly_dir, 0.8)

    print(f"File-level preprocessing for {type} done.")