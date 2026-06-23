import pandas as pd
import sys

from utils.file_utils import create_directory, create_csv_from_data, get_data_from_csv
from utils.input_utils import get_numeric_input, get_y_n_bool
from utils.paths import (
    DATA_DIR, ATTACK_CAT_DIR_NAME,
    NORMAL_ATTACK_DIR_NAME, JOINT_DIR_NAME,
    NORMAL_FILE_STEM, NB15_ATTACKS_SCALED_FILE_STEM,
    SAT20_ATTACKS_SCALED_FILE_STEM, TER20_ATTACKS_SCALED_FILE_STEM,
    NB15_SCALED_FILE_STEM, NB15_PREPROCESSED_DIR_NAME,
    JOINT_NORMAL_SAT20_FILE_STEM, JOINT_NORMAL_TER20_FILE_STEM,
    PREPROCESSED_DIR_SUFFIX, PREPROCESSED_MAIN_FILE_SUFFIX
)


def safe_stratified_sample(df, n_samples, replacing_mode, random_state=42):
    """
    Estrae 'n_samples' dal DataFrame preservando rigorosamente la proporzione 
    della colonna 'split_type' (Train/Test) per evitare Data Leakage.
    """
    if 'split_type' not in df.columns:
        # Fallback di sicurezza se la colonna non esiste
        return df.sample(n=n_samples, replace=replacing_mode, random_state=random_state)
    
    train_df = df[df['split_type'] == 'train']
    test_df = df[df['split_type'] == 'test']
    
    total_len = len(df)
    if total_len == 0:
        return df
        
    # Calcola le proporzioni reali nel file sorgente
    train_ratio = len(train_df) / total_len
    
    # Calcola il numero esatto di campioni da estrarre per ogni split
    n_train = int(n_samples * train_ratio)
    n_test = n_samples - n_train  # La differenza garantisce che il totale sia esattamente n_samples
    
    sampled_parts = []
    
    if n_train > 0 and not train_df.empty:
        sampled_parts.append(train_df.sample(n=n_train, replace=replacing_mode, random_state=random_state))
    if n_test > 0 and not test_df.empty:
        sampled_parts.append(test_df.sample(n=n_test, replace=replacing_mode, random_state=random_state))
        
    if not sampled_parts:
        return pd.DataFrame(columns=df.columns)
        
    # Concatena le due estrazioni e mescola l'ordine delle righe
    return pd.concat(sampled_parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def split_by_attack_cat(data, dest_path):
    attack_cat_dir_path = create_directory(ATTACK_CAT_DIR_NAME, dest_path)
    if 'attack_cat' not in data.columns:
        print("Error: 'attack_cat' column not found during split.")
        sys.exit(1)
        
    attack_cat_list = data['attack_cat'].unique()

    for attack_cat in attack_cat_list:
        attack_cat_data = data[data['attack_cat'] == attack_cat].copy()
        create_csv_from_data(attack_cat_data, attack_cat, attack_cat_dir_path)
    print("Dataset split by attack category.")
    
    return attack_cat_dir_path


def merge_normal_attack(source_path, dest_path, normal_attack_ratio, replacing_mode):
    normal_attack_dir_path = create_directory(NORMAL_ATTACK_DIR_NAME, dest_path)
    attack_cat_list = list(source_path.iterdir())
    
    normal_cat_path = next((attack_cat for attack_cat in attack_cat_list if attack_cat.stem == NORMAL_FILE_STEM), None)
    if normal_cat_path is None:
        print("Error: Normal category not found!")
        return normal_attack_dir_path
    
    normal_data = pd.DataFrame(get_data_from_csv(normal_cat_path))
    normal_samples = normal_data.shape[0]

    for attack_cat_path in attack_cat_list:
        if attack_cat_path.stem == NORMAL_FILE_STEM: continue

        attack_data = pd.DataFrame(get_data_from_csv(attack_cat_path))
        attack_samples = attack_data.shape[0]

        if not replacing_mode and normal_samples < attack_samples * normal_attack_ratio:
            n_attack_target = int(normal_samples / normal_attack_ratio)
            current_attack_data = safe_stratified_sample(attack_data, n_attack_target, replacing_mode=False)
            current_normal_data = normal_data
        else:
            current_attack_data = attack_data
            n_normal_target = int(attack_samples * normal_attack_ratio)
            current_normal_data = safe_stratified_sample(normal_data, n_normal_target, replacing_mode=replacing_mode)

        df = pd.concat([current_normal_data, current_attack_data])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        create_csv_from_data(df, f"Normal_{attack_cat_path.stem}", normal_attack_dir_path)

    print("Normal and attack samples merged.")
    return normal_attack_dir_path


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
        sampled_train = df_train.sample(n=min_train, random_state=42)
        sampled_test = df_test.sample(n=min_test, random_state=42)
        
        # 4. Ricombina i due split bilanciati per la classe corrente
        balanced_class_df = pd.concat([sampled_train, sampled_test])
        balanced_dfs.append(balanced_class_df)

    # 5. Concatena tutte le classi bilanciate e rimescola le righe
    df_final = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
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

    df_final = pd.concat(final_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
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

        df_joint = pd.concat([normal_df] + attack_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        create_csv_from_data(df_joint, joint_file_stem, joint_dir_path)

        sub_dir_path = create_directory(f"{ds_type}_joint", joint_dir_path)
        
        for p in attack_files:
            df = get_data_from_csv(p)
            sampled_atk = safe_stratified_sample(df, total_atk_needed, replacing_mode)
            df_single_joint = pd.concat([normal_df, sampled_atk]).sample(frac=1, random_state=42).reset_index(drop=True)
            create_csv_from_data(df_single_joint, f"Normal_{p.stem}", sub_dir_path)


def file_preprocessing(data, dataset_type, base_dest_dir, normal_attack_ratio=None, replacing_mode=None):
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
    merge_attacks_scaled(attack_cat_dir_path, main_dir_path, dataset_type)

    if dataset_type == 'nb15':
        merge_normal_attack(attack_cat_dir_path, main_dir_path, normal_attack_ratio, replacing_mode)
        create_scaled_dataset(attack_cat_dir_path, main_dir_path, normal_attack_ratio, replacing_mode)

    print(f"File-level preprocessing for {dataset_type} done.")