import pandas as pd
import sys

from utils.file_utils import create_directory, create_csv_from_data, get_data_from_csv
from utils.input_utils import get_numeric_input, get_y_n_bool
from utils.paths import (
    CLASS_DIR_NAME, DATA_DIR, ATTACK_CAT_DIR_NAME, NB15_PREFIX, NORMAL_ANOMALY_DIR_NAME,
    NORMAL_ATTACK_DIR_NAME, JOINT_DIR_NAME,
    NORMAL_FILE_STEM, NB15_ATTACKS_SCALED_FILE_STEM, PREPROCESSED_SUFFIX,
    SAT20_ATTACKS_SCALED_FILE_STEM, STIN_PREFIX, TER20_ATTACKS_SCALED_FILE_STEM,
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













def split_by_class(data_prep, dataset_type, class_dir):
    if 'class' not in data_prep.columns:
        print(f"Error: 'class' column not found during split. Skipping {dataset_type}.")
        return

    for c in data_prep['class'].unique():
        class_data = data_prep[data_prep['class'] == c].copy()
        create_csv_from_data(class_data, c, class_dir)

    print("Dataset split by class.")







def merge_normal_anomaly(dataset_type, src_dir, dst_dir, normal_anomaly_ratio):
    class_list = list(src_dir.iterdir())
    
    normal_path = next((c for c in class_list if c.stem == 'Normal'), None)
    if normal_path is None:
        print(f"Error: Normal class not found! Skipping {dataset_type}.")
        return
    
    normal_data = pd.DataFrame(get_data_from_csv(normal_path))
    normal_samples = normal_data.shape[0]

    for c in class_list:
        if c.stem == "Normal": continue

        anomaly_data = pd.DataFrame(get_data_from_csv(c))
        anomaly_samples = anomaly_data.shape[0]

        if normal_samples < anomaly_samples * normal_anomaly_ratio:
            n_anomaly_target = int(normal_samples / normal_anomaly_ratio)
            curr_anomaly_data = safe_stratified_sample(anomaly_data, n_anomaly_target, replacing_mode=False)
            curr_normal_data = normal_data
        else:
            n_normal_target = int(anomaly_samples * normal_anomaly_ratio)
            curr_normal_data = safe_stratified_sample(normal_data, n_normal_target, replacing_mode=False)
            curr_anomaly_data = anomaly_data

        df = pd.concat([curr_normal_data, curr_anomaly_data])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        create_csv_from_data(df, f"Normal_{c.stem}", dst_dir)

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









def single_dataset_file_preprocessing(data_prep, dataset_type):
    """ Create the main directories and files of the dataset preprocessed

        v {dataset_type}_prep
            {dataset_type}_prep.csv or {dataset_type}_prep_aggr.csv
            {dataset_type}_prep_aggr_scaled.csv or None
            > class
            > normal_anomaly or None
    """

    print(f"Running file-level preprocessing for {dataset_type}...")

    dataset_prep_dir = create_directory(f"{dataset_type}{PREPROCESSED_SUFFIX}", DATA_DIR)
    class_dir = create_directory(CLASS_DIR_NAME, dataset_prep_dir)

    create_csv_from_data(data_prep, f'{dataset_type}{PREPROCESSED_SUFFIX}', dataset_prep_dir)
    split_by_class(data_prep, dataset_type, class_dir)

    if dataset_type == NB15_PREFIX:
        normal_anomaly_dir = create_directory(NORMAL_ANOMALY_DIR_NAME, dataset_prep_dir)
        merge_normal_anomaly(dataset_type, class_dir, normal_anomaly_dir, 0.8)

    

        

    




    
    merge_attacks_scaled(attack_cat_dir_path, dataset_prep_dir, dataset_type)

    if dataset_type == 'nb15':
        
        create_scaled_dataset(attack_cat_dir_path, dataset_prep_dir, normal_attack_ratio, replacing_mode)

    print(f"File-level preprocessing for {dataset_type} done.")