from utils.file_utils import *
import pathlib
import pandas as pd
import sys


def split_by_attack_cat(data, dest_path):
    print("Splitting the dataset by attack category...")

    attack_cat_dir_path = create_directory('attack_cat', dest_path)
    if 'attack_cat' not in data.columns:
        print("Error: 'attack_cat' column not found during split.")
        sys.exit(1)
        
    attack_cat_list = data['attack_cat'].unique()

    for attack_cat in attack_cat_list:
        attack_cat_data = data[data['attack_cat'] == attack_cat]
        create_csv_from_data(attack_cat_data, attack_cat, attack_cat_dir_path)

    print("Splitting the dataset by attack category completed successfully!")
    
    return attack_cat_dir_path


def merge_normal_attack(source_path, dest_path, normal_attack_ratio, replacing_mode=False):
    print("Merging normal and attack data...")
    
    normal_attack_dir_path = create_directory('normal_attack', dest_path)
    # Convertiamo in lista per poter iterare più volte se necessario
    attack_cat_list = list(source_path.iterdir())

    normal_cat_path = next((attack_cat for attack_cat in attack_cat_list if attack_cat.stem == 'Normal'), None)
    if normal_cat_path is None:
        print("Error: Normal category not found!")
        return normal_attack_dir_path
    
    normal_data = pd.DataFrame(get_data_from_csv(normal_cat_path))
    normal_samples = normal_data.shape[0]

    for attack_cat_path in attack_cat_list:
        if attack_cat_path.stem == 'Normal': continue

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

    print("Merging normal and attack data completed successfully!")

    return normal_attack_dir_path


def merge_attacks(source_path, dest_path):
    print("Merging attacks data...")

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

    print("Merging attacks data completed successfully!")

    return create_csv_from_data(df, "Attacks", dest_path)


def file_preprocessing(data, dataset_type):
    if dataset_type == 'nb15':
        print(f"Settings for {dataset_type}...")
        
        try:
            normal_attack_ratio = float(input("Insert the normal-attack ratio: [> 0] (e.g. 10, 10 normal data for 1 attack data) "))
            if normal_attack_ratio <= 0:
                print("Error: The ratio must be greater than 0.")
                sys.exit(1)
        except ValueError:
            print("Error: Invalid input. Please enter a numeric value for the ratio.")
            sys.exit(1)

        replacing_input = input("Do you want to use replacing mode? [y/n] ").lower()
        if replacing_input not in ['y', 'n']:
            print("Error: Invalid input. Please enter 'y' or 'n'.")
            sys.exit(1)
        replacing_mode = replacing_input == 'y'

        print(f"Settings for {dataset_type} saved successfully!")
    
    print(f"File preprocessing for {dataset_type} started...")

    # Definisce la root del progetto partendo dalla posizione di questo file
    # src/preprocessing/file_preprocessing.py -> .parent (preprocessing) -> .parent (src) -> .parent (root)
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    main_dir_path = create_directory(f'{dataset_type}_preprocessed', project_root / 'data')

    create_csv_from_data(data, f'{dataset_type}_preprocessed', main_dir_path)

    attack_cat_dir_path = split_by_attack_cat(data, main_dir_path)

    if dataset_type == 'nb15':
        merge_attacks(attack_cat_dir_path, main_dir_path)
        merge_normal_attack(attack_cat_dir_path, main_dir_path, normal_attack_ratio, replacing_mode)

    print(f"File preprocessing for {dataset_type} completed successfully!")