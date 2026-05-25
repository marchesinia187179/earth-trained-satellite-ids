from utils.file_utils import *
import pandas as pd


def split_by_attack_cat(data, dest_path):
    print("Splitting the dataset by attack category...")

    attack_cat_dir_path = create_directory('attack_cat', dest_path)
    attack_cat_list = data['attack_cat'].unique()

    for attack_cat in attack_cat_list:
        attack_cat_data = data[data['attack_cat'] == attack_cat]
        create_csv_from_data(attack_cat_data, attack_cat, attack_cat_dir_path)

    print("Splitting the dataset by attack category completed successfully!")
    
    return attack_cat_dir_path


def merge_normal_attack(source_path, dest_path, normal_attack_ratio, replacing_mode=False):
    print("Merging normal and attack data...")
    
    normal_attack_dir_path = create_directory('normal_attack', dest_path)
    attack_cat_list = source_path.iterdir()

    normal_cat_path = next((attack_cat for attack_cat in attack_cat_list if attack_cat.name == 'Normal'), None)
    normal_data = get_data_from_csv(normal_cat_path)
    normal_data = pd.DataFrame(normal_data)
    normal_samples = normal_data.shape[0]

    for attack_cat_path in attack_cat_list:
        if attack_cat_path.name == 'Normal':
            continue

        attack_data = get_data_from_csv(attack_cat_path)
        attack_data = pd.DataFrame(attack_data)
        attack_samples = attack_data.shape[0]

        if replacing_mode == False and normal_samples < attack_samples * normal_attack_ratio:
            attack_data = attack_data.sample(n=(normal_samples / normal_attack_ratio), replace=replacing_mode, random_state=42)
        else:
            normal_data = normal_data.sample(n=(attack_samples * normal_attack_ratio), replace=replacing_mode, random_state=42)

        df = pd.concat([normal_data, attack_data])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        create_csv_from_data(df, f"{normal_data.name}_{attack_data.name}", normal_attack_dir_path)

    print("Merging normal and attack data completed successfully!")

    return normal_attack_dir_path


def merge_attacks(source_path, dest_path):
    print("Merging attacks data...")

    attack_cat_list = source_path.iterdir()

    df = None

    for attack_cat_path in attack_cat_list:
        attack_data = get_data_from_csv(attack_cat_path)
        attack_data = pd.DataFrame(attack_data)
        df = pd.concat([attack_data])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Merging attacks data completed successfully!")
    
    return create_csv_from_data(df, "Attacks", dest_path)


def file_preprocessing(data, type):
    print(f"File preprocessing settings for {type} started")
    
    print(f"Insert the normal-attack ratio: [0-100] (e.g. 10, 10 normal data for 1 attack data)")
    normal_attack_ratio = input()
    
    print("Do you want to use replacing mode? [y/n]")
    replacing_mode = [True if input() == 'y' else False]
    
    print(f"File preprocessing for {type} started...")

    main_dir_path = create_directory(f'{type}_preprocessed', pathlib.Path.cwd()/'data')
    create_csv_from_data(data, f'{type}_preprocessed', main_dir_path)

    attack_cat_dir_path = split_by_attack_cat(data, main_dir_path)

    if type == 'nb15':
        merge_attacks(attack_cat_dir_path, main_dir_path)
        merge_normal_attack(attack_cat_dir_path, main_dir_path, normal_attack_ratio, replacing_mode)

    print(f"File preprocessing for {type} completed successfully!")