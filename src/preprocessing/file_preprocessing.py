from utils.file_utils import *
import pandas as pd





"""

(main_dir_path,
    attack_cat_dir_path, 
    attack_cat_train_dir_path, 
    attack_cat_test_dir_path,
    normal_attack_dir_path, 
    normal_attack_train_dir_path, 
    normal_attack_test_dir_path) = create_directories_for_nb15()

    
    data_preprocessed_path = create_csv_from_data(data_preprocessed, 'nb15_preprocessed', main_dir_path)
    training_data_path, testing_data_path = split_csv_training_testing_data(data_preprocessed_path, main_dir_path, train_ratio)

    attack_cat_train_list_path = split_csv_by_attack_cat(training_data_path, attack_cat_train_dir_path)
    attack_cat_test_list_path = split_csv_by_attack_cat(testing_data_path, attack_cat_test_dir_path, normal_ratio, attack_ratio, replacing_mode)
    
    normal_attack_train_list_path = create_csv_normal_attack(attack_cat_train_list_path, normal_attack_train_dir_path)
    normal_attack_test_list_path = create_csv_normal_attack(attack_cat_test_list_path, normal_attack_test_dir_path, normal_ratio, attack_ratio, replacing_mode)


    pass
"""





"""match data_type:
        case 'nb15':
            print("\nFor Random Forest [RF] classification")
            print(30*'-')

            print("Insert the train ratio: [0-1]")
            train_ratio = float(input())

            print("Insert the normal-attack ratio: [0-100] (e.g. 10 1)")
            normal_ratio, attack_ratio = input().split()

            print("Do you want to use replacing mode? [y/n]")
            replacing_mode = [True if input() == 'y' else False]

            file_preprocessing_rf(data_preprocessed, train_ratio, normal_ratio, attack_ratio, replacing_mode)

            print("\nFor Isolation Forest [IF] classification")
            print(30*'-')

            print("Insert the train ratio: [0-1]")
            train_ratio = float(input())

            file_preprocessing_if(data_preprocessed, train_ratio)

        case 'sat20':
            pass
        case 'ter20':
            pass
        case _:
            print("Invalid type!")
    pass
"""



"""
def split_csv_by_attack_cat(data_path, dir_path):

    data = get_data_from_csv(data_path)
    path_list = []

    for attack_cat in data['attack_cat'].unique():
        attack_cat_data = data[data['attack_cat'] == attack_cat]
        path_list.append(
            create_csv_from_data(
                data=attack_cat_data, 
                file_name=attack_cat, 
                file_path=dir_path
            )
        )

    return path_list


def create_csv_normal_attack(list_path, dir_path, normal_ratio, attack_ratio, replacing_mode):

    normal_data_path = None
    attack_data_list_path = []
    normal_attack_list_path = []
    
    for path in list_path:
        if path.name == 'normal':
            normal_data_path = path
        else:
            attack_data_list_path.append(path)
        
    normal_data = get_data_from_csv(normal_data_path)

    for attack_data_path in attack_data_list_path:
        attack_data = get_data_from_csv(attack_data_path)
        normal_cat = pd.concat([normal_data, attack_data])
        df = normal_cat.sample(frac=1)

        normal_attack_list_path.append(
            create_csv_from_data(
                data=df, 
                file_name='{normal_data_path.name}_{attack_data_path.name}', 
                file_path=dir_path
            )
        )

    return normal_attack_list_path


def create_directories_for_nb15():
    main_dir_path = create_directory('nb15', pathlib.Path.cwd()/'data')

    attack_cat_dir_path = create_directory('attack_cat', main_dir_path)
    attack_cat_train_dir_path = create_directory('attack_cat_train', attack_cat_dir_path)
    attack_cat_test_dir_path = create_directory('attack_cat_test', attack_cat_dir_path)
    
    normal_attack_dir_path = create_directory('normal_attack', main_dir_path)
    normal_attack_train_dir_path = create_directory('normal_attack_train', normal_attack_dir_path)
    normal_attack_test_dir_path = create_directory('normal_attack_test', normal_attack_dir_path)

    return (
        main_dir_path,
            attack_cat_dir_path, 
                attack_cat_train_dir_path, 
                attack_cat_test_dir_path,
            normal_attack_dir_path, 
                normal_attack_train_dir_path, 
                normal_attack_test_dir_path
    )


def split_csv_training_testing_data(data_path, dir_path, train_ratio):
    data = get_data_from_csv(data_path)
    df = pd.DataFrame(data)
    training_data = df.sample(frac=train_ratio)
    testing_data = df.drop(training_data.index)

    training_data_path = create_csv_from_data(training_data, 'training_data', dir_path)
    testing_data_path = create_csv_from_data(testing_data, 'testing_data', dir_path)

    return training_data_path, testing_data_path
"""


def split_by_attack_cat(data, dest_path):
    attack_cat_dir_path = create_directory('attack_cat', dest_path)

    # TODO

    return attack_cat_dir_path


def merge_normal_attack(source_path, dest_path):
    normal_attack_dir_path = create_directory('normal_attack', dest_path)

    # TODO

    return normal_attack_dir_path


def merge_attacks(source_path, dest_path):
    pass




def file_preprocessing(data, type):

    main_dir_path = create_directory(f'{type}_preprocessed', pathlib.Path.cwd()/'data')
    create_csv_from_data(data, f'{type}_preprocessed', main_dir_path)

    attack_cat_dir_path = split_by_attack_cat(data, main_dir_path)

    if type == 'nb15':
        merge_attacks(attack_cat_dir_path, main_dir_path)
        merge_normal_attack(attack_cat_dir_path, main_dir_path)