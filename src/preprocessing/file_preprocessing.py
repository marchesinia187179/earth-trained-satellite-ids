import pandas as pd

from utils.file_utils import concat_and_shuffle, create_directory, create_csv_from_data
from utils.paths import (
    CLASS_DIR_NAME, NORMAL_ANOMALY_DIR_NAME, DATA_DIR, 
    NB15_PREFIX, NB15_STIN_PREFIX, NB15_SAT20_PREFIX, NB15_TER20_PREFIX, 
    PREPROCESSED_SCALED_SUFFIX, PREPROCESSED_SUFFIX, 
    RANDOM_STATE, NORMAL_ANOMALY_RATIO
)


def _get_correct_normal_and_anomaly_data_samples(normal_data, anomaly_data, normal_samples, anomaly_samples):
    """
    Calculates and return the `normal` and `anomaly data sample` based on the `NORMAL_ANOMALY_RATIO`
    
    :param normal_data: normal dataset
    :param anomaly_data: anomaly dataset
    :param normal_samples: number of data in normal dataset
    :param anomaly_samples: number of data in anomaly dataset
    :return: correct `normal` and `anomaly data sample`
    """
    # Calculate and save the normal and anomaly data sample
    if normal_samples < anomaly_samples * NORMAL_ANOMALY_RATIO:
        n_anomaly_target = int(normal_samples / NORMAL_ANOMALY_RATIO)
        anomaly_data_sample = _safe_stratified_sample(anomaly_data, n_anomaly_target)
        normal_data_sample = normal_data
    else:
        n_normal_target = int(anomaly_samples * NORMAL_ANOMALY_RATIO)
        normal_data_sample = _safe_stratified_sample(normal_data, n_normal_target)
        anomaly_data_sample = anomaly_data

    return normal_data_sample, anomaly_data_sample


def _safe_stratified_sample(data, n_samples):
    """
    Takes `n_samples` from `data` preserving the `split_type ratio` to avoid the Data Leakage.

    :param data: pool of data to get the correct data sample
    :param n_samples: number of samples
    :return: correct `data` sample
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
        
    # Concatenate the two samples and shuffle them
    return concat_and_shuffle(sampled_parts)


def _split_by_class_and_save(data, type, dst_dir):
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


def _merge_normal_anomaly_and_save(data, type, dst_dir):
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
        normal_data_sample, anomaly_data_sample = _get_correct_normal_and_anomaly_data_samples(
            normal_data, anomaly_data, normal_samples, anomaly_samples)

        # Save the dataset
        df = concat_and_shuffle([normal_data_sample, anomaly_data_sample])
        create_csv_from_data(df, f"Normal_{c}", dst_dir)

    print("Normal and anomaly samples merged.")


def _scale_by_normal_anomaly_ratio_and_save(data, type, dst_dir):
    """
    Updates the `data` with a new data scaled by the `normal_anomaly_ratio` and
    creates a new csv file
    
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
    
    # Get the Normal and Anomaly data
    normal_data = data[data['class'] == "Normal"]
    anomaly_data = data[data['label'] == 1]
    
    normal_samples = normal_data.shape[0]
    anomaly_samples = anomaly_data.shape[0]

    # Get the correct data samples based on the ratio given
    normal_data_sample, anomaly_data_sample = _get_correct_normal_and_anomaly_data_samples(
            normal_data, anomaly_data, normal_samples, anomaly_samples)

    # Save the dataset
    df = concat_and_shuffle([normal_data_sample, anomaly_data_sample])
    create_csv_from_data(df, f"{type}{PREPROCESSED_SCALED_SUFFIX}", dst_dir)


def hybrid_dataset_file_preprocessing(nb15_normal_data, sat20_anomaly_data, ter20_anomaly_data):
    """
    Creates the main directories and files of the hybrid dataset

    :param nb15_normal_data: normal data from nb15 dataset
    :param sat20_anomaly_data: anomaly data from sat20 dataset
    :param ter20_anomaly_data: anomaly data from ter20 dataset
    :return: None
    """
    print(f"Running file-level preprocessing for hybrid datasets...")
    # Security check if the nb15 dataset contains only Normal data and the others one (sat20, ter20) only Anomaly data
    if not nb15_normal_data[nb15_normal_data['label'] == 1].empty:
        print(f"Error: the nb15 dataset passed must be only Normal data. \
            You passed {len(nb15_normal_data[nb15_normal_data['label'] == 1])} Anomaly data!")
        return
    
    for i, anomaly_data in enumerate([sat20_anomaly_data, ter20_anomaly_data]):
        if not anomaly_data[anomaly_data['label'] == 0].empty:
            print(f"Error: the dataset number {i+1} passed must be only Anomaly data. \
                You passed {len(anomaly_data[anomaly_data['label'] == 0])} Normal data!")
            return

    # Create the main directories
    dataset_prep_dir = create_directory(f"{NB15_STIN_PREFIX}{PREPROCESSED_SUFFIX}", DATA_DIR)
    nb15_sat20_normal_anomaly_dir = create_directory(f"{NB15_SAT20_PREFIX}_{NORMAL_ANOMALY_DIR_NAME}", dataset_prep_dir)
    nb15_ter20_normal_anomaly_dir = create_directory(f"{NB15_TER20_PREFIX}_{NORMAL_ANOMALY_DIR_NAME}", dataset_prep_dir)

    # Create the hybrid data
    nb15_stin_data = concat_and_shuffle([nb15_normal_data, sat20_anomaly_data, ter20_anomaly_data])
    nb15_sat20_data = concat_and_shuffle([nb15_normal_data, sat20_anomaly_data])
    nb15_ter20_data = concat_and_shuffle([nb15_normal_data, ter20_anomaly_data])

    # Combine the hybrid data with own dataset type
    datasets = [
        {'type': NB15_STIN_PREFIX, 'data': nb15_stin_data},
        {'type': NB15_SAT20_PREFIX, 'data': nb15_sat20_data},
        {'type': NB15_TER20_PREFIX, 'data': nb15_ter20_data}
    ]

    # Save the hybrid data
    for d in datasets:
        dataset_type = d['type']
        dataset_data = d['data']

        create_csv_from_data(dataset_data, f'{dataset_type}{PREPROCESSED_SUFFIX}', dataset_prep_dir)
        _scale_by_normal_anomaly_ratio_and_save(dataset_data, dataset_type, dataset_prep_dir)

        # Save the hybrid data for single normal_anomaly case
        if dataset_type == NB15_SAT20_PREFIX:
            _merge_normal_anomaly_and_save(dataset_data, dataset_type, nb15_sat20_normal_anomaly_dir)
        elif dataset_type == NB15_TER20_PREFIX:
            _merge_normal_anomaly_and_save(dataset_data, dataset_type, nb15_ter20_normal_anomaly_dir)

    print(f"File-level preprocessing for hybrid datasets done.")


def single_dataset_file_preprocessing(data, type):
    """
    Creates the main directories and files of the data

    :param data: data preprocessed
    :param type: name of the dataset type
    """
    print(f"Running file-level preprocessing for {type}...")

    # Create the main directories
    dataset_prep_dir = create_directory(f"{type}{PREPROCESSED_SUFFIX}", DATA_DIR)
    class_dir = create_directory(CLASS_DIR_NAME, dataset_prep_dir)

    # Save the data
    create_csv_from_data(data, f'{type}{PREPROCESSED_SUFFIX}', dataset_prep_dir)
    _split_by_class_and_save(data, type, class_dir)

    # Save the normal_anomaly data
    if type == NB15_PREFIX:
        _scale_by_normal_anomaly_ratio_and_save(data, type, dataset_prep_dir)
        normal_anomaly_dir = create_directory(NORMAL_ANOMALY_DIR_NAME, dataset_prep_dir)
        _merge_normal_anomaly_and_save(data, type, normal_anomaly_dir)

    print(f"File-level preprocessing for {type} done.")


if __name__ == "__main__":
    # TODO
    pass