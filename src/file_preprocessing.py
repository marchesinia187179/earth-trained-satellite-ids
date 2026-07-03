"""
Preprocessing functions for handling and preparing datasets.
"""
import pandas as pd

from .utils.file_utils import store_file_info, concat_and_shuffle, create_directory, create_csv_from_data, update_or_append_csv
from .utils.config import MLConstants, Naming, ProjectPaths
from .utils.metrics import calculate_mean_and_variance


# --- Internal Functions ---
def _get_correct_normal_and_anomaly_data_samples(normal_data, anomaly_data, normal_samples, anomaly_samples):
    """
    Calculates and return the `normal` and `anomaly data sample` based on the `MLConstants.NORMAL_ANOMALY_RATIO`
    
    :param normal_data: normal dataset
    :param anomaly_data: anomaly dataset
    :param normal_samples: number of data in normal dataset
    :param anomaly_samples: number of data in anomaly dataset
    :return: correct `normal` and `anomaly data sample`
    """
    # Calculate and save the normal and anomaly data sample
    if normal_samples < anomaly_samples * MLConstants.NORMAL_ANOMALY_RATIO:
        n_anomaly_target = int(normal_samples / MLConstants.NORMAL_ANOMALY_RATIO)
        anomaly_data_sample = _safe_stratified_sample(anomaly_data, n_anomaly_target)
        normal_data_sample = normal_data
    else:
        n_normal_target = int(anomaly_samples * MLConstants.NORMAL_ANOMALY_RATIO)
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
        return data.sample(n=n_samples, random_state=MLConstants.RANDOM_STATE)
    
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
        sampled_parts.append(data_train.sample(n=n_train, random_state=MLConstants.RANDOM_STATE))
    if n_test > 0 and not data_test.empty:
        sampled_parts.append(data_test.sample(n=n_test, random_state=MLConstants.RANDOM_STATE))
        
    if not sampled_parts:
        return pd.DataFrame(columns=data.columns)
        
    # Concatenate the two samples and shuffle them
    return concat_and_shuffle(sampled_parts)




def _split_by_class(data, type):
    """
    Splits the `data` by class and return a list of unique data
    
    :param data: data to be split by class
    :param type: type of the data
    :return: list of dataframes split by class
    """
    # Security check if the feature exists
    if 'class' not in data.columns:
        print(f"Error: 'class' column not found during split. Skipping {type}.")
        return
    
    # Split the data by class and save them in a list
    return [data[data['class'] == c].copy() for c in data['class'].unique()]












def _merge_normal_anomaly(data, type, dst_dir):
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

    normal_anomaly_data_list = []
    for c in class_list:
        if c == "Normal": continue

        # Get the anomaly data
        anomaly_data = data[data['class'] == c]
        anomaly_samples = anomaly_data.shape[0]

        # Get the correct data samples based on the ratio given
        normal_data_sample, anomaly_data_sample = _get_correct_normal_and_anomaly_data_samples(
            normal_data, anomaly_data, normal_samples, anomaly_samples)

        # Concat and shuffle the samples
        normal_anomaly_data = concat_and_shuffle([normal_data_sample, anomaly_data_sample])

        normal_anomaly_data_list.append(normal_anomaly_data_list)

    return normal_anomaly_data_list







def _scale_by_normal_anomaly_ratio(data, type):
    """
    Updates the `data` with a new data scaled by the `MLConstants.NORMAL_ANOMALY_RATIO` and
    creates a new csv file
    
    :param data: pool of data to get normal and anomaly data
    :param type: type of the data
    :return: scaled data
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
            normal_data=normal_data, 
            anomaly_data=anomaly_data, 
            normal_samples=normal_samples, 
            anomaly_samples=anomaly_samples
        )

    # Concat and shuffle the samples
    return concat_and_shuffle([normal_data_sample, anomaly_data_sample])












def _get_mean_and_variance(data, dataset_type):
    """
    Calculates the mean and variance of each feature in the dataset and saves them into two separate csv files.
    
    :param data: Input pandas DataFrame containing the dataset
    :param dataset_type: The type of the dataset
    """

    feature_mean_list = []
    feature_variance_list = []

    # Calculate and save the mean and variance for the aggregated dataset
    mean_record, variance_record = calculate_mean_and_variance(data, dataset_type, 'aggregated')

    feature_mean_list.append(mean_record)
    feature_variance_list.append(variance_record)

    # Calculate and save the mean and variance for each class in the dataset
    for class_type in data['class'].unique():
        class_data = data[data['class'] == class_type]
        mean_record, variance_record = calculate_mean_and_variance(class_data, dataset_type, class_type)

        feature_mean_list.append(mean_record)
        feature_variance_list.append(variance_record)

    return feature_mean_list, feature_variance_list





def _save_data_and_store_info(data, file_name, dst_dir, dataset_type=None, store_info=True):
    """
    
    """

    file_path = create_csv_from_data(data, file_name, dst_dir)

    if store_info:
        store_file_info(file_path, dataset_type)







# --- Public Functions ---
def hybrid_dataset_file_preprocessing(nb15_normal_data, sat20_anomaly_data, ter20_anomaly_data):
    """
    Creates the main directories and files of the hybrid dataset

    :param nb15_normal_data: normal data from nb15 dataset
    :param sat20_anomaly_data: anomaly data from sat20 dataset
    :param ter20_anomaly_data: anomaly data from ter20 dataset
    :return: None
    """
    # --- Security check ---
    # Check if the nb15 dataset contains only Normal data
    if not nb15_normal_data[nb15_normal_data['label'] == 1].empty:
        print(f"Error: the nb15 dataset passed must be only Normal data. \
            You passed {len(nb15_normal_data[nb15_normal_data['label'] == 1])} Anomaly data!")
        return
    
    # Check if the sat20 and ter20 dataset contains only Anomaly data
    for i, anomaly_data in enumerate([sat20_anomaly_data, ter20_anomaly_data]):
        if not anomaly_data[anomaly_data['label'] == 0].empty:
            print(f"Error: the dataset number {i+1} passed must be only Anomaly data. \
                You passed {len(anomaly_data[anomaly_data['label'] == 0])} Normal data!")
            return
    
    print(f"Running file-level preprocessing for hybrid datasets...")
    
    # --- Create the cirectories ---

    # Create hybrid directory
    hybrid_dir = create_directory(
        dir_name=Naming.HYBRID, 
        parent_path=ProjectPaths.PREP_DATA_DIR
    )

    # Create nb15_sat20 directory
    nb15_sat20_dir = create_directory(
        dir_name=Naming.NB15_SAT20,
        parent_path=hybrid_dir
    )

    # Create nb15_ter20 directory
    nb15_ter20_dir = create_directory(
        dir_name=Naming.NB15_TER20,
        parent_path=hybrid_dir
    )
    
    # Create normal_anomaly directory for nb15_sat20 and nb15_ter20
    for parent_path in [nb15_sat20_dir, nb15_ter20_dir]:
        nb15_sat20_normal_anomaly_dir = create_directory(
            dir_name=ProjectPaths.DIR_NORMAL_ANOMALY, 
            parent_path=parent_path
        )




    # --- Process the data ---
    # Create the hybrid data
    nb15_stin_data = concat_and_shuffle([nb15_normal_data, sat20_anomaly_data, ter20_anomaly_data])
    nb15_sat20_data = concat_and_shuffle([nb15_normal_data, sat20_anomaly_data])
    nb15_ter20_data = concat_and_shuffle([nb15_normal_data, ter20_anomaly_data])

    # Combine the hybrid data with own dataset type
    datasets = [
        {'type': Naming.HYBRID, 'data': nb15_stin_data},
        {'type': Naming.NB15_SAT20, 'data': nb15_sat20_data},
        {'type': Naming.NB15_TER20, 'data': nb15_ter20_data}
    ]

    # Save the hybrid data
    for d in datasets:
        dataset_type = d['type']
        dataset_data = d['data']

        file_path = create_csv_from_data(dataset_data, f'{dataset_type}{Naming.PREP}', data_prep_dir)
        store_file_info(file_path, dataset_type)
        _scale_by_normal_anomaly_ratio_and_save(dataset_data, dataset_type, scaled_dir)

        # Save the hybrid data for single normal_anomaly case
        if dataset_type == Naming.NB15_SAT20:
            _merge_normal_anomaly_and_save(dataset_data, dataset_type, nb15_sat20_normal_anomaly_dir)
        elif dataset_type == Naming.NB15_TER20:
            _merge_normal_anomaly_and_save(dataset_data, dataset_type, nb15_ter20_normal_anomaly_dir)

        # Calculate and save feature mean and variance
        _get_mean_and_variance_and_save(dataset_data, dataset_type)


    print(f"File-level preprocessing for hybrid datasets done.")








def single_dataset_file_preprocessing(data, dataset_type):
    """
    Creates the main directories and files of a single dataset.

    :param data: dataset to be preprocessed
    :param dataset_type: type of the dataset (nb15, sat20, ter20)
    :return: None
    """
    print(f"Running file-level preprocessing for {dataset_type}...")

    # --- Create the directories ---
    # Create dataset_type directory
    data_prep_dir = create_directory(
        dir_name=dataset_type, 
        parent_path=ProjectPaths.PREP_DATA_DIR
    )
    
    # Create single_classes directory
    single_classes_dir = create_directory(
        dir_name=ProjectPaths.DIR_SINGLE_CLASSES, 
        parent_path=data_prep_dir
    )

    # If nb15 than create normal_anomaly directory
    if dataset_type == Naming.NB15:
        normal_anomaly_dir = create_directory(
            dir_name=ProjectPaths.DIR_NORMAL_ANOMALY, 
            parent_path=data_prep_dir
        )

    # --- Process the data ---
    # Get data splitted by class
    class_data_list = _split_by_class(data, dataset_type)

    # If nb15 than scale preprocessed data by normal anomaly ratio, 
    # merge them into normal anomaly data
    if dataset_type == Naming.NB15:
        data_scaled = _scale_by_normal_anomaly_ratio(data, dataset_type)
        normal_anomaly_data_list = _merge_normal_anomaly(data, dataset_type)

    # Get feature mean and variance
    feature_mean_list, feature_variance_list = _get_mean_and_variance(data, dataset_type)

    # --- Save the preprocessed data ---
    # Save preprocessed data
    _save_data_and_store_info(
        data=data,
        file_name=f'{type}{Naming.AGGR}',
        dst_dir=data_prep_dir,
        dataset_type=dataset_type
    )

    # Save preprocessed data scaled
    _save_data_and_store_info(
        data=data_scaled,
        file_name=f"{type}{Naming.AGGR_SCALED}",
        dst_dir=data_prep_dir,
        dataset_type=dataset_type
    )

    # Save the single class data
    for class_data in class_data_list:
        _save_data_and_store_info(
            data=class_data, 
            file_name=class_data['class'].iloc[0],
            dst_dir=single_classes_dir,
            dataset_type=dataset_type
        )

    # Save normal_anomaly data
    for normal_anomaly_data in normal_anomaly_data_list:
        # Get anomaly class name for file_name
        anomaly_class_mask = normal_anomaly_data['class'] != 'Normal'
        anomaly_class = normal_anomaly_data[anomaly_class_mask]['class'].iloc[0]

        _save_data_and_store_info(
            data=normal_anomaly_data,
            file_name=f"Normal_{anomaly_class}",
            dst_dir=normal_anomaly_dir,
            dataset_type=dataset_type
        )

    # Save feature mean
    for feature_mean in feature_mean_list:
        update_or_append_csv(
            file_path=ProjectPaths.DATASETS_FEATURES_MEAN,
            data_dict=feature_mean,
            match_keys=['dataset_type', 'class'],
            id_column='id'
        )

    # Save feature variance
    for feature_variance in feature_variance_list:
        update_or_append_csv(
            file_path=ProjectPaths.DATASETS_FEATURES_VAR,
            data_dict=feature_variance,
            match_keys=['dataset_type', 'class'],
            id_column='id'
        )

    print(f"File-level preprocessing for {dataset_type} done.")


if __name__ == "__main__":
    pass
