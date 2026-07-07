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
    Determines the correct number of samples to take from the normal and anomaly data based on the
    `MLConstants.NORMAL_ANOMALY_RATIO`. It returns the sampled normal and anomaly data as two separate pandas DataFrames.

    :param normal_data: Input pandas DataFrame containing the normal data
    :param anomaly_data: Input pandas DataFrame containing the anomaly data
    :param normal_samples: Number of samples in the normal data
    :param anomaly_samples: Number of samples in the anomaly data
    :return: Two pandas DataFrames containing the sampled normal and anomaly data
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
    Safely samples `n_samples` from the `data` while maintaining the original split proportions of the data.

    :param data: Input pandas DataFrame containing the dataset
    :param n_samples: Number of samples to draw from the dataset
    :return: A new pandas DataFrame containing the sampled data
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


def _split_by_class(data, dataset_type):
    """
    Splits the `data` by class and returns a list of data for each class.

    :param data: pool of data to be split
    :param dataset_type: type of the data
    :return: list of data for each class
    """
    # Security check if the feature exists
    if 'class' not in data.columns:
        print(f"Error: 'class' column not found during split. Skipping {dataset_type}.")
        return
    
    # Split the data by class and save them in a list
    return [data[data['class'] == c].copy() for c in data['class'].unique()]


def _merge_normal_anomaly(data, dataset_type):
    """
    Merges the `Normal` and `Anomaly` data based on the `MLConstants.NORMAL_ANOMALY_RATIO` and returns a list of merged data.

    :param data: pool of data to be merged
    :param dataset_type: type of the data
    :return: list of merged data
    """
    # Security check if the Normal class exists
    class_list = data['class'].unique()
    if not "Normal" in class_list:
        print(f"Error: Normal class not found! Skipping {dataset_type}.")
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

        # Append normal_anomaly data
        normal_anomaly_data_list.append(normal_anomaly_data)

    return normal_anomaly_data_list


def _scale_by_normal_anomaly_ratio(data, dataset_type):
    """
    Scales the `data` by the `MLConstants.NORMAL_ANOMALY_RATIO` and returns the scaled data.

    :param data: pool of data to be scaled
    :param dataset_type: type of the data
    :return: scaled data
    """
    # Security check if the Normal class exists
    class_list = data['class'].unique()
    if not "Normal" in class_list:
        print(f"Error: Normal class not found! Skipping {dataset_type}.")
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
    Calculates the mean and variance for the aggregated dataset and for each class in the dataset.

    :param data: Input pandas DataFrame containing the dataset
    :param dataset_type: The type of the dataset
    :return: Two lists containing the mean and variance records for the aggregated dataset and each class
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


def _save_data_and_store_info(data, file_name, dst_dir, dataset_type):
    """
    Saves the given data to a CSV file and stores the file information.

    :param data: Input pandas DataFrame containing the dataset
    :param file_name: Name of the CSV file to be created
    :param dst_dir: Destination directory where the CSV file will be saved
    :param dataset_type: The type of the dataset
    :return: None
    """

    # Save the data to a CSV file
    file_path = create_csv_from_data(data, file_name, dst_dir)

    # Store the file information
    store_file_info(file_path, dataset_type)


# --- Public Functions ---
def hybrid_dataset_file_preprocessing(nb15_normal_data, nb15_anomaly_data, sat20_anomaly_data, ter20_anomaly_data):
    """
    Creates the main directories and files of the hybrid dataset.

    :param nb15_normal_data: normal data from the nb15 dataset
    :param nb15_anomaly_data: anomaly data from the nb15 dataset
    :param sat20_anomaly_data: anomaly data from the sat20 dataset
    :param ter20_anomaly_data: anomaly data from the ter20 dataset
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
    
    # --- Create the directories ---
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
    
    # Create normal_anomaly directory for nb15_sat20
    nb15_sat20_normal_anomaly_dir = create_directory(
        dir_name=ProjectPaths.DIR_NORMAL_ANOMALY, 
        parent_path=nb15_sat20_dir
    )

    # Create normal_anomaly directory for nb15_ter20
    nb15_ter20_normal_anomaly_dir = create_directory(
        dir_name=ProjectPaths.DIR_NORMAL_ANOMALY, 
        parent_path=nb15_ter20_dir
    )

    # --- Process the data ---
    # Create stin data
    stin_anomaly_data = concat_and_shuffle([sat20_anomaly_data, ter20_anomaly_data])
    
    # Get stin_anomaly and nb15_anomaly data sample with proportion equal 1/3 and 2/3
    stin_size = min(len(stin_anomaly_data), len(nb15_anomaly_data) // 2)
    nb15_anomaly_size = stin_size * 2

    stin_anomaly_sampled = _safe_stratified_sample(
        data=stin_anomaly_data,
        n_samples=stin_size
    )

    nb15_anomaly_sampled = _safe_stratified_sample(
        data=nb15_anomaly_data,
        n_samples=nb15_anomaly_size
    )

    # Create the hybrid data
    hybrid_data = concat_and_shuffle([nb15_normal_data, stin_anomaly_sampled, nb15_anomaly_sampled])
    nb15_stin_data = concat_and_shuffle([nb15_normal_data, stin_anomaly_data])
    nb15_sat20_data = concat_and_shuffle([nb15_normal_data, sat20_anomaly_data])
    nb15_ter20_data = concat_and_shuffle([nb15_normal_data, ter20_anomaly_data])

    # Combine the hybrid data with own dataset type
    datasets = [
        {'dataset_type': Naming.HYBRID, 'data': hybrid_data},
        {'dataset_type': Naming.NB15_STIN, 'data': nb15_stin_data},
        {'dataset_type': Naming.NB15_SAT20, 'data': nb15_sat20_data},
        {'dataset_type': Naming.NB15_TER20, 'data': nb15_ter20_data}
    ]

    for d in datasets:
        # Select data and dataset_type
        dataset_type = d['dataset_type']
        data = d['data']
        
        print(f"\nRunning file-level preprocessing for {dataset_type}...")

        # Select current directory
        if dataset_type == Naming.HYBRID or dataset_type == Naming.NB15_STIN:
            data_prep_dir = hybrid_dir
        elif dataset_type == Naming.NB15_SAT20:
            data_prep_dir = nb15_sat20_dir
            normal_anomaly_dir = nb15_sat20_normal_anomaly_dir
        elif dataset_type == Naming.NB15_TER20:
            data_prep_dir = nb15_ter20_dir
            normal_anomaly_dir = nb15_ter20_normal_anomaly_dir

        # --- Process the data of the current dataset
        # Scale preprocessed data by normal anomaly ratio
        data_scaled = _scale_by_normal_anomaly_ratio(data, dataset_type)

        # Create normal anomaly data
        if dataset_type == Naming.NB15_SAT20 or dataset_type == Naming.NB15_TER20:
            normal_anomaly_data_list = _merge_normal_anomaly(data, dataset_type)

        # Get feature mean and variance
        feature_mean_list, feature_variance_list =_get_mean_and_variance(data, dataset_type)

        # --- Save the processed data of the current dataset ---
        # Save preprocessed data
        _save_data_and_store_info(
            data=data,
            file_name=f'{dataset_type}{Naming.AGGR}',
            dst_dir=data_prep_dir,
            dataset_type=dataset_type
        )
        
        # Save preprocessed data scaled
        _save_data_and_store_info(
            data=data_scaled,
            file_name=f"{dataset_type}{Naming.AGGR_SCALED}",
            dst_dir=data_prep_dir,
            dataset_type=dataset_type
        )
        
        # Save normal_anomaly data
        if dataset_type == Naming.NB15_SAT20 or dataset_type == Naming.NB15_TER20:
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

        # Map mean and variance lists with own file_path
        jobs = [
            (feature_mean_list, ProjectPaths.DATASETS_FEATURES_MEAN),
            (feature_variance_list, ProjectPaths.DATASETS_FEATURES_VAR)
        ]

        for data_list, file_path in jobs:
            for feature_data in data_list:
                update_or_append_csv(
                    file_path=file_path,
                    data_dict=feature_data,
                    match_keys=['dataset_type', 'class'],
                    id_column='id'
                )

        print(f"\nFile-level preprocessing for {dataset_type} done.")


def single_dataset_file_preprocessing(data, dataset_type):
    """
    Creates the main directories and files of the single dataset.
    
    :param data: Input pandas DataFrame containing the dataset
    :param dataset_type: The type of the dataset
    :return: None
    """
    print(f"\nRunning file-level preprocessing for {dataset_type}...")

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
    # create normal anomaly data
    if dataset_type == Naming.NB15:
        data_scaled = _scale_by_normal_anomaly_ratio(data, dataset_type)
        normal_anomaly_data_list = _merge_normal_anomaly(data, dataset_type)

    # Get feature mean and variance
    feature_mean_list, feature_variance_list = _get_mean_and_variance(data, dataset_type)

    # --- Save the preprocessed data ---
    # Save preprocessed data
    _save_data_and_store_info(
        data=data,
        file_name=f'{dataset_type}{Naming.AGGR}',
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

    # Save preprocessed data scaled
    if dataset_type == Naming.NB15:
        _save_data_and_store_info(
            data=data_scaled,
            file_name=f"{dataset_type}{Naming.AGGR_SCALED}",
            dst_dir=data_prep_dir,
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
    
    # Map mean and variance lists with own file_path
    jobs = [
        (feature_mean_list, ProjectPaths.DATASETS_FEATURES_MEAN),
        (feature_variance_list, ProjectPaths.DATASETS_FEATURES_VAR)
    ]

    for data_list, file_path in jobs:
        for feature_data in data_list:
            update_or_append_csv(
                file_path=file_path,
                data_dict=feature_data,
                match_keys=['dataset_type', 'class'],
                id_column='id'
            )

    print(f"\nFile-level preprocessing for {dataset_type} done.")


if __name__ == "__main__":
    pass
