import pandas as pd
import sys


def minority_removal_nb15(data):
    """
    Removes specific minority attack categories from the NB15 dataset to reduce noise or focus on main classes.

    :param data: Input pandas DataFrame containing the NB15 dataset.
    :internal minority_classes: List of attack categories to be filtered out ('Analysis', 'Backdoor', 'Shellcode', 'Worms').
    :return: Filtered pandas DataFrame.
    """
    if 'attack_cat' not in data.columns:
        print("Error: Column 'attack_cat' missing in NB15 dataset.")
        sys.exit(1)

    minority_classes = ['Analysis', 'Backdoor', 'Shellcode', 'Worms']
    data = data[~data['attack_cat'].isin(minority_classes)]
    print("Minority classes removed (NB15).")
    return data


def merge_minority_stin(data):
    """
    Groups minority attack classes into broader categories for the STIN dataset.

    :param data: Input pandas DataFrame containing the STIN dataset.
    :internal botnet_classes: List of classes to be renamed to 'Botnet'.
    :internal ddos_classes: List of classes to be renamed to 'DDoS'.
    :return: Pandas DataFrame with merged labels.
    """
    if 'label' not in data.columns:
        print("Error: Column 'label' missing in dataset.")
        sys.exit(1)

    botnet_classes = ['Botnet', 'Web Attack', 'Backdoor']
    ddos_classes = ['LDAP_DDoS', 'MSSQL_DDoS', 'NetBIOS_DDoS', 'Portmap_DDoS']

    data['label'] = data['label'].replace(botnet_classes, 'Botnet')
    data['label'] = data['label'].replace(ddos_classes, 'DDoS')

    print("Minority classes grouped (STIN).")
    return data


def align_nb15(data):
    """
    Aligns the NB15 dataset by transforming and selecting specific features to match the required schema.

    :param data: Input pandas DataFrame (NB15 raw data).
    :internal new_df: Temporary DataFrame used to store calculated and renamed features.
    :internal required_columns: List of columns needed for the alignment process.
    :return: Aligned pandas DataFrame with standardized feature names.
    """
    required_columns = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'swin', 'dwin', 'sload', 'dload', 'sinpkt', 'dinpkt', 'attack_cat', 'label']
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        print(f"Error: Missing columns in NB15 for alignment: {missing}")
        sys.exit(1)

    new_df = pd.DataFrame()

    new_df['duration'] = data['dur'] * 1000000
    new_df['src_bytes'] = data['sbytes']
    new_df['dst_bytes'] = data['dbytes']
    new_df['src_pkts'] = data['spkts']
    new_df['dst_pkts'] = data['dpkts']
    new_df['src_win_byt'] = data['swin']
    new_df['dst_win_byt'] = data['dwin']
    new_df['load_s'] = ((data['sload'] + data['dload']) / 8)
    new_df['fl_iat_min'] = data[['sinpkt', 'dinpkt']].min(axis=1) * 1000
    new_df['down_up_ratio'] = data['dpkts'] / (data['spkts'] + 1e-6)
    new_df['total_bytes'] = new_df['src_bytes'] + new_df['dst_bytes']
    new_df['total_pkts'] = new_df['src_pkts'] + new_df['dst_pkts']
    new_df['src_mean_pkt_size'] = new_df['src_bytes'] / (new_df['src_pkts'] + 1e-6)
    new_df['dst_mean_pkt_size'] = new_df['dst_bytes'] / (new_df['dst_pkts'] + 1e-6)
    new_df['pkts_per_sec'] = (new_df['src_pkts'] + new_df['dst_pkts']) / (data['dur'] + 1e-6)
    new_df['win_diff'] = new_df['src_win_byt'] - new_df['dst_win_byt']
    new_df['byte_ratio'] = new_df['dst_bytes'] / (new_df['src_bytes'] + 1e-6)
    new_df['attack_cat'] = data['attack_cat']
    new_df['label'] = data['label']

    print("Dataset aligned (NB15).")
    return new_df


def align_stin(data):
    """
    Aligns the STIN dataset by transforming and selecting specific features to match the required schema.

    :param data: Input pandas DataFrame (STIN raw data).
    :internal new_df: Temporary DataFrame used to store calculated and renamed features.
    :internal required_columns: List of columns needed for the alignment process.
    :return: Aligned pandas DataFrame with standardized feature names.
    """
    required_columns = ['fl_dur', 'l_fw_pkt', 'l_bw_pkt', 'fw_pk', 'bw_pkt_s', 'fw_win_byt', 'bw_win_byt', 'fl_byt_s', 'fl_iat_min', 'down_up_ratio', 'label']
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        print(f"Error: Missing columns in STIN for alignment: {missing}")
        sys.exit(1)

    new_df = pd.DataFrame()

    new_df['duration'] = data['fl_dur']
    new_df['src_bytes'] = data['l_fw_pkt']
    new_df['dst_bytes'] = data['l_bw_pkt']
    new_df['src_pkts'] = data['fw_pk']
    new_df['dst_pkts'] = data['bw_pkt_s'] * data['fl_dur'] / 1000000
    new_df['src_win_byt'] = data['fw_win_byt']
    new_df['dst_win_byt'] = data['bw_win_byt']
    new_df['load_s'] = data['fl_byt_s']
    new_df['fl_iat_min'] = data['fl_iat_min']
    new_df['down_up_ratio'] = data['down_up_ratio']
    new_df['total_bytes'] = new_df['src_bytes'] + new_df['dst_bytes']
    new_df['total_pkts'] = new_df['src_pkts'] + new_df['dst_pkts']
    new_df['src_mean_pkt_size'] = new_df['src_bytes'] / (new_df['src_pkts'] + 1e-6)
    new_df['dst_mean_pkt_size'] = new_df['dst_bytes'] / (new_df['dst_pkts'] + 1e-6)
    new_df['pkts_per_sec'] = ((new_df['src_pkts'] + new_df['dst_pkts']) / (new_df['duration'] + 1e-6)) * 1000000
    new_df['win_diff'] = new_df['src_win_byt'] - new_df['dst_win_byt']
    new_df['byte_ratio'] = new_df['dst_bytes'] / (new_df['src_bytes'] + 1e-6)
    new_df['attack_cat'] = data['label']
    new_df['label'] = 1

    print("Dataset aligned (STIN).")
    return new_df


def normalize_dataset_independent(data):
    """
    Normalizes numerical features of the dataset using Min-Max scaling.
    Formula: X_i = (x_i - min(x_i)) / (max(x_i) - min(x_i))

    :param data: Input pandas DataFrame.
    :internal cols_to_exclude: Features that should not be normalized ('attack_cat', 'label').
    :return: Normalized pandas DataFrame.
    """
    cols_to_exclude = ['attack_cat', 'label', 'split_type']

    data_train = data[data['split_type'] == 'train']
    
    for col in data.columns:
        if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(data[col]):

            if not data_train.empty:
                min_val = data_train[col].min()
                max_val = data_train[col].max()
            else:
                min_val = data[col].min()
                max_val = data[col].max()
            
            diff = max_val - min_val
            
            if diff != 0:
                data[col] = (data[col] - min_val) / diff
            else:
                data[col] = 0.0
                
    print("Features normalized.")
    return data


def normalize_dataset_dependent(data, scaler_stats=None):
    """
    Normalizza le feature numeriche. Se scaler_stats è None, calcola i minimi e massimi (fase di Fit).
    Se scaler_stats è fornito, usa quei valori per normalizzare (fase di Transform).
    """
    cols_to_exclude = ['attack_cat', 'label', 'split_type']
    
    # Individuiamo le colonne numeriche da normalizzare
    numeric_cols = [col for col in data.columns if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(data[col])]
    
    # Se siamo sul Train Set (NB15), calcoliamo e salviamo i parametri
    if scaler_stats is None:
        data_train = data[data['split_type'] == 'train']

        scaler_stats = {}
        for col in numeric_cols:
            if not data_train.empty:
                min_val = data_train[col].min()
                max_val = data_train[col].max()
            else:
                min_val = data[col].min()
                max_val = data[col].max()
                
            scaler_stats[col] = {
                'min': min_val,
                'max': max_val
            }

        print("Parametri di normalizzazione calcolati (Fit).")
    else:
        print("Parametri di normalizzazione ereditati (Transform).")

    # Applichiamo la normalizzazione
    for col in numeric_cols:
        if col in scaler_stats:
            min_val = scaler_stats[col]['min']
            max_val = scaler_stats[col]['max']
            diff = max_val - min_val
            
            if diff != 0:
                data[col] = (data[col] - min_val) / diff
            else:
                data[col] = 0.0
                
    print("Features normalized.")
    return data, scaler_stats


def stratified_split(data, train_split):
    # Add split column for training/testing (Stratified Split per attack_cat)
    # Using an explicit loop and concat to ensure 'attack_cat' column is preserved
    split_groups = []
    for _, group in data.groupby('attack_cat'):
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        n_group_train = int(len(group) * train_split)
        group['split_type'] = ['train'] * n_group_train + ['test'] * (len(group) - n_group_train)
        split_groups.append(group)
    
    return pd.concat(split_groups)


def data_preprocessing(data, dataset_type, dependent=True, scaler_stats=None, train_split=0.8):
    """
    Main orchestration function for data preprocessing based on the dataset type.

    :param data: Raw pandas DataFrame.
    :param dataset_type: String indicating the dataset type ('nb15', 'ter20', 'sat20').
    :param train_split: Percentage of data to be assigned to training (0.0 to 1.0).
    :return: Preprocessed and normalized pandas DataFrame.
    """
    print(f"Running data-level preprocessing for {dataset_type}...")

    data = data.copy()
        
    if dataset_type == 'nb15':
        data = minority_removal_nb15(data)
        data = align_nb15(data)

    if dataset_type == 'ter20':
        data = merge_minority_stin(data)
        data = align_stin(data)
    
    if dataset_type == 'sat20':
        data = align_stin(data)

    data = stratified_split(data, train_split)
    
    if dependent:
        data, scaler_stats = normalize_dataset_dependent(data, scaler_stats)
    else:
        data = normalize_dataset_independent(data)

    # Final shuffle to mix train/test labels
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Data-level preprocessing for {dataset_type} done.")
    return data, scaler_stats