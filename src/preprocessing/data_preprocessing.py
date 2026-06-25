import pandas as pd
import sys

from utils.paths import TRAIN_SPLIT


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

    new_df['duration'] = data['dur'] * 1000000  # flow duration [sE+6 = µs]
    new_df['src_bytes'] = data['sbytes']    # total length of forward packets [Byte]
    new_df['dst_bytes'] = data['dbytes']    # total length of forward packets [Byte]
    new_df['src_pkts'] = data['spkts']  # total packets in the forward direction
    new_df['dst_pkts'] = data['dpkts']  # total packets in the backward direction
    new_df['src_win_byt'] = data['swin']    # number of forward bytes in the initial window [Byte]
    new_df['dst_win_byt'] = data['dwin']    # number of backward bytes in the initial window [Byte]
    new_df['load_s'] = ((data['sload'] + data['dload']) / 8)    # packets bytes transmitted per second [(bit/8)/s = B/s]
    new_df['down_up_ratio'] = data['dpkts'] / (data['spkts'] + 1e-6)    # asymmetry between downlink and uplink channels
    new_df['total_bytes'] = new_df['src_bytes'] + new_df['dst_bytes']   # total bytes of channels [Byte]
    new_df['total_pkts'] = new_df['src_pkts'] + new_df['dst_pkts']  # total number of packets of channels
    new_df['src_mean_pkt_size'] = new_df['src_bytes'] / (new_df['src_pkts'] + 1e-6)     # mean of packets length for source channel [Byte]
    new_df['dst_mean_pkt_size'] = new_df['dst_bytes'] / (new_df['dst_pkts'] + 1e-6)     # mean of packets length for destination channel [Byte]
    new_df['pkts_per_sec'] = (new_df['src_pkts'] + new_df['dst_pkts']) / (data['dur'] + 1e-6)   # packets per second [1/s]
    new_df['win_diff'] = new_df['src_win_byt'] - new_df['dst_win_byt']  # difference between number of source and destination bytes in the initial window [Byte]
    new_df['byte_ratio'] = new_df['dst_bytes'] / (new_df['src_bytes'] + 1e-6)   # asymmetry between destination and source bytes [Byte]
    new_df['class'] = data['attack_cat']    # string of class of the data
    new_df['label'] = data['label']     # bolean of class of the data: 1 for anomaly or 0 for normal

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

    new_df['duration'] = data['fl_dur']     # flow duration [µs]
    new_df['src_bytes'] = data['l_fw_pkt']  # total length of forward packets [Byte]
    new_df['dst_bytes'] = data['l_bw_pkt']  # total length of forward packets [Byte]
    new_df['src_pkts'] = data['fw_pk']  # total packets in the forward direction
    new_df['dst_pkts'] = data['bw_pkt_s'] * data['fl_dur'] / 1000000    # total packets in the backward direction: backward packets per second * flow duration / 1E+6 [Byte µs / s E+6 = Byte]
    new_df['src_win_byt'] = data['fw_win_byt']  # number of forward bytes in the initial window [Byte]
    new_df['dst_win_byt'] = data['bw_win_byt']  # number of backward bytes in the initial window [Byte]
    new_df['load_s'] = data['fl_byt_s']     # packets bytes transmitted per second [B/s]
    new_df['down_up_ratio'] = data['down_up_ratio'] # asymmetry between downlink and uplink channels
    new_df['total_bytes'] = new_df['src_bytes'] + new_df['dst_bytes']   # total bytes of channels [Byte]
    new_df['total_pkts'] = new_df['src_pkts'] + new_df['dst_pkts']  # total number of packets of channels
    new_df['src_mean_pkt_size'] = new_df['src_bytes'] / (new_df['src_pkts'] + 1e-6)     # mean of packets length for source channel [Byte]
    new_df['dst_mean_pkt_size'] = new_df['dst_bytes'] / (new_df['dst_pkts'] + 1e-6)     # mean of packets length for destination channel [Byte]
    new_df['pkts_per_sec'] = ((new_df['src_pkts'] + new_df['dst_pkts']) / (new_df['duration'] + 1e-6)) * 1000000    # packets per second [1/µs s E+6 = 1/s]
    new_df['win_diff'] = new_df['src_win_byt'] - new_df['dst_win_byt']  # difference between number of source and destination bytes in the initial window [Byte]
    new_df['byte_ratio'] = new_df['dst_bytes'] / (new_df['src_bytes'] + 1e-6)   # asymmetry between destination and source bytes [Byte]
    new_df['class'] = data['label']     # string of class of the data
    new_df['label'] = 1     # bolean of class of the data: 1 for anomaly or 0 for normal

    print("Dataset aligned (STIN).")
    return new_df


def stratified_split(data):
    split_groups = []
    for _, group in data.groupby('class'):
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        n_group_train = int(len(group) * TRAIN_SPLIT)
        group['split_type'] = ['train'] * n_group_train + ['test'] * (len(group) - n_group_train)
        split_groups.append(group)
    
    return pd.concat(split_groups)


def data_preprocessing(data, dataset_type):
    print(f"Running data-level preprocessing for {dataset_type}...")

    data = data.copy()
        
    if dataset_type == 'nb15':
        data = minority_removal_nb15(data)
        data = align_nb15(data)
    elif dataset_type == 'ter20':
        data = merge_minority_stin(data)
        data = align_stin(data)
    elif dataset_type == 'sat20':
        data = align_stin(data)

    data = stratified_split(data)

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Data-level preprocessing for {dataset_type} done.")
    return data


if __name__ == "__main__":
    # TODO
    pass