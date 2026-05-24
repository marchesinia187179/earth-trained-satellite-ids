import pandas as pd


def minority_removal_nb15(data):
    minority_classes = ['Analysis', 'Backdoor', 'Shellcode', 'Worms']
    
    return data[~data['attack_cat'].isin(minority_classes)]


def merge_minority_stin(data):
    botnet_classes = ['Botnet', 'Web Attack', 'Backdoor']
    ddos_classes = ['LDAP_DDoS', 'MSSQL_DDoS', 'NetBIOS_DDoS', 'Portmap_DDoS']

    data['label'] = data['label'].replace(botnet_classes, 'Botnet')
    data['label'] = data['label'].replace(ddos_classes, 'DDoS')

    return data


def align_nb15(data):
    """
    Allinea il dataset NB15 trasformando e selezionando le feature specificate.
    """
    new_df = pd.DataFrame()

    # Calcolo delle nuove features
    new_df['duration'] = data['dur'] * 1000000
    new_df['src_bytes'] = data['sbytes']
    new_df['dst_bytes'] = data['dbytes']
    new_df['src_pkts'] = data['spkts']
    new_df['dst_pkts'] = data['dpkts']
    new_df['src_win_byt'] = data['swin']
    new_df['dst_win_byt'] = data['dwin']
    new_df['load_s'] = ((data['sload'] + data['dload']) / 8) / 1000000
    new_df['fl_iat_min'] = data[['sinpkt', 'dinpkt']].min(axis=1) * 1000
    new_df['down_up_ratio'] = data['dpkts'] / (data['spkts'] + 1e-6)
    new_df['total_bytes'] = new_df['src_bytes'] + new_df['dst_bytes']
    new_df['total_pkts'] = new_df['src_pkts'] + new_df['dst_pkts']
    new_df['src_mean_pkt_size'] = new_df['src_bytes'] / (new_df['src_pkts'] + 1e-6)
    new_df['dst_mean_pkt_size'] = new_df['dst_bytes'] / (new_df['dst_pkts'] + 1e-6)
    new_df['pkts_per_sec'] = (new_df['src_pkts'] + new_df['dst_pkts']) / (new_df['duration'] + 1e-6)
    new_df['win_diff'] = new_df['src_win_byt'] - new_df['dst_win_byt']
    new_df['byte_ratio'] = new_df['dst_bytes'] / (new_df['src_bytes'] + 1e-6)
    new_df['attack_cat'] = data['attack_cat']
    new_df['label'] = data['label']

    return new_df


def align_stin(data):
    """
    Allinea il dataset STIN trasformando e selezionando le feature specificate.
    """
    new_df = pd.DataFrame()

    # Calcolo delle nuove features
    new_df['duration'] = data['fl_dur']
    new_df['src_bytes'] = data['l_fw_pkt']
    new_df['dst_bytes'] = data['l_bw_pkt']
    new_df['src_pkts'] = data['fw_pk']
    new_df['dst_pkts'] = data['bw_pkt_s'] * data['fl_dur'] * 1000000
    new_df['src_win_byt'] = data['fw_win_byt']
    new_df['dst_win_byt'] = data['bw_win_byt']
    new_df['load_s'] = data['fl_byt_s'] / 1000000
    new_df['fl_iat_min'] = data['fl_iat_min']
    new_df['down_up_ratio'] = data['down_up_ratio']
    new_df['total_bytes'] = new_df['src_bytes'] + new_df['dst_bytes']
    new_df['total_pkts'] = new_df['src_pkts'] + new_df['dst_pkts']
    new_df['src_mean_pkt_size'] = new_df['src_bytes'] / (new_df['src_pkts'] + 1e-6)
    new_df['dst_mean_pkt_size'] = new_df['dst_bytes'] / (new_df['dst_pkts'] + 1e-6)
    new_df['pkts_per_sec'] = (new_df['src_pkts'] + new_df['dst_pkts']) / (new_df['duration'] + 1e-6)
    new_df['win_diff'] = new_df['src_win_byt'] - new_df['dst_win_byt']
    new_df['byte_ratio'] = new_df['dst_bytes'] / (new_df['src_bytes'] + 1e-6)
    new_df['attack_cat'] = data['label']
    new_df['label'] = 1

    return new_df


def normalize_dataset(data):
    """
    Normalizza le feature numeriche del dataset utilizzando il Min-Max scaling.
    Formula: X_i = (x_i - min(x_i)) / (max(x_i) - min(x_i))
    """
    # Escludiamo le colonne di target dalla normalizzazione
    cols_to_exclude = ['attack_cat', 'label']
    
    for col in data.columns:
        # Procediamo solo se la colonna è numerica e non è tra quelle da escludere
        if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(data[col]):
            min_val = data[col].min()
            max_val = data[col].max()
            
            # Calcolo della differenza (max - min) per il denominatore
            diff = max_val - min_val
            
            if diff != 0:
                data[col] = (data[col] - min_val) / diff
            else:
                # Se tutti i valori sono uguali, impostiamo a 0.0
                data[col] = 0.0
                
    return data


def data_preprocessing(data, type):

    match type:
        case 'nb15':
            data = minority_removal_nb15(data)
            data = align_nb15(data)
        case 'ter20':
            data = merge_minority_stin(data)
            data = align_stin(data)
        case 'sat20':
            data = align_stin(data)
        case _:
            print("Invalid type!")
            return None
    
    data = normalize_dataset(data)

    return data