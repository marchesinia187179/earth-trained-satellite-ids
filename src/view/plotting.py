import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.paths import INDEPENDENT_RESULTS_MODELS_DIR, DEPENDENT_RESULTS_MODELS_DIR

def generate_custom_recall_heatmap(mode='independent', save_dir=None):
    """
    Genera una heatmap comparativa della Recall con Ordine e Nomi personalizzati,
    risolvendo le collisioni di nomi dei dataset tramite chiavi composte.
    """
    
    # =========================================================================
    # CONFIGURAZIONE PERSONALIZZATA
    # =========================================================================
    CUSTOM_MODELS_MAP = {
        'rf_model_6': 'Aggregate',
        'rf_model_1': 'Normal_DoS',
        'rf_model_2': 'Normal_Exploits',
        'rf_model_3': 'Normal_Fuzzers',
        'rf_model_4': 'Normal_Generic',
        'rf_model_5': 'Normal_Reconnaissance',
        'rf_model_7': 'Normal_SAT20',
        'rf_model_8': 'Normal_TER20',
    }

    CUSTOM_DATASETS_MAP = {
        # NB15 Baseline
        'nb15:nb15_preprocessed_scaled': 'NB15: Aggregate',
        'nb15:DoS': 'NB15: DoS',
        'nb15:Exploits': 'NB15: Exploits',
        'nb15:Fuzzers': 'NB15: Fuzzers',
        'nb15:Generic': 'NB15: Generic',
        'nb15:Normal': 'NB15: Normal',
        'nb15:Reconnaissance': 'NB15: Recon.',
        
        # SAT20 Scenarios
        'nb15+sat20:Normal_sat20': 'Aggr.: Normal + All SAT20',
        'sat20:Syn_DDoS': 'SAT20: Syn DDoS',
        'sat20:UDP_DDoS': 'SAT20: UDP DDoS',

        # TER20 Scenarios
        'nb15+ter20:Normal_ter20': 'Aggr.: Normal + All TER20',
        'ter20:Botnet': 'TER20: Botnet',
        'ter20:DDoS': 'TER20: DDoS',
        'ter20:Syn_DDoS': 'TER20: Syn DDoS',
        'ter20:UDP_DDoS': 'TER20: UDP DDoS'
    }
    # =========================================================================

    # 1. Identifica la cartella sorgente
    models_root = INDEPENDENT_RESULTS_MODELS_DIR if mode == 'independent' else DEPENDENT_RESULTS_MODELS_DIR
    all_data = []
    
    # 2. Carica i dati basandoti sui modelli configurati nella tua mappa personalizzata
    for model_key in CUSTOM_MODELS_MAP.keys():
        csv_path = models_root / model_key / f"{model_key}_classification_results.csv"
        
        if not csv_path.exists():
            print(f"Warning: File {model_key} in {csv_path} not found. Skipped.")
            continue
            
        df = pd.read_csv(csv_path)
        df = df[df['model_type'] == 'random_forest']
        df['recall'] = df['recall'].replace('None', np.nan).astype(float)
        
        for _, row in df.iterrows():
            # Genera la chiave composta dinamica usando i dati della riga corrente
            composite_key = f"{row['dataset_type']}:{row['testing_dataset']}"
            
            # Controlla la corrispondenza con la mappa univoca
            if composite_key in CUSTOM_DATASETS_MAP:
                all_data.append({
                    'Model': CUSTOM_MODELS_MAP[model_key],          
                    'Dataset': CUSTOM_DATASETS_MAP[composite_key],  # Mappa il nome formale corretto
                    'Recall': row['recall']
                })
            
    if not all_data:
        print("Errore: Nessun dato corrispondente alla configurazione trovato.")
        return

    # 3. Costruisci il DataFrame
    results_df = pd.DataFrame(all_data)

    # 4. TRASFORMAZIONE IN PIVOT MATRIX
    matrix_df = results_df.pivot(index='Model', columns='Dataset', values='Recall')
    
    # Mantiene l'ordine esatto imposto dalle chiavi dei tuoi dizionari configurati
    ordered_rows = [CUSTOM_MODELS_MAP[k] for k in CUSTOM_MODELS_MAP.keys() if CUSTOM_MODELS_MAP[k] in matrix_df.index]
    ordered_cols = [CUSTOM_DATASETS_MAP[k] for k in CUSTOM_DATASETS_MAP.keys() if CUSTOM_DATASETS_MAP[k] in matrix_df.columns]
    
    matrix_df = matrix_df.reindex(index=ordered_rows, columns=ordered_cols)

    # 5. PLOTTING CON MATPLOTLIB E SEABORN
    plt.figure(figsize=(18, 9))  # Leggermente più largo per visualizzare tutte le colonne separate
    sns.set_theme(style="white")

    ax = sns.heatmap(
        matrix_df, 
        annot=True, 
        fmt=".4f", 
        cmap="YlGnBu", 
        linewidths=.7, 
        vmin=0.0, 
        vmax=1.0,
        cbar_kws={'label': 'Recall Score'},
        annot_kws={"size": 10, "weight": "bold"}
    )

    # Configurazione grafiche ed etichette
    plt.title(f"Random Forest Cross-Domain Performance - Recall Matrix ({mode.upper()} MODE)", fontsize=14, pad=20, weight='bold')
    plt.xlabel("Testing Scenarios (j-th Dataset)", fontsize=12, labelpad=15, weight='bold')
    plt.ylabel("Trained Classifiers (i-th Model)", fontsize=12, labelpad=15, weight='bold')
    
    plt.xticks(rotation=35, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()

    # 6. SALVATAGGIO
    if save_dir:
        out_path = Path(save_dir)
    else:
        out_path = models_root.parent / "plots"
        
    out_path.mkdir(parents=True, exist_ok=True)
    file_name = out_path / f"rf_custom_recall_matrix_{mode}.png"
    
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"Heatmap personalizzata salvata con successo in: {file_name}")

if __name__ == "__main__":
    generate_custom_recall_heatmap(mode='independent')
    generate_custom_recall_heatmap(mode='dependent')