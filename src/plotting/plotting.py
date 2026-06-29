import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.file_utils import create_directory
from utils.paths import PLOTTING_DIR


def _generate_heatmap_for_feature(data, dst_path, feature):
    """
    Generates and saves an optimized, auto-scaling standalone heatmap for a given performance metric

    :param data: DataFrame containing classification results with columns: ['model_name', 'dataset_type', 'classes', feature]
    :param dst_path: Path to save the generated heatmap image
    :param feature: The performance metric to visualize
    """
    print(f"Generating optimized heatmap for {feature}")
    
    # 1. Prepare data and create the combined X-axis label
    data = data.copy()
    data[feature] = pd.to_numeric(data[feature], errors='coerce')
    data['dataset_label'] = data['dataset_type'] + " (" + data['classes'] + ")"
    
    # 2. Pivot dataframe (Rows = Models, Columns = Test Datasets)
    pivot_data = data.pivot(index='model_name', columns='dataset_label', values=feature)

    num_models = len(pivot_data.index)       
    num_datasets = len(pivot_data.columns)   

    # 3. Dynamic layout and size tuning to give cells maximum breathing room
    cell_size = 1.0  # Inches per cell
    
    # Base padding for axes labels and long text strings
    margin_x = 6.0  
    margin_y = 2.5  
    
    fig_width = max(num_datasets * cell_size + margin_x, 14)
    fig_height = max(num_models * cell_size + margin_y, 8)

    # 4. Initialize figure and global aesthetic style
    plt.figure(figsize=(fig_width, fig_height))
    sns.set_theme(style="white") 
    
    # 5. Generate heatmap with maximized inner font sizes
    sns.heatmap(
        pivot_data, 
        annot=True, 
        fmt=".3f", 
        annot_kws={"size": 11, "weight": "bold"},  # Scaled up to occupy max internal space
        cmap="Blues", 
        vmin=0.0, 
        vmax=1.0, 
        square=True,                                # Keeps cells perfectly 1:1 square
        cbar_kws={
            'label': f'{feature} Value', 
            'shrink': 0.6,                          # Prevents colorbar from stretching past the matrix
            'pad': 0.03                             # Brings colorbar closer to the matrix
        }
    )
    
    # 6. Configure text labels and fix font size hierarchy
    plt.title(f"{feature} Performance Matrix", pad=25, fontsize=18, fontweight='bold')
    plt.ylabel("Trained Models", fontsize=14, fontweight='bold', labelpad=15)
    plt.xlabel("Testing Datasets & Classes", fontsize=14, fontweight='bold', labelpad=15)
    
    # Rotate long dataset labels to 45° to drastically reduce required vertical space
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=11)
    
    # 7. Save figure using 'bbox_inches' to cleanly dynamic-fit all outer elements without clipping
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{feature} Heatmap successfully saved to: {dst_path}")


def plotting_processing(data, mode, metrics):
    """
    Processing function to generate heatmaps for some metrics from classification results


    """
    plot_dir = create_directory(mode, PLOTTING_DIR)

    for feature in metrics:
        dst_path = plot_dir / f"{feature}_matrix.png"
        
        _generate_heatmap_for_feature(data, dst_path, feature)


if __name__ == "__main__":
    # Define your paths consistently with your pipeline architecture
    # global_csv = CLASSIFICATIONS_DIR / mode / CLASSIFICATIONS_FILENAME
    # plot_dir = global_csv.parent / "plots"
    # plot_dir.mkdir(parents=True, exist_ok=True)
    
    # generate_tpr_heatmap(global_csv, plot_dir / "tpr_matrix.png")
    # generate_tnr_heatmap(global_csv, plot_dir / "tnr_matrix.png")
    pass