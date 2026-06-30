import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.file_utils import create_directory
from utils.config import ProjectPaths


def _build_clean_label(classes_val, dataset_type_val, prefix=""):
        """
        Parses a comma-separated string of classes and returns a clean
        condensed label string formatted for plotting axes

        :param classes_val: string of classes
        :param dataset_type_val: string indicating the dataset type
        :param prefix: optional string to prepend to the label
        :return: string - formatted label for plotting
        """
        # Split, strip whitespace, and filter out empty strings
        elements = [w.strip() for w in str(classes_val).split(',') if w.strip()]
        
        # If there are more than 2 classes (e.g., Normal + multiple attack types), it's an Aggregate dataset
        if len(elements) > 2:
            label_text = f"Aggregate {dataset_type_val}"
        else:
            # Filter out "Normal" to isolate the specific cyber-attack class
            attack_only = ", ".join([w for w in elements if w != "Normal"])
            label_text = f"{attack_only} {dataset_type_val}"
            
        return f"{prefix}{label_text})" if prefix else label_text


def _generate_heatmap_for_feature(models, data, dst_path, feature, row_order=None, col_order=None):
    """
    Generates and saves an optimized, auto-scaling standalone heatmap for a given performance metric

    :param models: DataFrame containing model information with columns: ['model_name', 'dataset_type', 'classes', ...]
    :param data: DataFrame containing classification results with columns: ['model_name', 'dataset_type', 'classes', feature]
    :param dst_path: Path to save the generated heatmap image
    :param feature: The performance metric to visualize
    """
    print(f"Generating optimized heatmap for {feature}")
    
    # Create deep copies to avoid SettingWithCopyWarning
    data = data.copy()
    models = models.copy()

    # Safely convert the target evaluation metric to numeric
    data[feature] = pd.to_numeric(data[feature], errors='coerce')

    # Rename columns in the models DataFrame to avoid collisions during the merge
    models = models[['model_name', 'dataset_type', 'classes']].rename(
        columns={
            'dataset_type': 'model_dataset_type',
            'classes': 'model_classes'
        }
    )

    # Merge the data DataFrame with the models metadata on 'model_name'
    # This brings the true training domain and classes into each test evaluation row
    merged_data = pd.merge(data, models, on='model_name', how='left')

    # Generate X-Axis Labels (Test Datasets) using 'data' columns
    merged_data['dataset_label'] = merged_data.apply(
        lambda row: _build_clean_label(row['classes'], row['dataset_type']), 
        axis=1
    )

    # Generate Y-Axis Labels (Trained Models) using joined 'models' columns
    merged_data['model_label'] = merged_data.apply(
        lambda row: _build_clean_label(row['model_classes'], row['model_dataset_type'], prefix="RF ("), 
        axis=1
    )

    # Pivot the merged dataframe into the final performance matrix for the heatmap
    pivot_data = merged_data.pivot(index='model_label', columns='dataset_label', values=feature)

    # Sort Rows (Models)
    if row_order is not None:
        valid_rows = [r for r in row_order if r in pivot_data.index]
        remaining_rows = [r for r in pivot_data.index if r not in valid_rows]
        pivot_data = pivot_data.reindex(index=valid_rows + remaining_rows)

    # Sort Columns (Test Datasets)
    if col_order is not None:
        valid_cols = [c for c in col_order if c in pivot_data.columns]
        remaining_cols = [c for c in pivot_data.columns if c not in valid_cols]
        pivot_data = pivot_data.reindex(columns=valid_cols + remaining_cols)

    # Get num of models for Y-axis and num of datasets for X-axis
    num_models = len(pivot_data.index)       
    num_datasets = len(pivot_data.columns)   

    # Dynamic layout and size tuning to give cells maximum breathing room
    cell_size = 1.0  # Inches per cell
    
    # Base padding for axes labels and long text strings
    margin_x = 6.0  
    margin_y = 2.5  
    
    fig_width = max(num_datasets * cell_size + margin_x, 14)
    fig_height = max(num_models * cell_size + margin_y, 8)

    # Initialize figure and global aesthetic style
    plt.figure(figsize=(fig_width, fig_height))
    sns.set_theme(style="white") 
    
    # Generate heatmap with maximized inner font sizes
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
    
    # Configure text labels and fix font size hierarchy
    plt.title(f"{feature} Performance Matrix", pad=25, fontsize=18, fontweight='bold')
    plt.ylabel("Trained Models", fontsize=14, fontweight='bold', labelpad=15)
    plt.xlabel("Testing Datasets & Classes", fontsize=14, fontweight='bold', labelpad=15)
    
    # Rotate long dataset labels to 45° to drastically reduce required vertical space
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=11)
    
    # Save figure using 'bbox_inches' to cleanly dynamic-fit all outer elements without clipping
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"{feature} Heatmap successfully saved to: {dst_path}")


def plotting_processing(models, data, mode, metrics, row_order=None, col_order=None):
    """
    Processing function to generate heatmaps for some metrics from classification results

    :param models: DataFrame containing model information with columns: ['model_name', 'dataset_type', 'classes', ...]
    :param data: DataFrame containing classification results with columns: ['model_name', 'dataset_type', 'classes', metrics...]
    :param mode: string indicating the mode (normalized or unnormalized)
    :param metrics: list of performance metrics to visualize
    """
    plot_dir = create_directory(mode, ProjectPaths.PLOTTING_DIR)

    for feature in metrics:
        dst_path = plot_dir / f"{feature}_matrix.png"
        
        _generate_heatmap_for_feature(models, data, dst_path, feature, row_order, col_order)


if __name__ == "__main__":
    # Define your paths consistently with your pipeline architecture
    # global_csv = CLASSIFICATIONS_DIR / mode / CLASSIFICATIONS_FILENAME
    # plot_dir = global_csv.parent / "plots"
    # plot_dir.mkdir(parents=True, exist_ok=True)
    
    # generate_tpr_heatmap(global_csv, plot_dir / "tpr_matrix.png")
    # generate_tnr_heatmap(global_csv, plot_dir / "tnr_matrix.png")
    pass