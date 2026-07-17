"""
Heatmap generation functions for visualizing model performance across different datasets and features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .utils.file_utils import create_directory
from .utils.config import MLConstants, Naming, PlotConfig, ProjectPaths
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance


# --- Internal Helper Functions ---
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
    
    # Create label
    if len(elements) > 2:
        label_text = f"Aggregate {dataset_type_val}"
    elif len(elements) == 1 and elements[0] == "Normal":
        label_text = f"Normal {dataset_type_val}"
    else:
        attack_only = ", ".join([w for w in elements if w != "Normal"])
        label_text = f"{attack_only} {dataset_type_val}"
        
    return f"{prefix}{label_text})" if prefix else label_text


# --- Public Functions ---
def save_feature_importances_plot(model, model_name, feature_names, dst_dir, X_test=None, y_test=None):
    """
    Plots feature importances for Random Forest, Decision Tree, or HistGradientBoosting models.
    Uses MDI (Mean Decrease in Impurity) for RF and DT if available, and falls back to 
    Permutation Importance for HGB or if testing data is provided.

    :param model: the trained classifier object (RF, DT, or HGB)
    :param model_name: standardized name for the trained model
    :param feature_names: list of feature names corresponding to the importances
    :param dst_dir: destination directory to save the feature importance plot
    :param X_test: DataFrame containing test features (required for HGB permutation importance)
    :param y_test: Series containing test labels (required for HGB permutation importance)
    """
    importances = None
    std = None

    # Check if the model supports impurity-based feature importances (RF, DT)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Calculate standard deviation across trees ONLY if it is a Random Forest ensemble
        if hasattr(model, 'estimators_'):
            std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            
    # Fallback to Permutation Importance if native importances are missing (HGB)
    elif X_test is not None and y_test is not None:
        print(f" -> [INFO] Computing permutation importance for {model_name}...")
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=MLConstants.MAIN_SEED, n_jobs=-1
        )
        importances = result.importances_mean
        std = result.importances_std
    else:
        print(f" -> [WARNING] Skipping plot: {model_name} does not support native feature importances, "
              f"and X_test/y_test were not provided for permutation importance.")
        return

    # Target path for saving the figure
    dst_path = dst_dir / f"{model_name}{Naming.PLOT_EXT}"

    # Construct the data structure for sorting and plotting
    data_dict = {'importance': importances}
    if std is not None:
        data_dict['std'] = std

    df = pd.DataFrame(data_dict, index=feature_names).sort_values(by='importance', ascending=False)

    # Plotting configuration
    plt.figure(figsize=(10, 6))
    
    # Render bars with or without standard deviation error bars dynamically
    yerr_vals = df['std'] if 'std' in df.columns else None
    df['importance'].plot.bar(
        yerr=yerr_vals, capsize=4, color='skyblue', edgecolor='black'
    )
    
    # Set plot aesthetics and descriptive titles
    title_suffix = "via Permutation Importance" if not hasattr(model, 'feature_importances_') else "via MDI"
    plt.title(f"Feature Importances {title_suffix}", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Importance Score", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot with high resolution and close the figure
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature importance plot successfully saved to: {dst_path}")


def save_all_pca_plots(config_csv_path, label_column=MLConstants.Y_LABEL, drop_labels=MLConstants.X_DROP_LABELS):
    """
    Automated batch runner that reads a configuration CSV, loads each dataset,
    extracts the test split, applies PCA (2 components), and generates 2D scatter plots.
    This function independently fits and transforms the data tailored to each dataset.

    :param config_csv_path: Path to the configuration CSV file containing 'path' and 'dataset_type'
    :param label_column: The name of the ground-truth target column
    :param drop_labels: List of columns (or features) to drop from the feature matrix before PCA
    """
    config_path = Path(config_csv_path)
    if not config_path.exists():
        print(f"[ERROR] Configuration CSV file not found at: {config_csv_path}")
        return

    # Load the master configuration file
    try:
        df_config = pd.read_csv(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to read configuration CSV: {e}")
        return

    # Validate required columns in the configuration file
    if 'path' not in df_config.columns or 'dataset_type' not in df_config.columns:
        print("[ERROR] The configuration CSV must contain both 'path' and 'dataset_type' columns.")
        return

    # Ensure the main output directory for PCA plots exists
    base_pca_dir = Path(ProjectPaths.PCA_PLOTS_DIR)
    base_pca_dir.mkdir(parents=True, exist_ok=True)

    print("Starting automated batch PCA plotting...")
    
    # Iterate through each dataset defined in the config file
    for _, row in df_config.iterrows():
        dataset_path_str = row['path']
        dataset_type = row['dataset_type']
        
        if pd.isna(dataset_path_str) or pd.isna(dataset_type):
            continue
            
        file_path = Path(dataset_path_str)
        if not file_path.exists():
            print(f" -> [WARNING] Dataset file not found, skipping: {file_path}")
            continue

        # Automatically extract the dataset name from the file path
        dataset_name = file_path.stem

        # Define the final plot filename and destination path
        pca_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
        dst_path = base_pca_dir / pca_filename

        try:
            # Read the dataset
            df_data = pd.read_csv(file_path)
            
            # Ensure 'split_type' exists to extract the test set
            if 'split_type' not in df_data.columns:
                print(f" -> [SKIP] 'split_type' column not found in {dataset_name}. Cannot extract test set.")
                continue
                
            # Isolate the test data
            test_data = df_data[df_data['split_type'] == 'test']
            if test_data.empty:
                print(f" -> [SKIP] No 'test' data found in {dataset_name}.")
                continue

            # Safely drop non-feature columns
            cols_to_drop = [col for col in drop_labels if col in test_data.columns]
            X_test = test_data.drop(columns=cols_to_drop)

            # Ensure the label column exists
            if label_column not in test_data.columns:
                print(f" -> [SKIP] Label column '{label_column}' not found in {dataset_name}")
                continue
                
            y_test = test_data[label_column]

            # Compute a brand new space tailored to this specific dataset (Fit + Transform)
            scaler = StandardScaler()
            pca = PCA(n_components=MLConstants.PCA_COMPONENTS, random_state=MLConstants.MAIN_SEED)
            
            X_scaled = scaler.fit_transform(X_test)
            X_pca = pca.fit_transform(X_scaled)
            
            # Map class labels to human-readable strings for plotting
            labels = y_test.map({0: 'Normal Traffic', 1: 'Attack/Anomaly'}).fillna(y_test)
            var = pca.explained_variance_ratio_ * 100
            
            # Configure the scatter plot with appropriate aesthetics
            plt.figure(figsize=(10, 6))
            ax = sns.scatterplot(
                x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, alpha=0.5,
                palette={'Normal Traffic': '#1f77b4', 'Attack/Anomaly': '#ff7f0e'}
            )
            
            # Set plot titles, labels, and grid for better readability
            plt.title(f"PCA 2D Projection ({dataset_type} - {dataset_name})\nTotal Variance Explained: {var.sum():.2f}%", fontsize=14, fontweight='bold', pad=15)
            plt.xlabel(f"PC1 ({var[0]:.2f}% Variance)", fontsize=12)
            plt.ylabel(f"PC2 ({var[1]:.2f}% Variance)", fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)

            # Add a legend with a title and customize its appearance
            legend = ax.legend(title='Target Traffic Class', loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1, 
                               frameon=True, framealpha=0.9, facecolor='white', edgecolor='black', title_fontsize=11)
            legend.get_title().set_fontweight('bold')
            
            # Save the PCA plot to the designated output folder
            plt.savefig(dst_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f" -> [OK] Saved: {pca_filename}")

        except Exception as e:
            print(f" -> [ERROR] Failed to process {dataset_name}: {e}")
            
    print("\n[COMPLETED] All PCA plots have been generated and saved directly to the target directory.")


def save_probability_plot(y_test, y_scores, model_name, dataset_type, dataset_name, dst_dir):
    """
    Plots the overlapping probability distributions for Normal traffic vs Attacks
    to visually demonstrate the classification threshold trap.
    Tailored specifically for supervised models (RF, DT, HGB) outputting probabilities.

    :param y_test: true class labels (0 for Normal, 1 for Attack)
    :param y_scores: predicted probabilities for the positive class (Attack)
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    :param dst_dir: destination path directory
    """
    # Create the main directory for probabilities
    prob_dir = create_directory(ProjectPaths.DIR_PROB_PLOTS, dst_dir)

    # Create a directory for the model's probability plots
    model_prob_dir = create_directory(model_name, prob_dir)

    # Define the filename and output path for the probability distribution plot
    prob_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
    
    # Define the full output path for the probability distribution plot
    dst_path = model_prob_dir / prob_filename

    # Map binary numeric labels to descriptive strings for the plot legend
    # Wrapping in pd.Series ensures safety and compatibility with both NumPy arrays and Pandas Series
    labels = pd.Series(y_test).map({0: 'Normal Traffic', 1: 'Attack/Anomaly'}).fillna(y_test)
    
    # Initialize the plot figure with specific dimensions
    plt.figure(figsize=(10, 6))
    
    # Dynamically determine the number of bins to prevent NumPy ValueError.
    # We use a tolerance of 1e-5 because if the range of probabilities is virtually zero,
    # splitting it into 50 bins is numerically unstable and visually meaningless.
    data_range = np.ptp(y_scores) if len(y_scores) > 0 else 0
    num_bins = 1 if data_range < 1e-5 else 50

    # Generate overlapping density histograms directly from the arrays.
    # We wrap the plotting call in a try-except block to guarantee that even under
    # extreme numerical edge cases, the pipeline does not crash.
    try:
        ax = sns.histplot(
            x=y_scores, hue=labels, element='step', stat='density', common_norm=False, 
            alpha=0.4, bins=num_bins, palette={'Normal Traffic': '#1f77b4', 'Attack/Anomaly': '#ff7f0e'}
        )
    except ValueError:
        # Ultimate fallback: if 50 bins still fail due to float precision, force 1 bin
        ax = sns.histplot(
            x=y_scores, hue=labels, element='step', stat='density', common_norm=False, 
            alpha=0.4, bins=1, palette={'Normal Traffic': '#1f77b4', 'Attack/Anomaly': '#ff7f0e'}
        )

    # Extract auto-generated legend elements from Seaborn before adding the manual threshold line
    sb_legend = ax.get_legend()
    if sb_legend is not None:
        # Cross-version compatibility for Matplotlib legend handles extraction
        handles = list(getattr(sb_legend, 'legend_handles', getattr(sb_legend, 'legendHandles', [])))
        labels_list = [t.get_text() for t in sb_legend.get_texts()]
        # Remove the default seaborn legend to avoid overlapping
        sb_legend.remove()
    else:
        handles, labels_list = ax.get_legend_handles_labels()

    # Since RF, DT, and HGB are supervised classifiers, they natively output probabilities [0, 1]
    thr_line = plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Standard Threshold (0.5)')
    plt.xlabel("Predicted Probability of Anomaly (Output of predict_proba)", fontsize=12)
    plt.xlim(0, 1)
        
    plt.title("Classification Score Distribution (Threshold Analysis)", fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.ylabel("Density", fontsize=12)
    
    # Append the threshold line handle and label to the existing legend items
    handles.append(thr_line)
    labels_list.append(thr_line.get_label())
    
    # Generate the final combined legend using the explicit handles and labels
    legend = ax.legend(
        handles=handles, labels=labels_list,
        title='Traffic Class / Threshold', loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1, 
        frameon=True, framealpha=0.9, facecolor='white', edgecolor='black', title_fontsize=11
    )
    legend.get_title().set_fontweight('bold')
    
    # Save the figure dynamically fitting all outer elements without clipping, then free up memory
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Output console feedback
    print(f"Probability distribution plot successfully saved to: {dst_path}")


def save_pr_curve_plot(y_test, y_scores, model_name, dataset_type, dataset_name, dst_dir):
    """
    Plots the Precision-Recall (PR) Curve to evaluate model performance (RF, DT, or HGB), 
    especially useful for highly imbalanced datasets.

    :param y_test: true class labels (0 for Normal, 1 for Attack)
    :param y_scores: predicted probabilities for the positive class (Attack)
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    :param dst_dir: destination path directory
    """
    # Guard clause: Cannot compute PR curve if there are no positive samples (Attacks)
    # This prevents errors when evaluating on purely "Normal" datasets
    if np.sum(y_test) == 0:
        print(f"Skipping PR Curve for {dataset_name}: No attack/anomaly samples present.")
        return

    # Create the main pr curve directory
    pr_dir = create_directory(ProjectPaths.DIR_PR_CURVE_PLOTS, dst_dir)

    # Create a directory for the model's PR Curve plots
    model_pr_dir = create_directory(model_name, pr_dir)

    # Define the filename and output path
    pr_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
    dst_path = model_pr_dir / pr_filename

    # Calculate Precision, Recall and the Average Precision (AP) score (valid for RF, DT, HGB)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap_score = average_precision_score(y_test, y_scores)

    # Calculate the baseline (No-skill classifier) which is just the ratio of positive cases
    baseline = np.sum(y_test) / len(y_test)

    # Initialize the plot figure
    plt.figure(figsize=(10, 6))

    # Plot the baseline (Random chance for imbalanced datasets)
    plt.plot([0, 1], [baseline, baseline], linestyle='--', color='gray', 
             linewidth=2, label=f'Baseline (AP = {baseline:.3f})')

    # Plot the actual PR Curve
    plt.plot(recall, precision, color='#1f77b4', linewidth=2, 
             label=f'Model PR Curve (AP = {ap_score:.3f})')
    
    # Lightly fill the area under the PR curve for visual emphasis
    plt.fill_between(recall, precision, alpha=0.1, color='#1f77b4')

    # Set plot aesthetics, labels, and limits
    plt.title(f"Precision-Recall Curve ({dataset_type})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Recall (True Positive Rate)", fontsize=12)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
    
    # PR curves must always be shown on a strictly [0, 1] grid
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle=':', alpha=0.6)

    # Customize the legend
    legend = plt.legend(title='Classifier Performance', loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1, 
                        frameon=True, framealpha=0.9, facecolor='white', edgecolor='black', title_fontsize=11)
    legend.get_title().set_fontweight('bold')

    # Save the plot
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Precision-Recall Curve successfully saved to: {dst_path}")


def save_threshold_metrics_plot(y_test, y_scores, model_name, dataset_type, dataset_name, dst_dir):
    """
    Plots Precision, Recall, and F1-Score as a function of the decision threshold.
    Highlights the threshold that maximizes the F1-Score for supervised models (RF, DT, HGB).

    :param y_test: true class labels (0 for Normal, 1 for Attack)
    :param y_scores: predicted probabilities for the positive class (Attack)
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    :param dst_dir: destination path directory
    """
    # Guard clause: Skip if no attacks exist in the ground truth
    if np.sum(y_test) == 0:
        print(f"Skipping Threshold Plot for {dataset_name}: No attack/anomaly samples present.")
        return

    # Create the main threshold_metrics directory
    thr_dir = create_directory(ProjectPaths.DIR_THRESHOLD_PLOTS, dst_dir)

    # Create a directory for the model's Threshold Sensitivity plots
    model_thr_dir = create_directory(model_name, thr_dir)

    # Define the filename and output path
    thr_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
    dst_path = model_thr_dir / thr_filename

    # Calculate precision, recall, and thresholds dynamically via scikit-learn
    # Note: len(precisions) == len(thresholds) + 1 (the last element corresponds to precision=1, recall=0)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    # Calculate F1-Score for each threshold using numpy vectorization
    # Suppress zero division warnings for thresholds where precision + recall = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
        # Replace resulting NaNs with 0 to keep the plot clean
        f1_scores = np.nan_to_num(f1_scores)

    # Initialize the plot figure
    plt.figure(figsize=(10, 6))

    # Plot the three metric curves against the thresholds array
    plt.plot(thresholds, precisions[:-1], label='Precision', color='#1f77b4', linestyle='--', linewidth=2)
    plt.plot(thresholds, recalls[:-1], label='Recall', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', color='#2ca02c', linewidth=3)

    # Automatically find and highlight the optimal threshold (Max F1-Score)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    plt.scatter(optimal_threshold, optimal_f1, color='red', s=80, zorder=5, 
                label=f'Max F1 ({optimal_f1:.3f} at Thr={optimal_threshold:.2f})')

    # Draw the default baseline threshold at 0.5 (standard for probability-based classifiers)
    plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, label='Default Threshold (0.5)')
    plt.xlim(0, 1)
    plt.xlabel("Classification Threshold (Probability Score)", fontsize=12)

    # Set plot aesthetics, labels, and limits
    plt.title(f"Sensitivity Analysis: Metrics vs Threshold ({dataset_type})", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Metric Score", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Customize the legend to appear outside on the right side
    legend = plt.legend(title='Metrics & Thresholds', loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1, 
                        frameon=True, framealpha=0.9, facecolor='white', edgecolor='black', title_fontsize=11)
    legend.get_title().set_fontweight('bold')

    # Save the plot with tight layout to prevent legend clipping
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Threshold sensitivity plot successfully saved to: {dst_path}")


def save_all_kde_plots(config_csv_path, features_to_plot, label_column=MLConstants.Y_LABEL):
    """
    Automated batch runner that reads a configuration CSV, loads each dataset,
    extracts metadata, and generates KDE plots to analyze feature distributions.
    This analysis is model-agnostic and fully supports any workflow using RF, DT, or HGB.

    :param config_csv_path: Path to the configuration CSV file containing 'path' and 'dataset_type'
    :param features_to_plot: List of feature column names to analyze and plot
    :param label_column: The name of the ground-truth target column in your datasets
    """
    config_path = Path(config_csv_path)
    if not config_path.exists():
        print(f"[ERROR] Configuration CSV file not found at: {config_csv_path}")
        return

    # Load the master configuration file
    try:
        df_config = pd.read_csv(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to read configuration CSV: {e}")
        return

    # Validate required columns in the configuration file
    if 'path' not in df_config.columns or 'dataset_type' not in df_config.columns:
        print("[ERROR] The configuration CSV must contain both 'path' and 'dataset_type' columns.")
        return

    # Ensure the main output directory for KDE plots exists
    base_kde_dir = Path(ProjectPaths.KDE_PLOTS_DIR)
    base_kde_dir.mkdir(parents=True, exist_ok=True)

    print("Starting automated batch KDE plotting...")
    
    # Iterate through each dataset defined in the config file
    for _, row in df_config.iterrows():
        dataset_path_str = row['path']
        dataset_type = row['dataset_type']
        
        if pd.isna(dataset_path_str) or pd.isna(dataset_type):
            continue
            
        file_path = Path(dataset_path_str)
        if not file_path.exists():
            print(f" -> [WARNING] Dataset file not found, skipping: {file_path}")
            continue

        # Automatically extract the dataset name from the file path
        dataset_name = file_path.stem

        # Define the final plot filename and destination path
        kde_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
        dst_path = base_kde_dir / kde_filename

        try:
            # Preview columns first to optimize memory allocation
            df_preview = pd.read_csv(file_path, nrows=1)
            valid_features = [f for f in features_to_plot if f in df_preview.columns]
            
            if not valid_features:
                print(f" -> [SKIP] No target features found in {dataset_name}")
                continue
                
            if label_column not in df_preview.columns:
                print(f" -> [SKIP] Label column '{label_column}' not found in {dataset_name}")
                continue
            
            # Load ONLY the required features and the label column to keep RAM safe
            cols_to_load = valid_features + [label_column]
            df_data = pd.read_csv(file_path, usecols=cols_to_load)
            
            X_test = df_data[valid_features]
            y_test = df_data[label_column]

            # Map numeric labels to descriptive strings for the legend
            labels = pd.Series(y_test).map({0: 'Normal Traffic', 1: 'Attack/Anomaly'}).fillna(y_test)
            
            # Determine dynamic grid layout
            n_features = len(valid_features)
            cols = min(2, n_features)
            rows = (n_features + 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            axes = [axes] if n_features == 1 else axes.flatten()

            # Cache handles and labels for the global figure legend
            global_handles, global_labels = [], []

            # Plot each valid feature
            for i, feature in enumerate(valid_features):
                ax = axes[i]
                
                sns.kdeplot(
                    data=X_test, x=feature, hue=labels, fill=True, alpha=0.3, 
                    common_norm=False, linewidth=2, palette={'Normal Traffic': '#1f77b4', 'Attack/Anomaly': '#ff7f0e'},
                    ax=ax, warn_singular=False, cut=0
                )
                
                # Subplot styling
                ax.set_title(f"Distribution of '{feature}'", fontsize=12, fontweight='bold')
                ax.set_xlabel(feature, fontsize=10)
                ax.set_ylabel("Density", fontsize=10)
                ax.set_yscale('symlog', linthresh=1e-5)
                ax.grid(True, linestyle=':', alpha=0.6)
                
                # Safely extract and cache the legend handles and labels from the first subplot
                if i == 0 and ax.get_legend() is not None:
                    sb_legend = ax.get_legend()
                    global_handles = list(getattr(sb_legend, 'legend_handles', getattr(sb_legend, 'legendHandles', [])))
                    global_labels = [t.get_text() for t in sb_legend.get_texts()]
                
                # Remove subplot legend to prevent duplicate clutter on individual axes
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

            # Clean up empty subplots if the grid has an odd number of features
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # Global figure formatting
            fig.suptitle(f"Feature Distribution Analysis ({dataset_type} - {dataset_name})", fontsize=16, fontweight='bold', y=1.02)
            
            # Create a single global external legend using the cached handles and labels
            if global_handles and global_labels:
                legend = fig.legend(
                    global_handles, global_labels, title='Traffic Class', loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1, 
                    frameon=True, framealpha=0.9, facecolor='white', edgecolor='black', title_fontsize=11
                )
                legend.get_title().set_fontweight('bold')

            # Save the clean plot
            plt.tight_layout()
            plt.savefig(dst_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f" -> [OK] Saved: {kde_filename}")

        except Exception as e:
            print(f" -> [ERROR] Failed to process {dataset_name}: {e}")
            
    print("\n[COMPLETED] All KDE plots have been generated and saved directly to the target directory.")


def save_shap_plot(model, X_test, y_test, model_name, dataset_type, dataset_name, dst_dir):
    """
    Generates a classic 1D SHAP summary (dot) plot for a given trained classifier 
    (compatible with RF, DT, and HGB) and its corresponding test matrix.

    :param model: Trained model object (RF, DT, or HGB)
    :param X_test: DataFrame containing test features
    :param y_test: Array or Series containing true class labels (0 or 1)
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    :param dst_dir: destination path directory
    """
    # Create the main shap directory
    shap_dir = create_directory(ProjectPaths.DIR_SHAP_PLOTS, dst_dir)

    # Create a directory for the model's SHAP plots
    model_shap_dir = create_directory(model_name, shap_dir)

    # Define the filename and output path for the SHAP summary plot
    shap_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
    
    # Define the full output path for the SHAP summary plot
    dst_path = model_shap_dir / shap_filename

    try:
        # Stratified sampling: Align labels with features
        y_series = pd.Series(y_test, index=X_test.index if hasattr(X_test, 'index') else None)
        
        # Calculate class quotas: e.g., 10:1 ratio on maximum samples
        n_anomaly = MLConstants.SHAP_MAX_SAMPLES // MLConstants.NORMAL_ANOMALY_RATIO
        n_normal = MLConstants.SHAP_MAX_SAMPLES - n_anomaly

        # Extract sampled indices for both classes, capping at available instances
        idx_normal = pd.Index([])
        if sum(y_series == 0) > 0:
            idx_normal = y_series[y_series == 0].sample(
                n=min(n_normal, sum(y_series == 0)), 
                random_state=MLConstants.MAIN_SEED
            ).index
            
        idx_anomaly = pd.Index([])
        if sum(y_series == 1) > 0:
            idx_anomaly = y_series[y_series == 1].sample(
                n=min(n_anomaly, sum(y_series == 1)), 
                random_state=MLConstants.MAIN_SEED
            ).index

        # Union of stratified indices
        sampled_idx = idx_normal.append(idx_anomaly)

        # Filter the original test set using the stratified indices (or fallback to standard random sampling)
        if not sampled_idx.empty:
            X_test_sampled = X_test.loc[sampled_idx]
        else:
            X_test_sampled = X_test.sample(
                n=min(MLConstants.SHAP_MAX_SAMPLES, len(X_test)), 
                random_state=MLConstants.MAIN_SEED
            )

        # Inizialize Explainer dynamically to support RF, DT, and HGB
        try:
            # First attempt: standard TreeExplainer optimized for RF and DT
            explainer = shap.TreeExplainer(
                model, 
                feature_perturbation="tree_path_dependent", 
                model_output="raw"
            )
        except Exception as explainer_err:
            # Fallback attempt: generic Explainer or simplified TreeExplainer (crucial for HGB compatibility)
            try:
                explainer = shap.TreeExplainer(model)
            except Exception:
                explainer = shap.Explainer(model, X_test_sampled)
        
        # Compute SHAP values with additivity check handling
        try:
            shap_values = explainer.shap_values(X_test_sampled, check_additivity=False)
        except TypeError:
            shap_values = explainer.shap_values(X_test_sampled)

        # Handle output disparities (e.g. Explainer objects vs raw ndarrays)
        # 1. If shap_values is a custom SHAP Explanation object, extract its raw values
        if hasattr(shap_values, "values"):
            shap_values_raw = shap_values.values
        else:
            shap_values_raw = shap_values

        # 2. Extract positive class ('Attack/Anomaly') values based on structural dimensions
        # Older/Standard SHAP versions return a list of arrays: [Normal_values, Attack_values]
        if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
            shap_values_raw = shap_values_raw[1]
        # Newer SHAP versions/HGB outputs might return a 3D array: (samples, features, classes)
        elif hasattr(shap_values_raw, 'shape') and len(shap_values_raw.shape) == 3:
            shap_values_raw = shap_values_raw[:, :, 1]
        # Explanation objects might contain 3D values: (samples, features, classes)
        elif hasattr(shap_values_raw, 'shape') and len(shap_values_raw.shape) == 2:
            # Already 2D (samples, features), no extraction needed
            pass

        # Generate and configure the dynamic plot
        n_features = X_test_sampled.shape[1]
        
        # Dynamically scale the plot height to comfortably accommodate all features
        plt.figure(figsize=(8, max(6, n_features * 0.4)))

        # Force the "dot" view to show individual feature impacts
        shap.summary_plot(
            shap_values_raw, 
            X_test_sampled, 
            show=False, 
            max_display=n_features, 
            plot_type="dot"
        )

        # Set typography and save the figure cleanly without clipping the axes
        plt.title("SHAP Feature Impact Analysis", fontsize=14, fontweight='bold', pad=15)
        plt.savefig(dst_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP summary successfully saved to: {dst_path}")
        
    except Exception as e:
        print(f"Warning: Could not generate SHAP plot for '{model_name}' ({e}). Skipping.")


def save_heatmap_for_metrics_plot(model_type, models, data, dst_dir):
    """
    Generates a performance heatmap for each evaluation metric across different trained models 
    and test datasets. Includes an average column and detaches the reference injection row.

    :param model_type: String representing the classifier type (e.g., 'rf', 'dt', 'hgb') to prefix row labels
    :param models: DataFrame containing model metadata (model_name, dataset_type, classes)
    :param data: DataFrame containing evaluation metrics for each model-dataset pair
    :param dst_dir: Destination path directory
    """
    heatmap_dir = create_directory(ProjectPaths.DIR_PERFORMANCE_PLOTS, dst_dir)

    # Prepare model metadata for merging
    models_prep = models[['model_name', 'dataset_type', 'classes']].rename(
        columns={'dataset_type': 'model_dataset_type', 'classes': 'model_classes'}
    )
    
    # Combine evaluation metrics with model metadata
    merged_data = pd.merge(data, models_prep, on='model_name', how='left')

    # Format axis labels for both datasets and models
    merged_data['dataset_label'] = [
        _build_clean_label(c, d) for c, d in zip(merged_data['classes'], merged_data['dataset_type'])
    ]
    
    merged_data['model_label'] = [
        _build_clean_label(c, d, prefix=f"{model_type.upper()} (") 
        for c, d in zip(merged_data['model_classes'], merged_data['model_dataset_type'])
    ]

    # Plot a heatmap for each active monitoring metric
    for feature in MLConstants.PLOTTING_METRICS:
        dst_path = heatmap_dir / f"{feature}_matrix{Naming.PLOT_EXT}"
    
        merged_data[feature] = pd.to_numeric(merged_data[feature], errors='coerce')
        pivot_data = merged_data.pivot(index='model_label', columns='dataset_label', values=feature)

        # Reorder rows based on PlotConfig templates
        if PlotConfig.HEATMAP_ROW_ORDER is not None:
            dynamic_row_order = [f"{model_type.upper()} {r}" for r in PlotConfig.HEATMAP_ROW_ORDER]
            valid_rows = [r for r in dynamic_row_order if r in pivot_data.index]
            remaining_rows = [r for r in pivot_data.index if r not in valid_rows]
            pivot_data = pivot_data.reindex(index=valid_rows + remaining_rows)

        # Reorder columns based on PlotConfig templates
        if PlotConfig.HEATMAP_COLUMN_ORDER is not None:
            valid_cols = [c for c in PlotConfig.HEATMAP_COLUMN_ORDER if c in pivot_data.columns]
            remaining_cols = [c for c in pivot_data.columns if c not in valid_cols]
            pivot_data = pivot_data.reindex(columns=valid_cols + remaining_cols)

        # Calculate row averages
        pivot_data['Average'] = pivot_data.mean(axis=1)
        
        # Insert an empty spacer column before the Average column
        cols = list(pivot_data.columns)
        cols.remove('Average')
        cols.append(' ')  # Single space serves as the column spacer label
        cols.append('Average')
        pivot_data[' '] = np.nan
        pivot_data = pivot_data[cols]

        # Locate the injection baseline row and insert an empty spacer row above it
        rows = list(pivot_data.index)
        ref_row_name = f"{model_type.upper()} (Aggregate injection)"
        
        if ref_row_name in rows:
            idx = rows.index(ref_row_name)
            rows.insert(idx, '  ')  # Double space serves as the row spacer label to avoid collisions
            pivot_data.loc['  '] = np.nan
            pivot_data = pivot_data.reindex(rows)
            
        # Dynamically scale canvas dimensions to maintain legible, square cells
        cell_size = 1.0  
        fig_width = max(len(pivot_data.columns) * cell_size + 6.0, 14)
        fig_height = max(len(pivot_data.index) * cell_size + 2.5, 8)

        plt.figure(figsize=(fig_width, fig_height))
        sns.set_theme(style="white") 
        
        # Mask NaN entries to create a visual gap for the spacers
        sns.heatmap(
            pivot_data, mask=pivot_data.isnull(), annot=True, fmt=".3f", 
            cmap="Blues", vmin=0.0, vmax=1.0, square=True,
            annot_kws={"size": 11, "weight": "bold"}, 
            cbar_kws={'label': f'{feature} Value', 'shrink': 0.6, 'pad': 0.03}
        )
        
        plt.title(f"{feature} Performance Matrix", pad=25, fontsize=18, fontweight='bold')
        plt.ylabel("Trained Models", fontsize=14, fontweight='bold', labelpad=15)
        plt.xlabel("Testing Datasets & Classes", fontsize=14, fontweight='bold', labelpad=15)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=11)
        
        plt.savefig(dst_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{feature} Heatmap successfully saved to: {dst_path}")


def save_aggregated_metrics_heatmap(model_type, models, data, dst_dir):
    """
    Generates an aggregated performance heatmap for each evaluation metric 
    across different trained models and test datasets over multiple seeds.
    Calculates both Mean and Variance, saves them to distinct CSVs, and plots 
    a "super heatmap" showing the mean as primary text and variance below it.

    :param model_type: String representing the classifier type (e.g., 'rf', 'dt', 'hgb')
    :param models: DataFrame containing model metadata (model_name, dataset_type, classes)
    :param data: Combined DataFrame containing evaluation metrics across all seeds
    :param dst_dir: Destination path directory
    """
    # Create the main performance plots directory inside destination
    heatmap_dir = create_directory(ProjectPaths.DIR_PERFORMANCE_PLOTS, dst_dir)

    # Prepare model metadata for merging
    models_prep = models[['model_name', 'dataset_type', 'classes']].rename(
        columns={'dataset_type': 'model_dataset_type', 'classes': 'model_classes'}
    )
    
    # Combine evaluation metrics with model metadata
    merged_data = pd.merge(data, models_prep, on='model_name', how='left')

    # Format axis labels for both datasets and models
    merged_data['dataset_label'] = [
        _build_clean_label(c, d) for c, d in zip(merged_data['classes'], merged_data['dataset_type'])
    ]
    
    merged_data['model_label'] = [
        _build_clean_label(c, d, prefix=f"{model_type.upper()} (") 
        for c, d in zip(merged_data['model_classes'], merged_data['model_dataset_type'])
    ]

    # Plot an aggregated heatmap for each active monitoring metric
    for feature in MLConstants.PLOTTING_METRICS:
        dst_path = heatmap_dir / f"{feature}_aggregated_matrix{Naming.PLOT_EXT}"
        mean_csv_path = heatmap_dir / f"{feature}_mean_matrix.csv"
        var_csv_path = heatmap_dir / f"{feature}_variance_matrix.csv"
    
        merged_data[feature] = pd.to_numeric(merged_data[feature], errors='coerce')
        
        # Aggregate data: compute Mean and Variance using pivot_table
        mean_pivot = merged_data.pivot_table(index='model_label', columns='dataset_label', values=feature, aggfunc='mean')
        var_pivot = merged_data.pivot_table(index='model_label', columns='dataset_label', values=feature, aggfunc='var').fillna(0.0)

        # Reorder rows based on PlotConfig templates
        if PlotConfig.HEATMAP_ROW_ORDER is not None:
            dynamic_row_order = [f"{model_type.upper()} {r}" for r in PlotConfig.HEATMAP_ROW_ORDER]
            valid_rows = [r for r in dynamic_row_order if r in mean_pivot.index]
            remaining_rows = [r for r in mean_pivot.index if r not in valid_rows]
            
            mean_pivot = mean_pivot.reindex(index=valid_rows + remaining_rows)
            var_pivot = var_pivot.reindex(index=valid_rows + remaining_rows)

        # Reorder columns based on PlotConfig templates
        if PlotConfig.HEATMAP_COLUMN_ORDER is not None:
            valid_cols = [c for c in PlotConfig.HEATMAP_COLUMN_ORDER if c in mean_pivot.columns]
            remaining_cols = [c for c in mean_pivot.columns if c not in valid_cols]
            
            mean_pivot = mean_pivot.reindex(columns=valid_cols + remaining_cols)
            var_pivot = var_pivot.reindex(columns=valid_cols + remaining_cols)

        # Calculate row averages
        mean_pivot['Average'] = mean_pivot.mean(axis=1)
        var_pivot['Average'] = var_pivot.mean(axis=1)
        
        # Insert an empty spacer column before the Average column
        cols = list(mean_pivot.columns)
        cols.remove('Average')
        cols.append(' ')  # Single space serves as the column spacer label
        cols.append('Average')
        
        mean_pivot[' '] = np.nan
        var_pivot[' '] = np.nan
        
        mean_pivot = mean_pivot[cols]
        var_pivot = var_pivot[cols]

        # Locate the injection baseline row and insert an empty spacer row above it
        rows = list(mean_pivot.index)
        ref_row_name = f"{model_type.upper()} (Aggregate injection)"
        
        if ref_row_name in rows:
            idx = rows.index(ref_row_name)
            rows.insert(idx, '  ')  # Double space serves as the row spacer label
            
            mean_pivot.loc['  '] = np.nan
            var_pivot.loc['  '] = np.nan
            
            mean_pivot = mean_pivot.reindex(rows)
            var_pivot = var_pivot.reindex(rows)
            
        # Save the Mean and Variance matrices to CSV files
        mean_pivot.to_csv(mean_csv_path)
        var_pivot.to_csv(var_csv_path)
        
        # Dynamically scale canvas dimensions
        cell_size = 1.0  
        fig_width = max(len(mean_pivot.columns) * cell_size + 6.0, 14)
        fig_height = max(len(mean_pivot.index) * cell_size + 2.5, 8)

        # Set up standard white background aesthetic for the plot
        sns.set_theme(style="white") 
        plt.figure(figsize=(fig_width, fig_height))
        
        # Generate the heatmap base without standard annotations
        ax = sns.heatmap(
            mean_pivot, mask=mean_pivot.isnull(), annot=False, 
            cmap="Blues", vmin=0.0, vmax=1.0, square=True,
            cbar_kws={'label': f'{feature} Mean Value', 'shrink': 0.6, 'pad': 0.03}
        )
        
        # Manually inject text into each cell to support different font sizes for mean and variance
        for i in range(mean_pivot.shape[0]):
            for j in range(mean_pivot.shape[1]):
                val_mean = mean_pivot.iloc[i, j]
                val_var = var_pivot.iloc[i, j]
                
                # Verify that the target cell is not empty (spacers)
                if not pd.isna(val_mean):
                    # Maintain optimal contrast based on background lightness
                    text_color = "white" if val_mean > 0.65 else "black"
                    
                    # Add Mean value as the dominant centered label
                    ax.text(j + 0.5, i + 0.40, f"{val_mean:.3f}", 
                            ha='center', va='center', fontsize=11, weight='bold', color=text_color)
                    
                    # Add Variance value below the mean using a smaller and softer font
                    ax.text(j + 0.5, i + 0.75, f"var: {val_var:.4f}", 
                            ha='center', va='center', fontsize=8, color=text_color, alpha=0.9)
        
        # Apply structured titles and labeling styles
        plt.title(f"{feature} Aggregated Matrix (Mean & Variance over {len(MLConstants.SEEDS)} seeds)", 
                  pad=25, fontsize=14, fontweight='bold')
        plt.ylabel("Trained Models", fontsize=12, fontweight='bold', labelpad=15)
        plt.xlabel("Testing Datasets & Classes", fontsize=12, fontweight='bold', labelpad=15)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=11)
        
        # Save the finalized figure using maximum quality
        plt.savefig(dst_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Aggregated heatmap successfully saved to: {dst_path}")


if __name__ == "__main__":
    pass
