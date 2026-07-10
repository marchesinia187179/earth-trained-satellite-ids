"""
Heatmap generation functions for visualizing model performance across different datasets and features.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .utils.file_utils import create_directory
from .utils.config import MLConstants, Naming, PlotConfig, ProjectPaths
from sklearn.metrics import precision_recall_curve, average_precision_score


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
def save_feature_importances_plot(model, model_name, feature_names, dst_dir):
    """
    Plots the MDI (Mean Decrease in Impurity) feature importances for a Random Forest model.
    Features are sorted in descending order of importance to ensure clear visual analysis.

    :param model: the trained Random Forest model object
    :param model_name: standardized name for the trained model
    :param feature_names: list of feature names corresponding to the importances
    :param dst_dir: destination directory to save the feature importance plot
    """
    # Get feature importances
    importances = model.feature_importances_

    # Calculate standard deviation of feature importances across all trees in the Random Forest
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    # Plot feature importance
    dst_path = dst_dir / f"{model_name}{Naming.PLOT_EXT}"

    # Align and sort in one step
    df = pd.DataFrame({'importance': importances, 'std': std}, index=feature_names)\
           .sort_values(by='importance', ascending=False)

    # Create a bar plot with error bars representing the standard deviation of feature importances
    plt.figure(figsize=(10, 6))
    
    # Plot the feature importances as a bar chart with error bars
    df['importance'].plot.bar(
        yerr=df['std'], capsize=4, color='skyblue', edgecolor='black'
    )
    
    # Set plot aesthetics and labels
    plt.title("Feature Importances via Mean Decrease in Impurity (MDI)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Mean Decrease in Impurity", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot to the specified destination path with high resolution and tight layout
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature importance plot successfully saved to: {dst_path}")


def save_pca_plot(X, y, dataset_type, dataset_name, precomputed_scaler=None, precomputed_pca=None):
    """
    Applies PCA (2 components) on the feature matrix X and plots a 2D scatter plot.
    Supports both INDEPENDENT (computes fit on X) and CROSS_DOMAIN (uses precomputed scaler/pca) methods.

    :param X: feature matrix (DataFrame or numpy array)
    :param y: class labels (Series or array-like)
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    :param precomputed_scaler: Fitted StandardScaler instance (optional, for CROSS_DOMAIN)
    :param precomputed_pca: Fitted PCA instance (optional, for CROSS_DOMAIN)
    """
    # Create the name of the file to be saved
    file_name = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"

    # --- CHOOSE PARADIGM: CROSS_DOMAIN vs INDEPENDENT ---
    if precomputed_scaler is not None and precomputed_pca is not None:
        # CROSS_DOMAIN Method: Project data onto the pre-existing baseline space (Only transform)
        X_scaled = precomputed_scaler.transform(X)
        X_pca = precomputed_pca.transform(X_scaled)
        current_pca = precomputed_pca  # Use the reference PCA for variance ratios

        dst_path = ProjectPaths.PCA_PLOTS_CROSS_DOMAIN_DIR / file_name
    else:
        # INDEPENDENT Method: Compute a brand new space tailored to this specific dataset (Fit + Transform)
        current_scaler = StandardScaler()
        current_pca = PCA(n_components=MLConstants.PCA_COMPONENTS, random_state=MLConstants.RANDOM_STATE)
        X_scaled = current_scaler.fit_transform(X)
        X_pca = current_pca.fit_transform(X_scaled)

        dst_path = ProjectPaths.PCA_PLOTS_INDEPENDENT_DIR / file_name
    
    # Map the class labels to human-readable strings for plotting
    labels = y.map({0: 'Normal Traffic', 1: 'Attack/Anomaly'}).fillna(y)
    var = current_pca.explained_variance_ratio_ * 100
    
    # Configure the scatter plot with appropriate aesthetics
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, alpha=0.5,
        palette={'Normal Traffic': '#1f77b4', 'Attack/Anomaly': '#ff7f0e'}
    )
    
    # Set plot titles, labels, and grid for better readability
    plt.title(f"PCA 2D Projection (Total Variance Explained: {var.sum():.2f}%)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f"PC1 ({var[0]:.2f}% Variance)", fontsize=12)
    plt.ylabel(f"PC2 ({var[1]:.2f}% Variance)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Add a legend with a title and customize its appearance
    legend = ax.legend(title='Target Traffic Class', loc='upper right', frameon=True, 
                       framealpha=0.9, facecolor='white', edgecolor='black', title_fontsize=11)
    legend.get_title().set_fontweight('bold')
    
    # Save the PCA plot to the specified destination path
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PCA plot successfully saved to: {dst_path}")


def save_probability_plot(y_test, y_scores, model_name, dataset_type, dataset_name):
    """
    Plots the overlapping probability distributions for Normal traffic vs Attacks
    to visually demonstrate the classification threshold trap.

    :param y_test: true class labels (0 for Normal, 1 for Attack)
    :param y_scores: predicted probabilities for the positive class (Attack)
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    """
    # Create a directory for the model's probability plots
    model_prob_dir = create_directory(model_name, ProjectPaths.PROB_PLOTS_DIR)

    # Define the filename and output path for the probability distribution plot
    prob_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
    
    # Define the full output path for the probability distribution plot
    dst_path = model_prob_dir / prob_filename

    # Map binary numeric labels to descriptive strings for the plot legend.
    # Wrapping in pd.Series ensures safety and compatibility with both NumPy arrays and Pandas Series.
    labels = pd.Series(y_test).map({0: 'Normal Traffic', 1: 'Attack/Anomaly'}).fillna(y_test)
    
    # Initialize the plot figure with specific dimensions
    plt.figure(figsize=(10, 6))
    
    # Generate overlapping density histograms directly from the arrays.
    # Using 'step' and 'density' allows comparing distributions regardless of dataset class imbalance.
    ax = sns.histplot(
        x=y_scores, hue=labels, element='step', stat='density', common_norm=False, 
        alpha=0.4, bins=50, palette={'Normal Traffic': '#1f77b4', 'Attack/Anomaly': '#ff7f0e'}
    )

    # Get dynamic range (Probability vs Anomaly Score)
    is_probability = (np.min(y_scores) >= 0.0) and (np.max(y_scores) <= 1.0)

    if is_probability:
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Standard Threshold (0.5)')
        plt.xlabel("Predicted Probability of Anomaly (Output of predict_proba)", fontsize=12)
        plt.xlim(0, 1)
    else:
        plt.axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.0)')
        plt.xlabel("Anomaly Score (Output of decision_function)", fontsize=12)
        
    plt.title("Classification Score Distribution (Threshold Analysis)", fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    plt.ylabel("Density", fontsize=12)
    
    # Customize the legend aesthetics for a clean and professional layout
    legend = ax.legend(title='Traffic Class / Threshold', loc='upper right', frameon=True, 
                       framealpha=0.9, facecolor='white', edgecolor='black', title_fontsize=11)
    legend.get_title().set_fontweight('bold')
    
    # Save the figure dynamically fitting all outer elements without clipping, then free up memory
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Output console feedback
    print(f"Probability distribution plot successfully saved to: {dst_path}")


def save_pr_curve_plot(y_test, y_scores, model_name, dataset_type, dataset_name):
    """
    Plots the Precision-Recall (PR) Curve to evaluate model performance, 
    especially useful for highly imbalanced datasets (e.g., Anomaly Detection).

    :param y_test: true class labels (0 for Normal, 1 for Attack)
    :param y_scores: predicted probabilities or anomaly scores for the positive class
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    """
    # Guard clause: Cannot compute PR curve if there are no positive samples (Attacks)
    # This prevents errors when evaluating on purely "Normal" datasets
    if np.sum(y_test) == 0:
        print(f"Skipping PR Curve for {dataset_name}: No attack/anomaly samples present.")
        return

    # Create a directory for the model's PR Curve plots
    model_pr_dir = create_directory(model_name, ProjectPaths.PR_CURVE_PLOTS_DIR)

    # Define the filename and output path
    pr_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
    dst_path = model_pr_dir / pr_filename

    # Calculate Precision, Recall and the Average Precision (AP) score
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
    legend = plt.legend(title='Classifier Performance', loc='upper right', frameon=True, 
                        framealpha=0.9, facecolor='white', edgecolor='black', title_fontsize=11)
    legend.get_title().set_fontweight('bold')

    # Save the plot
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Precision-Recall Curve successfully saved to: {dst_path}")


def save_threshold_metrics_plot(y_test, y_scores, model_name, dataset_type, dataset_name):
    """
    Plots Precision, Recall, and F1-Score as a function of the decision threshold.
    Highlights the threshold that maximizes the F1-Score to demonstrate domain shift adaptations.

    :param y_test: true class labels (0 for Normal, 1 for Attack)
    :param y_scores: predicted probabilities or anomaly scores for the positive class
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    """
    # Guard clause: Skip if no attacks exist in the ground truth
    if np.sum(y_test) == 0:
        print(f"Skipping Threshold Plot for {dataset_name}: No attack/anomaly samples present.")
        return

    # Create a directory for the model's Threshold Sensitivity plots
    model_thr_dir = create_directory(model_name, ProjectPaths.THRESHOLD_PLOTS_DIR)

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

    # Detect dynamic range (Probability vs Anomaly Score) to draw the correct default baseline
    is_probability = (np.min(y_scores) >= 0.0) and (np.max(y_scores) <= 1.0)

    if is_probability:
        plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, label='Default Threshold (0.5)')
        plt.xlim(0, 1)
        plt.xlabel("Classification Threshold (Probability Score)", fontsize=12)
    else:
        plt.axvline(x=0.0, color='gray', linestyle=':', linewidth=2, label='Default Threshold (0.0)')
        plt.xlabel("Classification Threshold (Anomaly Score)", fontsize=12)

    # Set plot aesthetics, labels, and limits
    plt.title(f"Sensitivity Analysis: Metrics vs Threshold ({dataset_type})", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Metric Score", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)

    # Customize the legend to appear outside at the bottom so it doesn't cover the lines
    legend = plt.legend(title='Metrics & Thresholds', loc='lower center', bbox_to_anchor=(0.5, -0.25), 
                        ncol=3, frameon=True, framealpha=0.9, facecolor='white', edgecolor='black', title_fontsize=11)
    legend.get_title().set_fontweight('bold')

    # Save the plot with extra bottom margin to accommodate the external legend
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Threshold sensitivity plot successfully saved to: {dst_path}")


def save_feature_kde_plot(X_test, y_test, model_name, dataset_type, dataset_name, features_to_plot):
    """
    Plots the Kernel Density Estimation (KDE) for a selected subset of top features.
    This helps visually detect 'Concept Drift' across different domains by showing
    how feature distributions change for Normal vs Attack traffic.

    :param X_test: DataFrame containing test features
    :param y_test: true class labels (0 for Normal, 1 for Attack)
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    :param features_to_plot: list of feature column names to plot
    """
    # Create a directory for the model's KDE plots
    model_kde_dir = create_directory(model_name, ProjectPaths.KDE_PLOTS_DIR)

    # Define the filename and output path
    kde_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
    dst_path = model_kde_dir / kde_filename

    # Filter out features that might not exist in the current X_test DataFrame
    valid_features = [f for f in features_to_plot if f in X_test.columns]
    
    if not valid_features:
        print(f"Skipping KDE Plot for {dataset_name}: No valid features found to plot.")
        return

    # Map numeric labels to descriptive strings for the legend
    labels = pd.Series(y_test, index=X_test.index).map({0: 'Normal Traffic', 1: 'Attack/Anomaly'})
    
    # Determine grid size (e.g., up to 4 features, plotted in a 2x2 grid or 1x3 row)
    n_features = len(valid_features)
    cols = min(2, n_features)
    rows = (n_features + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    
    # Flatten axes array for easy iteration, even if it's 1D or a single plot
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, feature in enumerate(valid_features):
        ax = axes[i]
        
        # Plot KDE with overlapping filled areas. 
        # common_norm=False ensures both classes are visible even with 10:1 imbalance
        sns.kdeplot(
            data=X_test, x=feature, hue=labels, fill=True, alpha=0.3, 
            common_norm=False, linewidth=2, palette={'Normal Traffic': '#1f77b4', 'Attack/Anomaly': '#ff7f0e'},
            ax=ax, warn_singular=False
        )
        
        # Subplot aesthetics
        ax.set_title(f"Distribution of '{feature}'", fontsize=12, fontweight='bold')
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Remove the internal legend created by seaborn to use a single global one later
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Hide any unused subplots (if features count is odd)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a global title
    fig.suptitle(f"Feature Distribution Analysis ({dataset_type})", fontsize=16, fontweight='bold', y=1.02)
    
    # Create a single global legend for the entire figure
    handles, labels_list = ax.get_legend_handles_labels()
    fig.legend(handles, labels_list, title='Traffic Class', loc='upper right', 
               bbox_to_anchor=(1.0, 1.0), frameon=True, framealpha=0.9, 
               facecolor='white', edgecolor='black', title_fontsize=11)

    # Save the plot cleanly
    plt.tight_layout()
    plt.savefig(dst_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"KDE Density plot successfully saved to: {dst_path}")


def save_shap_plot(model, X_test, y_test, model_name, dataset_type, dataset_name):
    """
    Generates a classic 1D SHAP summary (dot) plot for a given model and test matrix.

    :param model: Trained model object compatible with shap.TreeExplainer
    :param X_test: DataFrame containing test features
    :param y_test: Array or Series containing true class labels (0 or 1)
    :param model_name: name of the evaluated model
    :param dataset_type: type of the dataset being used
    :param dataset_name: name of the dataset being used
    """
    # Create a directory for the model's SHAP plots
    model_shap_dir = create_directory(model_name, ProjectPaths.SHAP_PLOTS_DIR)

    # Define the filename and output path for the SHAP summary plot
    shap_filename = f"{dataset_type.lower()}_{dataset_name}{Naming.PLOT_EXT}"
    
    # Define the full output path for the SHAP summary plot
    dst_path = model_shap_dir / shap_filename

    try:
        # Stratified sampling: Align labels with features
        y_series = pd.Series(y_test, index=X_test.index if hasattr(X_test, 'index') else None)
        
        # --- MODIFICA SPLIT (es. 10:1 su 500 records) ---
        # Calcola le quote: 500 // 10 = 50 Anomaly, il resto (450) è Normal
        n_anomaly = MLConstants.SHAP_MAX_SAMPLES // MLConstants.NORMAL_ANOMALY_RATIO
        n_normal = MLConstants.SHAP_MAX_SAMPLES - n_anomaly

        # Extract sampled indices for both classes, capping at available instances
        idx_normal = pd.Index([])
        if sum(y_series == 0) > 0:
            idx_normal = y_series[y_series == 0].sample(
                n=min(n_normal, sum(y_series == 0)), 
                random_state=MLConstants.RANDOM_STATE
            ).index
            
        idx_anomaly = pd.Index([])
        if sum(y_series == 1) > 0:
            idx_anomaly = y_series[y_series == 1].sample(
                n=min(n_anomaly, sum(y_series == 1)), 
                random_state=MLConstants.RANDOM_STATE
            ).index

        # Unione degli indici
        sampled_idx = idx_normal.append(idx_anomaly)
        # --- FINE MODIFICA ---

        # Filter the original test set using the stratified indices (or fallback to standard random sampling)
        if not sampled_idx.empty:
            X_test_sampled = X_test.loc[sampled_idx]
        else:
            X_test_sampled = X_test.sample(n=min(MLConstants.SHAP_MAX_SAMPLES, len(X_test)), random_state=MLConstants.RANDOM_STATE)

        # Compute SHAP values using the TreeExplainer
        explainer = shap.TreeExplainer(
            model, 
            feature_perturbation="tree_path_dependent", 
            model_output="raw"
        )
        
        try:
            shap_values = explainer.shap_values(X_test_sampled, check_additivity=False)
        except TypeError:
            shap_values = explainer.shap_values(X_test_sampled)

        # Handle SHAP version disparities to safely extract the "Attack" (positive) class values
        # Older SHAP versions return a list of arrays: [Normal_values, Attack_values]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        # Newer SHAP versions return a 3D array: (samples, features, classes)
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        # Generate and configure the dynamic plot
        import matplotlib.pyplot as plt
        n_features = X_test_sampled.shape[1]
        
        # Dynamically scale the plot height to comfortably accommodate all features
        plt.figure(figsize=(8, max(6, n_features * 0.4)))

        # Force the "dot" view to show individual feature impacts
        shap.summary_plot(shap_values, X_test_sampled, show=False, max_display=n_features, plot_type="dot")

        # Set typography and save the figure cleanly without clipping the axes
        plt.title("SHAP Feature Impact Analysis", fontsize=14, fontweight='bold', pad=15)
        plt.savefig(dst_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP summary successfully saved to: {dst_path}")
        
    except Exception as e:
        print(f"Warning: Could not generate SHAP plot ({e}). Skipping.")


def plot_heatmap_for_metrics(models, data):
    """
    Generates a heatmap for a specific evaluation metric across different trained models and test datasets.

    :param models: DataFrame containing model metadata (model_name, dataset_type, classes)
    :param data: DataFrame containing evaluation metrics for each model-dataset pair
    """
    dst_dir = create_directory(ProjectPaths.DIR_PERFORMANCE, ProjectPaths.RESULTS_PLOT_DIR)

    # --- Merge model metadata with evaluation metrics to create a comprehensive DataFrame ---
    # Prepare a DataFrame with model metadata for merging
    models_prep = models[['model_name', 'dataset_type', 'classes']].rename(
        columns={'dataset_type': 'model_dataset_type', 'classes': 'model_classes'}
    )
    
    # Merge the evaluation metrics data with the model metadata on 'model_name'
    merged_data = pd.merge(data, models_prep, on='model_name', how='left')

    # Generate clean labels for both datasets and models to be used in the heatmap axes
    merged_data['dataset_label'] = [
        _build_clean_label(c, d) for c, d in zip(merged_data['classes'], merged_data['dataset_type'])
    ]
    merged_data['model_label'] = [
        _build_clean_label(c, d, prefix=f"{m.split('_')[0].upper()} (") 
        for c, d, m in zip(merged_data['model_classes'], merged_data['model_dataset_type'], merged_data['model_name'])
    ]

    # --- Generate heatmaps for each evaluation metric defined in MLConstants.EVALUATION_METRICS ---
    for feature in MLConstants.PLOTTING_METRICS:
        # Define the destination path for the heatmap plot
        dst_path = dst_dir / f"{feature}_matrix{Naming.PLOT_EXT}"
    
        # This ensures that the heatmap can be generated without errors due to non-numeric values
        merged_data[feature] = pd.to_numeric(merged_data[feature], errors='coerce')
        
        # Create a pivot table with models as rows, datasets as columns, and the metric values as the cell values
        pivot_data = merged_data.pivot(index='model_label', columns='dataset_label', values=feature)

        # If specific row or column orders are defined in the configuration, apply them to the pivot table
        if PlotConfig.HEATMAP_ROW_ORDER is not None:
            valid_rows = [r for r in PlotConfig.HEATMAP_ROW_ORDER if r in pivot_data.index]
            remaining_rows = [r for r in pivot_data.index if r not in valid_rows]
            pivot_data = pivot_data.reindex(index=valid_rows + remaining_rows)

        if PlotConfig.HEATMAP_COLUMN_ORDER is not None:
            valid_cols = [c for c in PlotConfig.HEATMAP_COLUMN_ORDER if c in pivot_data.columns]
            remaining_cols = [c for c in pivot_data.columns if c not in valid_cols]
            pivot_data = pivot_data.reindex(columns=valid_cols + remaining_cols)

        # Dynamically calculate figure size based on the number of rows and columns to ensure readability
        cell_size = 1.0  
        fig_width = max(len(pivot_data.columns) * cell_size + 6.0, 14)
        fig_height = max(len(pivot_data.index) * cell_size + 2.5, 8)

        # Configurazione ed esportazione del Plot con Seaborn
        # Configures and exports the heatmap using Seaborn
        plt.figure(figsize=(fig_width, fig_height))
        sns.set_theme(style="white") 
        
        sns.heatmap(
            pivot_data, annot=True, fmt=".3f", cmap="Blues", vmin=0.0, vmax=1.0, square=True,
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


if __name__ == "__main__":
    pass
