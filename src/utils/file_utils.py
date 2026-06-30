"""
Utility functions for handling CSV file operations with update and append capabilities.
"""

import pandas as pd
import pathlib
import sys

from datetime import datetime
from utils.config import MLConstants, Naming, ProjectPaths


# --- Public Functions ---
def update_or_append_csv(file_path, data_dict, match_keys, id_column='id'):
    """
    Updates an existing row in a CSV file if specified matching keys match,
    otherwise generates a new incremental ID and appends the record.
    
    Ensures that the primary identifier column is always pinned as the first 
    column in the CSV layout for structural consistency.

    :param file_path: pathlib.Path object pointing to the target CSV file
    :param data_dict: dict containing the key-value pairs to write
    :param match_keys: list of strings representing keys to identify duplicates
    :param id_column: string name of the unique identifier column (default: 'id')
    """
    # Check if the target file exists to look for an existing record to update
    if file_path.exists():
        try:
            existing_data = get_data_from_csv(file_path)
            
            if not existing_data.empty:
                # Build a boolean mask to locate an existing row matching the keys
                matching_mask = True
                for key in match_keys:
                    matching_mask &= (existing_data[key] == data_dict[key])
                
                # Scenario 1: Existing match found -> Perform an in-place update
                if matching_mask.any():
                    target_index = existing_data.index[matching_mask][0]
                    
                    # Preserve the original ID for the record
                    data_dict[id_column] = existing_data.at[target_index, id_column]
                    
                    # Overwrite old values with the new data dictionary fields
                    for key, value in data_dict.items():
                        existing_data.at[target_index, key] = value
                        
                    # Enforce structural integrity by keeping the ID column first
                    reordered_columns = [id_column] + [col for col in existing_data.columns if col != id_column]
                    existing_data = existing_data[reordered_columns]
                    
                    # Rewrite the full dataset to save the updated values and exit
                    existing_data.to_csv(file_path, index=False, na_rep='None')
                    print(f"Updated existing entry in: {file_path.name}")
                    return
                
                # Scenario 2: File exists but no match -> Compute next incremental ID
                highest_id = existing_data[id_column].max()
                data_dict[id_column] = int(highest_id + 1) if pd.notnull(highest_id) else 1
            else:
                # Fallback if the file exists but contains no data entries
                data_dict[id_column] = 1
        except Exception as error:
            print(f"Warning: Could not parse {file_path.name} for update ({error}). Appending as new.")
            data_dict[id_column] = 1
    else:
        # Scenario 3: File does not exist yet -> Initialize the ID tracker
        data_dict[id_column] = 1

    # Convert the dictionary into a standalone single-row DataFrame
    new_record = pd.DataFrame([data_dict])
    
    # Enforce ID positioning for the new entry prior to writing operations
    column_order = [id_column] + [col for col in new_record.columns if col != id_column]
    new_record = new_record[column_order]

    if not file_path.exists():
        # Fresh initialization: Write new file including structural headers
        new_record.to_csv(file_path, index=False, na_rep='None')
    else:
        # Schema Alignment: Match layout of existing dataset to prevent appending mismatched column structures
        if 'existing_data' in locals() and not existing_data.empty:
            target_columns = [id_column] + [col for col in existing_data.columns if col != id_column]
            
            # Fill missing schema properties with nulls if the dictionary was missing columns
            for column in target_columns:
                if column not in new_record.columns:
                    new_record[column] = None
            new_record = new_record[target_columns]
        
        # Safe Append: Add data without overwriting the header file metadata layout
        new_record.to_csv(file_path, mode='a', header=False, index=False, na_rep='None')
        
    print(f"Data appended/created in: {file_path.name}")


def get_data_from_csv(file_path):
    """
    Reads a CSV file into a pandas DataFrame with specific null-value parsing.
    
    Terminates script execution immediately if the file cannot be found,
    accessed, or properly parsed by pandas.

    :param file_path: pathlib.Path or str pointing to the CSV file to be loaded
    :return: pd.DataFrame containing the parsed dataset data
    """
    try:
        # Log the file operation using the filename only for cleaner output
        print(f"Loading CSV: {file_path.name}")
        
        # low_memory=False: Prevents type-guessing warnings on large columns
        # na_values='None': Explicitly treats 'None' string occurrences as true NaN/null values
        return pd.read_csv(file_path, low_memory=False, na_values='None')
        
    except Exception as error:
        # Catch any I/O, parsing, or OS-level file access errors
        print(f"Error: Could not read file at {file_path}. Details: {error}")
        
        # Hard exit to prevent subsequent processing steps from failing on missing data
        sys.exit(1)


def create_csv_from_data(data, file_name, file_path):
    """
    Converts raw data structures into a pandas DataFrame and exports it to a CSV file.

    Handles explicit filename extension checks and constructs a standard absolute 
    destination path before performing disk serialization.

    :param data: Input data structure (can be a dictionary, list of records, or DataFrame)
    :param file_name: Target name of the output file (with or without the '.csv' extension)
    :param file_path: Parent directory path where the exported file will be saved
    :return: pathlib.Path object representing the complete path to the generated CSV file
    """
    # Safely convert the raw data input into a standardized pandas DataFrame
    output_dataframe = pd.DataFrame(data)
    
    # Guarantee the filename carries the mandatory file extension suffix
    if not str(file_name).endswith('.csv'):
        file_name = f"{file_name}.csv"
    
    # Build the cross-platform destination path using pathlib operator joining
    absolute_path = pathlib.Path(file_path) / file_name
    
    # Export data matrix with explicit UTF-8 text encoding and standard 'None' missing markers
    output_dataframe.to_csv(absolute_path, index=False, encoding='utf-8', na_rep='None')
    print(f"Created file: {absolute_path.name}")

    return absolute_path


def add_file_info_to_datasets_info(file_path, dataset_type):
    """
    Extracts high-level statistical metadata from a specific dataset file and logs 
    it into a centralized tracking inventory CSV.

    Computes basic characteristics like matrix dimensions (rows and columns) along 
    with domain-specific information like target class frequencies and train-test split 
    cross-tabulation distributions where applicable.

    :param file_path: pathlib.Path object indicating the source dataset to be analyzed
    :param dataset_type: string representing the dataset type (e.g., 'nb15', 'sat20')
    """
    # Load the targeted dataset into memory to parse its schema and internal matrix shape
    dataset_content = get_data_from_csv(file_path)

    # Compile the foundational metadata payload using file structural dimensions
    dataset_metadata = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_type': dataset_type,
        'filename': file_path.name,
        'path': str(file_path.relative_to(ProjectPaths.ROOT)), # Standardize path relative to project workspace
        'rows': dataset_content.shape[0],
        'columns': dataset_content.shape[1],
    }

    # Metric 1: Compute target label frequencies to check for class imbalances
    if 'class' in dataset_content.columns:
        class_frequencies = dataset_content['class'].value_counts()
        
        # Convert dictionary to a string and sanitize it by removing spaces/commas 
        # to guarantee safe cell-parsing when embedded inside the final global tracking CSV
        dataset_metadata['class_distribution'] = str(class_frequencies.to_dict()).replace(" ", "_").replace(",", "_")
    else:
        dataset_metadata['class_distribution'] = 'None'
        
    # Metric 2: Compute cross-tabulation distribution matrices across train/test splits
    if 'class' in dataset_content.columns and 'split_type' in dataset_content.columns:
        # Construct a pivot-style count distribution grouping elements by class vs split assignment
        distribution_pivot = dataset_content.groupby(['class', 'split_type']).size().unstack(fill_value=0)
        dataset_metadata['train_test_distribution'] = str(distribution_pivot.to_dict())
    else:
        dataset_metadata['train_test_distribution'] = 'None'

    # Save or update the record in the global log tracking repository using relative path matching
    update_or_append_csv(ProjectPaths.DATASETS_INFO, dataset_metadata, ['path'], id_column='id')


def group_datasets_paths_for_filename_list(src_path, dst_path, filename_list):
    """
    Filters a master dataset inventory to aggregate path information for a specific
    list of target configuration files, saving the consolidated references to a secondary log.

    This utility helps bundle cross-domain or single-domain paths required for batch processing
    by extracting metadata traces from a central catalog based on type and filename matches.

    :param src_path: pathlib.Path pointing to the source master inventory CSV
    :param dst_path: pathlib.Path pointing to the output aggregated database CSV
    :param filename_list: list of dictionaries, where each entry must contain 'dataset_type' and 'filename'
    """
    print("Group datasets...")

    # Load the master tracking database containing all registered system paths
    source_datasets = get_data_from_csv(src_path)
    
    # Process each requested file block configuration individually
    for target_file in filename_list:
        # Isolate rows where both the experimental domain (dataset_type) and structural filename match
        matching_records = source_datasets[
            (source_datasets['dataset_type'] == target_file['dataset_type']) & 
            (source_datasets['filename'] == target_file['filename'])
        ]
        
        # Scenario 1: Metadata entry found -> Extract and log target location
        if not matching_records.empty:
            # Extract attributes from the first valid occurrence in the data log
            extracted_dataset_type = matching_records.iloc[0]['dataset_type']
            extracted_path = matching_records.iloc[0]['path']

            # Build a lightweight lookup record for the pipeline map
            dataset_metadata = {
                'dataset_type': extracted_dataset_type,
                'path': extracted_path
            }
            
            # Upsert the path tracking data into the destination repository
            update_or_append_csv(dst_path, dataset_metadata, ['path'], id_column='id')
        # Scenario 2: Missing signature -> Skip gracefully to prevent execution failures
        else:
            print(f"Warning: Dataset file '{target_file}' not found in {src_path.name}. Skipping.")

    print("Group completed!")


def create_directory(dir_name, parent_path=None):
    """
    Creates a target directory structure safely, ensuring all intermediate 
    parent directories are automatically generated.

    If no explicit parent path is provided, the function resolves the new 
    directory relative to the Current Working Directory (CWD).

    :param dir_name: string or Path representing the name of the directory to create
    :param parent_path: optional parent directory path workspace (defaults to Path.cwd())
    :return: pathlib.Path object pointing directly to the verified or created directory
    """
    # Resolve the final destination path based on the availability of a parent anchor
    if parent_path:
        target_directory = pathlib.Path(parent_path) / dir_name
    else:
        target_directory = pathlib.Path.cwd() / dir_name
        
    # parents=True: enables recursive creation of any missing upstream directories
    # exist_ok=True: prevents throwing a FileExistsError if the folder is already present
    target_directory.mkdir(parents=True, exist_ok=True)
    
    # Log a clean relative slice of the path (showing the last two directory layers)
    print(f"Directory ready: {target_directory.relative_to(target_directory.parents[1])}")

    return target_directory


def concat_and_shuffle(data_list):
    """
    Concatenates a list of pandas DataFrames into a single DataFrame and shuffles all rows.

    This utility is commonly used to merge multiple data subsets (such as different 
    cyber-attack groups or distinct network logging partitions) and guarantee a uniform, 
    unbiased distribution of records prior to training or evaluating Machine Learning models.

    :param data_list: list of pandas DataFrames to be combined together
    :return: A single consolidated pandas DataFrame with randomized row sorting and reset indices
    """
    # Combine all individual DataFrames from the list vertically into a single matrix
    combined_dataframe = pd.concat(data_list)
    
    # Shuffle the entire dataset randomly 
    # frac=1 ensures that 100% of the original rows are retained in the sample
    # random_state enforces deterministic reproducibility across experimental pipeline executions
    shuffled_dataframe = combined_dataframe.sample(frac=1, random_state=MLConstants.RANDOM_STATE)
    
    # Reset row index markers to a clean, sequential order (0, 1, 2...) 
    # drop=True prevents the old, disorganized index sequence from being added as a new data column
    final_dataframe = shuffled_dataframe.reset_index(drop=True)
    
    return final_dataframe

def group_by_model_and_save(data, dst_dir):
    """
    Groups classification results by model name and exports each group into a separate CSV file.

    This utility takes a comprehensive evaluation log file, isolates rows based on their 
    associated 'model_name' field, and serializes each model's experimental performance data 
    into a dedicated partition inside the target destination directory.

    :param data: Path-like or string pointing to the master classification results CSV file
    :param dst_dir: pathlib.Path or string pointing to the directory where individual files will be saved
    """
    # Load the unified evaluation results database into a pandas DataFrame
    source_dataframe = get_data_from_csv(data)

    # Segment the data rows dynamically by splitting them into distinct sub-matrices per model architecture
    for model_name, model_group in source_dataframe.groupby('model_name'):
        # Generate a standardized filename combining the unique model identifier with the system's naming suffix
        # and serialize the specific model's performance logs to disk
        create_csv_from_data(model_group, f"{model_name}{Naming.CLASSIFICATION}", dst_dir)


def group_by_classes_and_save(data, dst_dir):
    """
    Hierarchically segments evaluation results by dataset type and attack classes, 
    saving the granular subsets into a nested directory structure.

    This utility parses a comprehensive evaluation log, isolates rows by their top-level 
    domain environment ('dataset_type'), creates a parent folder for that domain, and then 
    sub-segments those entries by their target attack profiles ('classes') into deep-nested 
    storage layers.

    :param data: Path-like or string pointing to the master classification results CSV file
    :param dst_dir: pathlib.Path or string pointing to the root destination directory for the layout
    """
    # Load the comprehensive evaluation log data into a pandas DataFrame
    source_dataframe = get_data_from_csv(data)

    # Outer Split: Isolate the evaluation traces by their dataset domain (e.g., 'nb15', 'sat20')
    for dataset_type, dataset_group in source_dataframe.groupby('dataset_type'):
        # Dynamically create a dedicated top-level directory for the specific dataset type
        target_dataset_directory = create_directory(dataset_type, dst_dir)
        
        # Inner Split: Further segment the domain's data by evaluated attack combinations/classes
        for class_identifier, class_group in dataset_group.groupby('classes'):
            # Generate the mandatory nested classes subdirectory inside the domain's folder
            target_classes_directory = create_directory(ProjectPaths.DIR_CLASSES, target_dataset_directory)
            
            # Sanitize the class string to make it safe for filesystem use by clearing spaces and commas
            formatted_class_name = class_identifier.replace(" ", "_").replace(",", "_")
            
            # Export the highly granular evaluation slice into its final localized CSV repository
            create_csv_from_data(
                class_group, 
                f"{formatted_class_name}{Naming.CLASSIFICATION}", 
                target_classes_directory
            )


if __name__ == "__main__":
    pass
