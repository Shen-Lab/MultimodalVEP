import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_label_stats_per_file(data_directory, label_column='DMS_score'):
    """
    Calculates the mean and std dev for each CSV file and for all data combined.
    """
    all_labels = []
    file_stats = []
    
    csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in '{data_directory}'.")
        return None
        
    print(f"Found {len(csv_files)} CSV files. Reading data...")

    for filename in tqdm(csv_files, desc="Processing files"):
        file_path = os.path.join(data_directory, filename)
        try:
            df = pd.read_csv(file_path)
            if label_column in df.columns:
                labels = df[label_column].dropna()
                if not labels.empty:
                    # Calculate stats for the current file
                    file_mean = labels.mean()
                    file_std = labels.std()
                    file_stats.append({
                        "filename": filename,
                        "mean": file_mean,
                        "std_dev": file_std,
                        "count": len(labels)
                    })
                    # Add this file's labels to the master list for overall stats
                    all_labels.extend(labels.tolist())
            else:
                print(f"Warning: Column '{label_column}' not found in {filename}.")
        except Exception as e:
            print(f"Could not read or process {filename}. Error: {e}")
            
    if not all_labels:
        print("No label data was collected from the files.")
        return None

    # --- Print the results in a formatted table ---
    # Create a DataFrame from the collected file statistics for nice printing
    stats_df = pd.DataFrame(file_stats)
    print("\n--- Statistics For Each File ---")
    print(stats_df.to_string(index=False, float_format="{:.4f}".format))

    # --- Calculate and print overall statistics ---
    overall_mean = np.mean(all_labels)
    overall_std = np.std(all_labels)
    
    print("\n--- Overall Label Statistics ---")
    print(f"Mean: {overall_mean:.4f}")
    print(f"Standard Deviation: {overall_std:.4f}")
    
    return stats_df

if __name__ == '__main__':
    # 👈 Change this to the path of your dataset directory
    DATASET_DIR = "../dataset/ProteinGym/substitution_split" 
    
    calculate_label_stats_per_file(DATASET_DIR)
