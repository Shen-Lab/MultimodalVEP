import pandas as pd
import os

# --- Configuration ---
# You can adjust these settings
BENCHMARK_CSV_PATH = "esm2_fast_benchmark.csv"
TARGET_HOURS_PER_JOB = 12
OUTPUT_FILE_PREFIX = "job_12h"
# -------------------

def partition_files_by_time(csv_path, target_hours):
    """
    Reads a benchmark CSV and partitions files into jobs based on a target runtime.

    Args:
        csv_path (str): Path to the benchmark CSV file.
        target_hours (int): The desired maximum runtime for each job in hours.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Benchmark file not found at '{csv_path}'")
        return

    # --- 1. Preparation ---
    max_seconds_per_job = target_hours * 3600
    # Sort files from longest to shortest estimated time for optimal packing
    df = df.sort_values(by="estimated_total_time_s", ascending=False).reset_index(drop=True)

    # --- 2. Check for Files Exceeding the Time Limit ---
    long_files = df[df['estimated_total_time_s'] > max_seconds_per_job]
    if not long_files.empty:
        print("--- ⚠️ Warning: Long-running files detected ---")
        for _, row in long_files.iterrows():
            long_hours = row['estimated_total_time_s'] / 3600
            print(f"  - File '{row['DMS_id']}' is estimated to take {long_hours:.2f} hours.")
        print(f"These files will be placed in their own dedicated jobs, exceeding the {target_hours}h target.\n")

    # --- 3. Partitioning Algorithm (Greedy Approach) ---
    job_bins = []         # A list of lists, where each inner list contains file IDs
    job_workloads_s = []  # The total runtime in seconds for each corresponding job bin

    print(f"Partitioning {len(df)} files into jobs with a ~{target_hours} hour target...")

    for _, row in df.iterrows():
        file_id = row['DMS_id']
        file_time_s = row['estimated_total_time_s']
        
        placed = False
        # Try to place the file in an existing job bin
        for i in range(len(job_bins)):
            if job_workloads_s[i] + file_time_s <= max_seconds_per_job:
                job_bins[i].append(file_id)
                job_workloads_s[i] += file_time_s
                placed = True
                break
        
        # If it didn't fit anywhere, create a new job bin for it
        if not placed:
            job_bins.append([file_id])
            job_workloads_s.append(file_time_s)

    # --- 4. Report and Save Results ---
    num_jobs_created = len(job_bins)
    max_job_time_s = max(job_workloads_s) if job_workloads_s else 0

    print("\n--- ✅ Job Splitting Complete ---")
    print(f"Generated {num_jobs_created} jobs.")
    
    print("\nEstimated time for each job:")
    for i, workload_s in enumerate(job_workloads_s):
        job_id = i + 1
        workload_h = workload_s / 3600
        num_files = len(job_bins[i])
        print(f"  - Job {job_id:02d}: {workload_h:.2f} hours ({num_files} files)")

    print("\n--- Summary ---")
    print(f"Estimated overall time to completion: {max_job_time_s / 3600:.2f} hours")
    print("This is determined by the runtime of the longest job.")
    print(f"Recommended SLURM setting: #SBATCH --time={int(max_job_time_s / 3600) + 1:02d}:00:00")

    # Write the file lists to disk
    for i, file_list in enumerate(job_bins):
        job_id = i + 1
        output_filename = f"{OUTPUT_FILE_PREFIX}_{job_id}_files.txt"
        with open(output_filename, "w") as f:
            for filename in file_list:
                f.write(f"{filename}.csv\n")
    
    print(f"\nGenerated {num_jobs_created} job file lists (e.g., '{OUTPUT_FILE_PREFIX}_1_files.txt').")


if __name__ == "__main__":
    partition_files_by_time(BENCHMARK_CSV_PATH, TARGET_HOURS_PER_JOB)
