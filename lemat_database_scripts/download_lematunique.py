# download_lematunique_array.py

# ==============================================================================
# 1. IMPORTS AND ARGUMENT PARSING
# ==============================================================================
import os
import sys
import time
from datasets import load_dataset
import pandas as pd

# --- Configuration ---
TOTAL_SHARDS = 500
NUM_PARALLEL_JOBS = 5 # The size of the job array (0, 1, 2)
OUTPUT_DIR = 'lemat_unique_csv_500_parts'
DATASET_NAME = 'LeMaterial/LeMat-BulkUnique'
CONFIG_NAME = 'unique_pbe'

# Check if the script received the required array ID argument
if len(sys.argv) != 2:
    print("Usage: python3 download_lematunique_array.py <SLURM_ARRAY_TASK_ID>")
    sys.exit(1)

# Parse the single command-line argument: the array ID (0, 1, or 2)
try:
    TASK_ID = int(sys.argv[1])
except ValueError:
    print("Error: SLURM_ARRAY_TASK_ID must be an integer.")
    sys.exit(1)

# ==============================================================================
# 2. CALCULATE SHARD RANGE BASED ON TASK_ID
# ==============================================================================
# We divide the 500 shards as evenly as possible among the 3 jobs.
# Job 0: Shards 0-166 (167 total)
# Job 1: Shards 167-333 (167 total)
# Job 2: Shards 334-499 (166 total)

SHARDS_PER_JOB = TOTAL_SHARDS // NUM_PARALLEL_JOBS # 500 // 3 = 166
REMAINDER = TOTAL_SHARDS % NUM_PARALLEL_JOBS       # 500 % 3 = 2 (Jobs 0 and 1 get one extra)

# Calculate the actual start and end shards for this specific job
# The start is the base calculation (SHARDS_PER_JOB * TASK_ID) plus any extra shards
# that preceding jobs (i.e., jobs with ID < TASK_ID) received.
START_SHARD = SHARDS_PER_JOB * TASK_ID + min(TASK_ID, REMAINDER)

# The number of shards this job handles
JOB_SHARD_COUNT = SHARDS_PER_JOB + (1 if TASK_ID < REMAINDER else 0)

# The end shard is exclusive (stops BEFORE this index)
END_SHARD = START_SHARD + JOB_SHARD_COUNT

print(f"--- Starting Data Processing Job (Task ID: {TASK_ID}) ---")
print(f"Goal: Process Shards {START_SHARD} to {END_SHARD - 1} (Total: {JOB_SHARD_COUNT} shards)")
start_time = time.time()

# ==============================================================================
# 3. DATASET LOADING, SETUP, AND SHARDING LOOP
# (The rest of the script is the same as the verbose version, adapted for the new range variables)
# ==============================================================================
# Ensure the output directory is created for all jobs
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nLoading dataset '{DATASET_NAME}/{CONFIG_NAME}' (from cache)...")
try:
    dataset = load_dataset(DATASET_NAME, CONFIG_NAME)
    train_dataset = dataset['train']
    print("SUCCESS: Dataset loaded into memory.")
except Exception as e:
    print(f"FATAL ERROR: Could not load dataset: {e}")
    sys.exit(1)

print(f"\nStarting processing loop...")

for i in range(START_SHARD, END_SHARD):
    # Get the specific shard index 'i' using the TOTAL_SHARDS for consistent partitioning
    shard = train_dataset.shard(num_shards=TOTAL_SHARDS, index=i)
    
    # Convert and Save (The I/O and CPU step)
    df_chunk = shard.to_pandas()
    
    # The filename uses the index 'i'
    output_filename = os.path.join(OUTPUT_DIR, f'part_{i:03d}.csv')
    
    df_chunk.to_csv(output_filename, index=False)
    
    print(f"  [Chunk {i:03d}/{TOTAL_SHARDS - 1:03d}]: Saved {len(df_chunk):,} rows to {output_filename}")

# ==============================================================================
# 4. COMPLETION
# ==============================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n--- Job Finished Successfully ---")
print(f"Processed shards {START_SHARD} to {END_SHARD - 1} in {elapsed_time:.2f} seconds.")