#!/usr/bin/env python3
"""
Parallel Index Building Script

This script is called by SLURM array jobs to build partial indices in parallel.
Each task processes a subset of CSV files and creates a partial index pickle file.

Usage:
    python3 build_index_parallel.py <task_id>

Environment Variables:
    SLURM_ARRAY_TASK_ID - Task ID from SLURM array job
"""

import sys
import os

# Configuration
TOTAL_PARTS = 500  # Total number of CSV part files
NUM_PARALLEL_JOBS = 10  # Number of parallel SLURM tasks
LEMAT_DIR = './processed_structures'
OUTPUT_DIR = './partial_indices'

def main():
    # Get task ID from command line or environment
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    elif 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        print("ERROR: No task ID provided")
        print("Usage: python3 build_index_parallel.py <task_id>")
        sys.exit(1)
    
    # Calculate which parts this task should process
    parts_per_job = TOTAL_PARTS // NUM_PARALLEL_JOBS
    start_part = task_id * parts_per_job
    end_part = start_part + parts_per_job
    
    # Last task gets any remaining parts
    if task_id == NUM_PARALLEL_JOBS - 1:
        end_part = TOTAL_PARTS
    
    print(f"Task {task_id}: Processing parts {start_part} to {end_part-1}")
    print(f"Total parts to process: {end_part - start_part}")
    print()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Output file for this task
    output_file = os.path.join(OUTPUT_DIR, f'partial_index_{task_id:02d}.pkl')
    
    # Import here to avoid loading pymatgen in parent process
    from build_lemat_index import build_lemat_index, save_index
    
    # Build partial index
    print("=" * 80)
    print(f"BUILDING PARTIAL INDEX {task_id}")
    print("=" * 80)
    print()
    
    structures_index, metadata = build_lemat_index(
        LEMAT_DIR,
        verbose=True,
        start_part=start_part,
        end_part=end_part
    )
    
    if structures_index is None:
        print(f"ERROR: Failed to build partial index {task_id}")
        sys.exit(1)
    
    # Save partial index
    save_index(structures_index, metadata, output_file)
    
    print()
    print("=" * 80)
    print(f"PARTIAL INDEX {task_id} COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"Structures indexed: {metadata['total_structures']:,}")
    print(f"Unique keys: {metadata['unique_keys']:,}")
    print("=" * 80)
    print()
    print("After all tasks complete, run:")
    print("  python3 merge_indices.py")
    print()

if __name__ == "__main__":
    main()

