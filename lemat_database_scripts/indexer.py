#!/usr/bin/env python3
"""
Parallel Indexer (Split Output): Builds the ProtoCSP index in multiple parts to avoid OOM.
Usage: python indexer.py --csv-dir ../lemat_unique_csv_500_parts --output-prefix lemat_lib --parts 10
"""

import os
import sys
import pickle
import argparse
import time
import pandas as pd
import csv
import gc
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import helper functions from your existing module
try:
    from lemat_unique_dataset.CrystalGuesser.core import reconstruct_structure
except ImportError:
    print("Error: Could not import 'reconstruct_structure' from crystal_guesser.py")
    print("Make sure both scripts are in the same directory.")
    sys.exit(1)

def process_single_csv(file_path):
    """
    Worker function: Processes one CSV file and returns a partial index.
    """
    local_index = defaultdict(list)
    count = 0
    
    try:
        # Standard pandas read
        df = pd.read_csv(file_path, on_bad_lines='warn', quoting=csv.QUOTE_MINIMAL)
        
        for _, row in df.iterrows():
            try:
                # UPDATED: entry is now a DICT {'structure': S, 'id': X, 'energy_per_atom': Y, ...}
                entry = reconstruct_structure(row)
                
                if entry:
                    # We still need the structure object to calculate the anonymous formula key
                    struct_obj = entry['structure']
                    anon_formula = struct_obj.composition.anonymized_formula
                    
                    # Store the FULL dictionary (metadata included)
                    local_index[anon_formula].append(entry)
                    count += 1
            except Exception:
                continue
                
    except Exception as e:
        return {}, 0, f"Error processing {os.path.basename(file_path)}: {e}"

    return local_index, count, None

def merge_indexes(main_index, partial_index):
    """Merges a worker's partial index into the main index."""
    for key, entries in partial_index.items():
        main_index[key].extend(entries)

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    parser = argparse.ArgumentParser(description='Parallel Index Builder (Split Output)')
    parser.add_argument('--csv-dir', required=True, help='Directory containing CSV parts')
    parser.add_argument('--output-prefix', default='lemat_indexed_library', help='Prefix for output pickle files')
    parser.add_argument('--parts', type=int, default=10, help='Number of split files to create (default: 10)')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of CPU cores to use')
    parser.add_argument('--limit', type=int, default=None, help='Limit total files to process (for testing)')
    
    args = parser.parse_args()

    # 1. Collect Files
    try:
        all_files = sorted([os.path.join(args.csv_dir, f) for f in os.listdir(args.csv_dir) if f.endswith('.csv')])
    except FileNotFoundError:
        print(f"Error: Directory {args.csv_dir} not found.")
        sys.exit(1)

    if args.limit:
        all_files = all_files[:args.limit]

    total_files = len(all_files)
    files_per_part = math.ceil(total_files / args.parts)

    print("="*60)
    print(f"PARALLEL INDEXER (BATCH MODE) - METADATA ENABLED")
    print(f"Source:      {args.csv_dir}")
    print(f"Total Files: {total_files}")
    print(f"Split Into:  {args.parts} parts (~{files_per_part} files/part)")
    print(f"Workers:     {args.workers}")
    print("="*60)

    overall_start = time.time()
    total_structures_all = 0

    # 2. Iterate over batches (Parts)
    # We use sequential processing for PARTS to flush memory between them.
    # We use parallel processing for FILES within a part.
    
    file_chunks = list(chunk_list(all_files, files_per_part))

    for part_idx, file_batch in enumerate(file_chunks):
        part_name = f"{args.output_prefix}_{part_idx:02d}.pkl"
        print(f"\nProcessing Part {part_idx+1}/{len(file_chunks)} -> {part_name}")
        
        part_index = defaultdict(list)
        part_structures = 0
        
        # Parallel Execution for this batch
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {executor.submit(process_single_csv, f): f for f in file_batch}
            
            with tqdm(total=len(file_batch), desc=f"  Indexing Part {part_idx+1}") as pbar:
                for future in as_completed(future_to_file):
                    partial_index, count, error = future.result()
                    
                    if error:
                        tqdm.write(f"  [Warn] {error}")
                    else:
                        merge_indexes(part_index, partial_index)
                        part_structures += count
                    
                    pbar.update(1)

        # Save Part
        print(f"  Saving {part_structures} entries (dicts) to {part_name}...")
        try:
            with open(part_name, 'wb') as f:
                pickle.dump(part_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  [Fatal Error] Could not save {part_name}: {e}")
            return

        total_structures_all += part_structures
        
        # MEMORY CLEANUP CRITICAL STEP
        print(f"  Cleaning memory...")
        del part_index
        del future_to_file
        gc.collect()

    total_time = time.time() - overall_start
    print("="*60)
    print(f"Indexing Complete in {total_time:.2f}s")
    print(f"Total Entries Indexed: {total_structures_all}")
    print(f"Created {len(file_chunks)} files: {args.output_prefix}_XX.pkl")

if __name__ == "__main__":
    main()