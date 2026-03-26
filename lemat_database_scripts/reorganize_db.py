#!/usr/bin/env python3
"""
Reorganize DB (Parallel): Converts split Pickle chunks into a Formula-Indexed JSON Database.
SORTED: Entries are now sorted by energy_per_atom (Ascending) before saving.
"""

import os
import sys
import json
import pickle
import glob
import re
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pymatgen.core import Structure 

def sanitize_filename(formula):
    return re.sub(r'[\\/*?:"<>|]', "", formula)

def process_formula_batch(args):
    """
    Worker function to process a single formula.
    Args: (formula, entries_list, output_dir)
    """
    formula, entries_list, output_dir = args
    
    try:
        clean_name = sanitize_filename(formula)
        json_path = os.path.join(output_dir, f"{clean_name}.json")
        
        # 1. Serialization
        json_ready_entries = []
        for entry in entries_list:
            new_entry = entry.copy()
            if 'structure' in new_entry and hasattr(new_entry['structure'], 'as_dict'):
                new_entry['structure'] = new_entry['structure'].as_dict()
            json_ready_entries.append(new_entry)
        
        # 2. Read Existing Data (if appending to existing JSON)
        current_data = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as jf:
                    current_data = json.load(jf)
            except json.JSONDecodeError:
                pass 
        
        # 3. Merge New and Old
        current_data.extend(json_ready_entries)

        # 4. CRITICAL STEP: SORT BY ENERGY PER ATOM
        # We push entries with no energy (None) to the end (float('inf'))
        current_data.sort(key=lambda x: x.get('energy_per_atom') if x.get('energy_per_atom') is not None else float('inf'))

        # 5. Save
        with open(json_path, 'w') as jf:
            json.dump(current_data, jf)
            
        return len(json_ready_entries), None
        
    except Exception as e:
        return 0, f"Error processing {formula}: {e}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Parallel Reorganize DB')
    parser.add_argument('--input-prefix', default='lemat_indexed_library', help='Prefix of split pkl files')
    parser.add_argument('--output-dir', default='lemat_formula_indexed', help='Folder to store JSON files')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of parallel workers')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pkl_files = sorted(glob.glob(f"{args.input_prefix}_*.pkl"))
    
    if not pkl_files:
        print(f"No files found matching {args.input_prefix}_*.pkl")
        sys.exit(1)

    print("="*60)
    print(f"PARALLEL DATABASE REORGANIZATION (SORTED BY ENERGY)")
    print(f"Input Files: {len(pkl_files)}")
    print(f"Output Dir:  {args.output_dir}/")
    print(f"Workers:     {args.workers}")
    print("="*60)

    total_entries = 0

    for pkl_idx, pkl_file in enumerate(pkl_files):
        print(f"\nLoading Chunk {pkl_idx+1}/{len(pkl_files)}: {pkl_file} ...")
        
        try:
            with open(pkl_file, 'rb') as f:
                data_chunk = pickle.load(f)
        except Exception as e:
            print(f"Skipping corrupt chunk {pkl_file}: {e}")
            continue

        tasks = [(formula, entries, args.output_dir) for formula, entries in data_chunk.items()]
        print(f"  Distributing {len(tasks)} formulas to workers...")
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_formula_batch, t) for t in tasks]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Processing Chunk {pkl_idx+1}"):
                count, error = future.result()
                if error:
                    tqdm.write(f"    [Warn] {error}")
                total_entries += count

        del data_chunk
        del tasks
        del futures
        gc.collect()

    print("\n" + "="*60)
    print("COMPLETE")
    print(f"Total Entries Processed: {total_entries}")

if __name__ == "__main__":
    main()