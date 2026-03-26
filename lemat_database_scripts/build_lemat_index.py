#!/usr/bin/env python3
"""
Build LeMat Structure Index

This script pre-processes all LeMat structures and builds an optimized index
for fast comparison. The index is saved as a pickle file that can be loaded
in seconds instead of minutes.

The index maps (space_group, elements) -> list of structure info, which is
exactly what compare_structures.py needs for matching.

Usage:
    python3 build_lemat_index.py [options]

Options:
    --lemat-dir DIR     Directory containing processed LeMat structures (default: ./processed_structures)
    --output FILE       Output pickle file (default: ./lemat_index.pkl)
    --verbose           Print detailed progress information
    --help              Show this help message

Performance:
    - Building index: ~10-20 minutes (one-time cost)
    - Loading index: ~5-10 seconds (every comparison)
    - Speedup: ~100-200x faster for subsequent comparisons
"""

import csv
import json
import sys
import os
import pickle
import argparse
import time
import psutil
from collections import defaultdict
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def build_lemat_index(lemat_dir: str, verbose: bool = False, start_part: int = None, end_part: int = None):
    """
    Build an index of all LeMat structures for fast comparison.

    Args:
        lemat_dir: Directory containing processed structure CSV files
        verbose: If True, print detailed progress information
        start_part: Starting part number (for parallel processing)
        end_part: Ending part number (for parallel processing)

    Returns:
        dict: Dictionary mapping (space_group, frozenset(elements)) -> list of structure info dicts
        dict: Metadata about the index (total structures, files processed, etc.)
    """
    print(f"Building LeMat structure index from '{lemat_dir}'...")
    if start_part is not None and end_part is not None:
        print(f"Processing parts {start_part} to {end_part-1} (parallel mode)")
    else:
        print("This is a one-time operation that may take 10-20 minutes.")
    print()

    # Get process for memory monitoring
    process = psutil.Process(os.getpid())

    # Increase CSV field size limit
    csv.field_size_limit(10000000)

    # Dictionary to store structures indexed by (space_group, elements)
    structures_index = defaultdict(list)

    # Statistics
    total_loaded = 0
    total_failed = 0
    files_processed = 0
    files_failed = 0

    start_time = time.time()
    last_report_time = start_time

    # Get all CSV files in the directory
    all_csv_files = sorted([f for f in os.listdir(lemat_dir) if f.endswith('.csv')])

    if not all_csv_files:
        print(f"ERROR: No CSV files found in '{lemat_dir}'")
        return None, None

    # Filter files if processing a subset
    if start_part is not None and end_part is not None:
        csv_files = [f for f in all_csv_files if f.startswith('part_')]
        csv_files = [f for f in csv_files if start_part <= int(f.split('_')[1]) < end_part]
    else:
        csv_files = all_csv_files

    print(f"Found {len(csv_files)} CSV files to process (out of {len(all_csv_files)} total)")
    print()

    # Process each CSV file
    for file_idx, csv_file in enumerate(csv_files, 1):
        file_path = os.path.join(lemat_dir, csv_file)
        file_start_time = time.time()

        # Always show progress for each file
        elapsed = time.time() - start_time
        rate = total_loaded / elapsed if elapsed > 0 else 0
        mem_mb = process.memory_info().rss / (1024 * 1024)

        # Estimate time remaining
        if file_idx > 1:
            avg_time_per_file = elapsed / (file_idx - 1)
            remaining_files = len(csv_files) - file_idx + 1
            eta_seconds = avg_time_per_file * remaining_files
            eta_str = f", ETA: {eta_seconds/60:.1f} min"
        else:
            eta_str = ""

        print(f"[{file_idx}/{len(csv_files)}] Processing {csv_file}...")
        print(f"  Progress: {total_loaded:,} structures loaded, {rate:.0f} struct/sec, "
              f"Memory: {mem_mb:.0f} MB{eta_str}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Parse the pymatgen Structure from JSON
                        structure_dict = json.loads(row['pymatgen_json'])
                        structure = Structure.from_dict(structure_dict)
                        
                        # Determine space group
                        try:
                            sga = SpacegroupAnalyzer(structure)
                            space_group = sga.get_space_group_symbol()
                        except Exception as e:
                            # If space group analysis fails, use a default
                            if verbose:
                                print(f"  Warning: Space group analysis failed for {row['immutable_id']}: {e}")
                            space_group = "Unknown"
                        
                        # Get elements as a frozenset (order-independent)
                        elements = frozenset(structure.composition.elements)
                        
                        # Create index key
                        key = (space_group, elements)
                        
                        # Store structure info
                        structure_info = {
                            'immutable_id': row['immutable_id'],
                            'formula': row['chemical_formula_descriptive'],
                            'space_group': space_group,
                            'elements': elements,
                            'structure': structure  # Store the full Structure object
                        }
                        
                        structures_index[key].append(structure_info)
                        total_loaded += 1
                        
                    except Exception as e:
                        if verbose:
                            print(f"  Error loading structure from {csv_file}: {e}")
                        total_failed += 1
            
            files_processed += 1

            # Report file completion
            file_elapsed = time.time() - file_start_time
            print(f"  ✓ Completed in {file_elapsed:.2f} seconds")
            print()

        except Exception as e:
            print(f"  ✗ ERROR: Failed to process {csv_file}: {e}")
            print()
            files_failed += 1

    elapsed_time = time.time() - start_time
    final_mem_mb = process.memory_info().rss / (1024 * 1024)
    
    # Build metadata
    metadata = {
        'total_structures': total_loaded,
        'failed_structures': total_failed,
        'files_processed': files_processed,
        'files_failed': files_failed,
        'unique_keys': len(structures_index),
        'build_time': elapsed_time,
        'build_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'lemat_dir': lemat_dir,
        'start_part': start_part,
        'end_part': end_part
    }

    print()
    print("=" * 80)
    print("INDEX BUILD COMPLETE")
    print("=" * 80)
    print(f"Total structures indexed: {total_loaded:,}")
    print(f"Failed to load: {total_failed:,}")
    print(f"Files processed: {files_processed}/{len(csv_files)}")
    print(f"Unique (space_group, elements) combinations: {len(structures_index):,}")
    print(f"Build time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Average rate: {total_loaded/elapsed_time:.0f} structures/second")
    print(f"Peak memory usage: {final_mem_mb:.0f} MB")
    print("=" * 80)
    
    return structures_index, metadata


def save_index(structures_index, metadata, output_file: str):
    """
    Save the index to a pickle file.
    
    Args:
        structures_index: The index dictionary
        metadata: Metadata about the index
        output_file: Path to output pickle file
    """
    print()
    print(f"Saving index to '{output_file}'...")
    
    # Package both index and metadata
    data = {
        'index': structures_index,
        'metadata': metadata
    }
    
    start_time = time.time()
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    save_time = time.time() - start_time
    file_size = os.path.getsize(output_file)
    
    print(f"✓ Index saved successfully!")
    print(f"  File size: {file_size / (1024**3):.2f} GB")
    print(f"  Save time: {save_time:.2f} seconds")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Build pre-indexed LeMat structure database for fast comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--lemat-dir',
        default='./processed_structures',
        help='Directory containing processed LeMat structures (default: ./processed_structures)'
    )
    
    parser.add_argument(
        '--output',
        default='./lemat_index.pkl',
        help='Output pickle file (default: ./lemat_index.pkl)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )

    parser.add_argument(
        '--start-part',
        type=int,
        help='Starting part number (for parallel processing)'
    )

    parser.add_argument(
        '--end-part',
        type=int,
        help='Ending part number (for parallel processing)'
    )

    args = parser.parse_args()
    
    # Check if lemat directory exists
    if not os.path.exists(args.lemat_dir):
        print(f"ERROR: Directory '{args.lemat_dir}' not found")
        print()
        print("Please ensure you have processed the LeMat structures first:")
        print("  sbatch submit_process_structures.sh")
        print("  # or #")
        print("  python3 structures_converter.py --all")
        sys.exit(1)
    
    # Build the index
    print("=" * 80)
    print("BUILDING LEMAT STRUCTURE INDEX")
    print("=" * 80)
    print()

    structures_index, metadata = build_lemat_index(
        args.lemat_dir,
        args.verbose,
        args.start_part,
        args.end_part
    )

    if structures_index is None:
        print("ERROR: Failed to build index")
        sys.exit(1)
    
    # Save the index
    save_index(structures_index, metadata, args.output)
    
    # Print usage instructions
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("The index has been built successfully! You can now use it for fast comparisons:")
    print()
    print(f"  python3 compare_structures.py \\")
    print(f"      --lemat-index {args.output} \\")
    print(f"      --comparison-file /path/to/comparison.csv \\")
    print(f"      --output results.csv")
    print()
    print("This will load the index in ~5-10 seconds instead of 10-20 minutes!")
    print("=" * 80)


if __name__ == "__main__":
    main()

