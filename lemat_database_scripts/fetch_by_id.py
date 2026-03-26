#!/usr/bin/env python3
"""
CLI Tool to fetch a CIF file from LeMaterial using its ID.
Usage: python fetch_by_id.py agm2000002426
"""

import argparse
import sys
import os
from id_manager import LeMatIDManager

def main():
    parser = argparse.ArgumentParser(description='Fetch Structure CIF by ID')
    parser.add_argument('id', help='The ID to search for (e.g., agm2000002426)')
    parser.add_argument('--db', default='lemat_formula_indexed', help='Path to database folder')
    parser.add_argument('--rebuild-index', action='store_true', help='Force rebuild of ID index')
    parser.add_argument('--output', '-o', default='.', help='Output directory for CIF')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"SEARCHING FOR ID: {args.id}")
    print("="*60)

    # Initialize Manager
    manager = LeMatIDManager(db_folder=args.db)
    
    # Load or Build Index
    try:
        manager.load_or_build_index(force_rebuild=args.rebuild_index)
    except FileNotFoundError:
        print(f"Error: Database folder '{args.db}' not found.")
        print("Please ensure you are pointing to the folder containing your JSON files.")
        sys.exit(1)

    # Fetch Entry
    entry = manager.get_entry_by_id(args.id)

    if entry:
        struct = entry['structure']
        formula = struct.composition.reduced_formula
        
        print(f"\n✓ FOUND MATCH!")
        print(f"  Formula: {formula}")
        print(f"  Volume:  {struct.volume:.3f} Å³")
        print(f"  Sites:   {struct.num_sites}")
        print(f"  Energy:  {entry.get('energy_per_atom', 'N/A')}")
        
        # Save CIF
        filename = f"{formula}_{args.id}.cif"
        output_path = os.path.join(args.output, filename)
        
        os.makedirs(args.output, exist_ok=True)
        struct.to(filename=output_path, fmt="cif")
        
        print(f"\n✓ Saved structure to: {output_path}")
    else:
        print(f"\n✗ ID '{args.id}' not found in database.")
        print("  (If you recently added new CSVs, run with --rebuild-index)")

if __name__ == "__main__":
    main()