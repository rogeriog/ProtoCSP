#!/usr/bin/env python3
"""
inspect_db.py: Checks the AB.json file for NaCl entries and verifies formula formatting.
"""
import json
import os
import sys
from pymatgen.core import Composition

# Path to the specific file where NaCl (1:1 ratio) must be
DB_FILE = "lemat_formula_indexed/AB.json"

def main():
    if not os.path.exists(DB_FILE):
        print(f"Error: {DB_FILE} not found.")
        sys.exit(1)

    print(f"Loading {DB_FILE}...")
    with open(DB_FILE, 'r') as f:
        data = json.load(f)

    print(f"Total entries in AB.json: {len(data)}")
    
    # 1. Check Metadata formatting (First 5 entries)
    print("\n--- Sample Formulas (First 5) ---")
    for i, entry in enumerate(data[:5]):
        print(f"Index {i}: {entry.get('reduced_formula')} | Energy: {entry.get('energy_per_atom')}")

    # 2. Deep Search for NaCl
    print("\n--- Searching for NaCl ---")
    target = Composition("NaCl")
    
    exact_string_matches = 0
    composition_matches = 0
    clna_matches = 0
    
    found_formulas = set()

    for entry in data:
        db_formula_str = entry.get('reduced_formula', '')
        
        # Check A: Strict String Match
        if db_formula_str == "NaCl":
            exact_string_matches += 1
            
        # Check B: Reverse String Match
        if db_formula_str == "ClNa":
            clna_matches += 1
            
        # Check C: Robust Chemical Equality (Slow but accurate)
        # We parse the string back into a Composition object
        try:
            db_comp = Composition(db_formula_str)
            if db_comp.reduced_composition == target.reduced_composition:
                composition_matches += 1
                found_formulas.add(db_formula_str)
        except:
            pass

    print(f"Target: {target.reduced_formula}")
    print(f"Exact String Matches ('NaCl'): {exact_string_matches}")
    print(f"Reverse String Matches ('ClNa'): {clna_matches}")
    print(f"Chemical Composition Matches (Total): {composition_matches}")
    print(f"Variations found in DB: {found_formulas}")

    if composition_matches > 0 and exact_string_matches == 0:
        print("\n[CONCLUSION] NaCl exists but the string formatting is different!")
        print("You need to update Strategy 0 to compare Composition objects, not strings.")
    elif composition_matches == 0:
        print("\n[CONCLUSION] NaCl is NOT in the database subset you indexed.")
        print("Check if the original CSVs actually contain Rock Salt NaCl.")
    else:
        print("\n[CONCLUSION] NaCl is present and formatted correctly. The issue might be elsewhere.")

if __name__ == "__main__":
    main()