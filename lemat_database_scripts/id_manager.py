#!/usr/bin/env python3
"""
ID Manager for LeMaterial/Alexandria Database.
Creates a reverse index (ID -> Formula/File) to allow fast O(1) lookups 
of structures by their immutable ID.
"""

import os
import json
import pickle
import sys
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from pymatgen.core import Structure
except ImportError:
    print("Error: 'pymatgen' not installed.")
    sys.exit(1)

class LeMatIDManager:
    def __init__(self, db_folder: str = "lemat_formula_indexed", index_file: str = "lemat_id_index.pkl"):
        self.db_folder = db_folder
        self.index_file = index_file
        self.id_map = {} # Maps ID -> Formula (Filename)

    def load_or_build_index(self, force_rebuild: bool = False):
        """Loads the ID index from disk, or builds it if missing."""
        if os.path.exists(self.index_file) and not force_rebuild:
            print(f"[INFO] Loading ID index from {self.index_file}...")
            with open(self.index_file, 'rb') as f:
                self.id_map = pickle.load(f)
            print(f"[INFO] Loaded {len(self.id_map)} IDs.")
        else:
            self._build_index()

    def _build_index(self):
        """Scans all JSON files in the DB folder to map IDs to Files."""
        print(f"[INFO] Building ID Index from {self.db_folder}...")
        
        if not os.path.isdir(self.db_folder):
            raise FileNotFoundError(f"Database folder {self.db_folder} not found.")

        json_files = [f for f in os.listdir(self.db_folder) if f.endswith('.json')]
        total_files = len(json_files)
        
        # Use a list to collect data, then dict update for speed
        temp_map = {}
        
        for f_name in tqdm(json_files, desc="Indexing IDs"):
            full_path = os.path.join(self.db_folder, f_name)
            try:
                with open(full_path, 'r') as f:
                    entries = json.load(f)
                    # The filename (minus .json) is the key to find this file again
                    formula_key = f_name.replace('.json', '')
                    
                    for entry in entries:
                        # Support 'id', 'immutable_id', or 'material_id'
                        uid = entry.get('id') or entry.get('immutable_id') or entry.get('material_id')
                        if uid:
                            temp_map[str(uid)] = formula_key
            except Exception as e:
                print(f"[WARN] Failed to read {f_name}: {e}")

        self.id_map = temp_map
        print(f"[INFO] Index complete. Found {len(self.id_map)} unique IDs.")
        
        print(f"[INFO] Saving index to {self.index_file}...")
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.id_map, f)

    def get_entry_by_id(self, target_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the full entry dictionary for a given ID."""
        if not self.id_map:
            self.load_or_build_index()

        target_id = str(target_id)
        
        # 1. Look up formula file
        formula_key = self.id_map.get(target_id)
        if not formula_key:
            return None

        # 2. Load the specific JSON file (efficient)
        json_path = os.path.join(self.db_folder, f"{formula_key}.json")
        try:
            with open(json_path, 'r') as f:
                entries = json.load(f)
                
            # 3. Find the specific entry in that list
            for entry in entries:
                uid = entry.get('id') or entry.get('immutable_id')
                if str(uid) == target_id:
                    # Hydrate structure
                    if isinstance(entry.get('structure'), dict):
                        entry['structure'] = Structure.from_dict(entry['structure'])
                    return entry
        except Exception as e:
            print(f"[ERROR] Could not read source file for {target_id}: {e}")
            
        return None