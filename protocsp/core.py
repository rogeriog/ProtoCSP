#!/usr/bin/env python3
"""
ProtoCSP: A tool for generating structural guesses by substituting chemical species into crystal frameworks.

This module implements the ProtoCSP class that handles complex stoichiometries,
element mapping, and physical validity checks for crystal structure prediction.
Updated to handle:
1. Complex multi-site solid solutions (e.g. La0.5Ca0.5Mn0.8Fe0.2O3) via Generalized Bucket Filling.
2. Interstitial alloys (e.g. Steel) via Atomic Radius checks and Monte Carlo insertion.
"""

import os
import sys
import json
import csv
import ast
import re
import gc
import math
import random
import itertools
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer # Ensure this is imported
try:
    from pymatgen.core import Structure, Composition, Element
    from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation
    from pymatgen.analysis.local_env import CrystalNN
except ImportError:
    print("Error: 'pymatgen' library not found. Please run 'pip install pymatgen'")
    sys.exit(1)

# Increase CSV field size limit
csv.field_size_limit(10000000)


class ProtoCSP:
    """
    A class for generating structural guesses for given compositions.
    Supports Lazy Loading to avoid instantiating thousands of Pymatgen objects unnecessarily.
    Operates on Entry Dictionaries to preserve metadata.
    """

    def __init__(self, library_source: Union[Dict[str, List[Dict]], str]):
        self.library_source = library_source
        self.MAX_ATOMS = 1200
        
        # Check mode
        self.is_folder_mode = False
        if isinstance(library_source, str):
            if os.path.isdir(library_source):
                self.is_folder_mode = True
                self._cache = {}
            else:
                print(f"Warning: {library_source} is not a valid directory.")

    def _get_entries(self, anon_formula: str) -> List[Dict[str, Any]]:
        """Helper to fetch entries (metadata + raw structure dict) without instantiation."""
        # 1. Memory Mode
        if not self.is_folder_mode:
            return self.library_source.get(anon_formula, [])
        
        # 2. Folder Mode (JSON)
        if anon_formula in self._cache:
            return self._cache[anon_formula]

        clean_name = re.sub(r'[\\/*?:"<>|]', "", anon_formula)
        json_path = os.path.join(self.library_source, f"{clean_name}.json")
        
        if not os.path.exists(json_path):
            return []
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            self._cache[anon_formula] = data
            return data
        except Exception as e:
            print(f"Error loading prototypes for {anon_formula}: {e}")
            return []

    def _hydrate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures the 'structure' key in the entry dict is a Pymatgen Object.
        Returns the modified dictionary (metadata preserved).
        """
        if 'structure' in entry:
            if isinstance(entry['structure'], dict):
                try:
                    entry['structure'] = Structure.from_dict(entry['structure'])
                except Exception:
                    return None
            return entry
        return None

    def generate(self, target_composition_str: str, max_bases: int = 3, top_k: int = 5, min_atoms: int = 20, 
                 randomize_scaling: bool = False, manual_base_struct: Optional[Structure] = None, manual_base_id: str = "",
                 symmetrize: bool = False) -> List[Dict[str, Any]]:
        """
        Main entry point. Generates candidates.
        Returns: List of Dictionaries [{'structure': S, 'id': '...', 'energy_per_atom': ...}, ...]
        """
        t0 = time.time()
        target_comp = Composition(target_composition_str)
        target_reduced_formula = target_comp.reduced_formula
        anon_formula = target_comp.anonymized_formula
        
        print(f"[DEBUG] Starting guess for {target_reduced_formula} (Anon: {anon_formula})")
        guesses = [] # List of Dicts

        # --- MANUAL BASE STRUCTURE HANDLING ---
        if manual_base_struct is not None:
            print(f"[DEBUG] Strategy: Bypassing database to use manual base structure...")
            try:
                sga = SpacegroupAnalyzer(manual_base_struct, symprec=0.1)
                sg = sga.get_space_group_symbol()
            except:
                sg = "Unknown"
                
            base_entry = {
                'structure': manual_base_struct,
                'id': 'provided_'+manual_base_id,
                'reduced_formula': manual_base_struct.composition.reduced_formula,
                'parent_space_group': sg,
                'energy_per_atom': None
            }
            
            # 1. CHECK FOR EXACT MATCH
            if manual_base_struct.composition.reduced_composition == target_comp.reduced_composition:
                print("[DEBUG] Manual base is an EXACT match. Returning as is.")
                base_entry['method'] = 'Manual Base (Exact Match)'
                guesses.append(base_entry)
                return self._finalize_candidates(guesses, top_k)

            # 2. CHECK FOR SIMPLE TRANSMUTATION (1-to-1 element mapping)
            elif len(manual_base_struct.composition.elements) == len(target_comp.elements):
                print("[DEBUG] Manual base has the same number of elements. Attempting transmutation...")
                candidate = self._transmute_structure(base_entry, target_comp)
                if candidate:
                    candidate['method'] = 'Manual Base (Transmuted)'
                    guesses.append(candidate)
                    return self._finalize_candidates(guesses, top_k)

            # 3. FALLBACK TO DOPING
            print("[DEBUG] Manual base requires complex doping/alloying...")
            major_elements = list(manual_base_struct.composition.elements)
            
            # --- MODIFIED SCALING LOGIC ---
            if symmetrize:
                # If symmetrizing, do not randomly scale up the cell to avoid enumlib explosion
                multipliers = [1.0]
                # Only do 1 attempt since enumlib returns a deterministic ranked list
                attempts = 1 
            else:
                multipliers = [1.0, 1.0, 1.0, 2.0, 3.0]
                attempts = max(2, top_k * 3)

            for i in range(attempts):
                mult = random.choice(multipliers)
                current_min_atoms = int(min_atoms * mult)
                try:
                    candidates = self._generate_doped_structure(
                        base_entry, 
                        major_elements, 
                        target_comp,
                        min_atoms=current_min_atoms if not symmetrize else 1,
                        randomize_scaling=randomize_scaling if not symmetrize else False,
                        symmetrize=symmetrize,
                        top_k=top_k
                    )
                    if candidates:
                        # Use extend because candidates is now a list
                        for cand in candidates:
                            cand['method'] = f'Manual Base Doping (Target Size: ~{current_min_atoms})'
                        guesses.extend(candidates)
                    
                except Exception as e:
                    print(f"[DEBUG] Manual Doping failed on iter {i}: {e}")
                    
            if guesses:
                print(f"[DEBUG] Successfully generated {len(guesses)} structures from manual base.")
            else:
                print(f"[WARNING] Failed to generate valid structures from the manual base.")
                
            return self._finalize_candidates(guesses, top_k)
        
        t_load = time.time()
        entries = self._get_entries(anon_formula)
        print(f"[DEBUG] Loaded {len(entries)} entries for {anon_formula} in {time.time()-t_load:.4f}s")
        
        if entries:
            # --- Strategy 0: Exact Database Match (Robust) ---
            # 1. Fast String Check
            exact_matches_entries = [
                e for e in entries 
                if e.get('reduced_formula') == target_reduced_formula
            ]
            
            # 2. Slow Composition Check (Fixes NaCl vs ClNa issue)
            if not exact_matches_entries:
                print(f"[DEBUG] Exact string match failed. Trying robust chemical comparison...")
                target_reduced_comp = target_comp.reduced_composition
                for e in entries:
                    try:
                        db_form = e.get('reduced_formula', '')
                        if db_form and Composition(db_form).reduced_composition == target_reduced_comp:
                            exact_matches_entries.append(e)
                    except: continue
        
            if exact_matches_entries:
                print(f"[DEBUG] Strategy 0: Found {len(exact_matches_entries)} exact matches.")
                
                # --- FIX: Explicitly sort by Energy (Lowest is best) ---
                # We use 999.0 for entries with no energy so they go to the bottom
                print([[entry['id'], entry['energy_per_atom']] for entry in exact_matches_entries])
                exact_matches_entries.sort(key=lambda x: x.get('energy_per_atom') or 999.0)

                limit_exact = top_k * 2
                processing_subset = exact_matches_entries[:limit_exact]
                
                hydrated_candidates = []
                for e in processing_subset:
                    hydrated = self._hydrate_entry(e)
                    if hydrated: 
                        hydrated['method'] = 'Database Exact Match'
                        hydrated_candidates.append(hydrated)
                
                print(f"[DEBUG] Total time: {time.time()-t0:.4f}s")
                return self._finalize_candidates(hydrated_candidates, top_k)
            else:
                if len(entries) > 0:
                    examples = [e.get('reduced_formula', 'N/A') for e in entries[:5]]
                    print(f"[DEBUG] Strategy 0 Failed completely. Target: '{target_reduced_formula}'. Top DB Entries: {examples}...")

            # --- Strategy 1: Transmutation ---
            limit_transmute = 50
            print(f"[DEBUG] Strategy 1: Transmuting top {min(len(entries), limit_transmute)} stable prototypes...")
            
            t_trans = time.time()
            count_trans = 0
            for e in entries[:limit_transmute]:
                parent = self._hydrate_entry(e)
                print(e['id'], e['energy_per_atom'], parent['reduced_formula'])
                if not parent: continue
                
                candidate = self._transmute_structure(parent, target_comp)
                if candidate:
                    guesses.append(candidate)
                    count_trans += 1
            print(f"[DEBUG] Strategy 1 generated {count_trans} candidates in {time.time()-t_trans:.4f}s")
            
            if guesses:
                return self._finalize_candidates(guesses, top_k)

        # --- Strategy 2: Generalized Grouping ---
        print(f"[DEBUG] Strategy 2: Attempting generalized grouping...")
        t_group = time.time()
        mixed_candidates = self._attempt_generalized_grouping(target_comp)
        print(f"[DEBUG] Strategy 2 generated {len(mixed_candidates)} candidates in {time.time()-t_group:.4f}s")
        
        if mixed_candidates:
            guesses.extend(mixed_candidates)
            return self._finalize_candidates(guesses, top_k)

        # --- Strategy 3: Recursive Reduction ---
        print(f"[DEBUG] Strategy 3: Recursive Reduction/Doping...")
        t_red = time.time()
         
        base_entries, major_elements = self._find_base_entries(target_comp, num_bases=max_bases)

        if base_entries:
            doped_guesses = []
            
            # --- MODIFIED LOGIC FOR SYMMETRIZE ---
            if symmetrize:
                attempts_per_base = 1
            else:
                attempts_per_base = max(2, (top_k * 3) // len(base_entries))
            # -------------------------------------
            
            for base_entry in base_entries:
                base_formula = base_entry.get('reduced_formula', 'Unknown')
                base_id = base_entry.get('id', 'Unknown')
                print(f"[DEBUG] Strategy 3: Doping base structure {base_formula} ({base_id})...")
                
                for i in range(attempts_per_base):
                    try:
                        # --- UPDATED CALL ---
                        candidates_list = self._generate_doped_structure(
                            base_entry, 
                            major_elements, 
                            target_comp,
                            min_atoms=min_atoms if not symmetrize else 1,
                            randomize_scaling=randomize_scaling if not symmetrize else False,
                            symmetrize=symmetrize,
                            top_k=top_k
                        )
                        if candidates_list:
                            # Use extend because it now returns a list
                            doped_guesses.extend(candidates_list)
                        # --------------------
                    except Exception as e:
                        print(f"[DEBUG] Doping failed on iter {i} for {base_id}: {e}")
                        continue

            if doped_guesses:
                print(f"[DEBUG] Strategy 3: Successfully generated {len(doped_guesses)} total doped structures")
                guesses.extend(doped_guesses)
            else:
                print(f"[DEBUG] Strategy 3: WARNING - Base structures found but all doping attempts failed!")
        else:
            print("[DEBUG] Strategy 3: No suitable base structure found.")

        print(f"[DEBUG] Strategy 3 finished in {time.time()-t_red:.4f}s")
        print(f"[DEBUG] Total time: {time.time()-t0:.4f}s")
        return self._finalize_candidates(guesses, top_k)
        
    def _finalize_candidates(self, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Scale, Validate, and Deduplicate results.
        Operates on Dictionaries.
        """
        valid_candidates = []
        seen_hashes = set()
        
        print(f"[DEBUG] Finalizing {len(candidates)} candidates...")
        
        for entry in candidates:
            struct = entry['structure']
            
            # 1. Scale Lattice (Update structure in place or new object)
            scaled_struct = self._scale_lattice(struct)
            
            # 2. Sanity Check
            if self._sanity_check(scaled_struct):
                try:
                    sg = scaled_struct.get_space_group_info()[0]
                except:
                    sg = "Unknown"
                
                # --- NEW: Structural Fingerprint for Permutations ---
                # This multiplies the atomic number (Z) of each atom by its coordinates.
                # If two different elements swap places, this number changes, 
                # allowing us to tell FPS permutations apart before volume relaxation!
                try:
                    struct_fp = round(sum(site.specie.Z * sum(site.frac_coords) for site in scaled_struct), 2)
                except:
                    struct_fp = 0.0
                
                # Deduplication Hash
                comp_vol_hash = (
                    scaled_struct.composition.formula, 
                    round(scaled_struct.volume, 2), 
                    round(scaled_struct.density, 3),
                    scaled_struct.num_sites,
                    sg,
                    struct_fp  # <--- Added fingerprint
                )
                
                if comp_vol_hash not in seen_hashes:
                    # Update the entry with the scaled structure
                    final_entry = entry.copy()
                    final_entry['structure'] = scaled_struct
                    final_entry['space_group'] = sg
                    
                    valid_candidates.append(final_entry)
                    seen_hashes.add(comp_vol_hash)
        
        print(f"[DEBUG] Returning top {min(len(valid_candidates), top_k)} unique candidates.")
        return valid_candidates[:top_k]

    def _transmute_structure(self, parent_entry: Dict[str, Any], target_comp: Composition) -> Optional[Dict[str, Any]]:
        """
        Takes a parent ENTRY (dict), transmutes the structure, and returns a new ENTRY (dict)
        carrying over metadata.
        """
        parent_struct = parent_entry['structure']
        
        parent_elements = sorted(parent_struct.composition.elements, key=lambda e: e.X)
        target_elements = sorted(target_comp.elements, key=lambda e: e.X)

        if len(parent_elements) != len(target_elements):
            return None 

        species_map = {p: t for p, t in zip(parent_elements, target_elements)}

        try:
            new_struct = parent_struct.copy()
            new_struct.replace_species(species_map)
            
            # Create new wrapper
            new_entry = {
                'structure': new_struct,
                'id': f"transmuted_from_{parent_entry.get('id', 'unknown')}",
                'parent_id': parent_entry.get('id'),
                'parent_formula': parent_entry.get('reduced_formula'),
                'parent_space_group': parent_entry.get('spacegroup'),
                'energy_per_atom': None, # Invalidate energy
                'method': 'Transmutation'
            }
            return new_entry
        except ValueError:
            return None

    def _generate_doped_structure(self, base_entry: Dict[str, Any], major_elements: List[Element], target_comp: Composition, 
                                  min_atoms: int = 20, randomize_scaling: bool = False, symmetrize: bool = False, top_k: int = 1) -> Optional[List[Dict[str, Any]]]:
        """
        Generates doped supercell. Handles 1->N element mapping (Solid Solution Base).
        """
        base_struct = base_entry['structure']
        
        # 1. Base Structure Transmutation
        major_comp = Composition({e: 1.0 for e in major_elements}) 
        
        base_elements = sorted(base_struct.composition.elements, key=lambda e: e.X)
        major_elements_sorted = sorted(major_elements, key=lambda e: e.X)
        
        species_map = {}
        
        print(f"[DEBUG] Doping: Mapping Target Major {major_elements_sorted} -> Base Sites {base_elements}")

        if len(base_elements) == len(major_elements_sorted):
            species_map = {p: t for p, t in zip(base_elements, major_elements_sorted)}
        else:
            # Bucket Fill for Alloys (e.g. Base=Fe, Target=[Fe, Mn, Cr])
            if len(base_elements) == 1:
                print(f"[DEBUG] Doping: Detected Single-Site Base (e.g. Metal). Mixing all major elements on site.")
                target_mix = {el: 1.0/len(major_elements_sorted) for el in major_elements_sorted}
                species_map = {base_elements[0]: target_mix}
            else:
                print(f"[DEBUG] Doping: Complex mismatch. Attempting naive mapping.")
                for i, base_el in enumerate(base_elements):
                    if i < len(major_elements_sorted):
                        species_map[base_el] = major_elements_sorted[i]
        
        try:
            transmuted_base = base_struct.copy()
            transmuted_base.replace_species(species_map)
        except ValueError:
            print(f"[DEBUG] Doping: Transmutation failed.")
            return None

        # 2. Scaling Logic
        total_atoms_target = sum(target_comp.values())
        fractions = {e: target_comp.get(e)/total_atoms_target for e in target_comp.elements}
        
        valid_fractions = [f for f in fractions.values() if f > 0]
        min_fraction = min(valid_fractions) if valid_fractions else 1.0
        needed_atoms = int(math.ceil(1.0 / min_fraction))
        target_cluster_size = max(needed_atoms, min_atoms) 
        target_cluster_size = min(target_cluster_size, self.MAX_ATOMS)
        
        current_size = transmuted_base.num_sites
        
        if target_cluster_size > current_size:
            # --- SMART CUBIC SCALING WITH RANDOMIZATION ---
            a, b, c = transmuted_base.lattice.abc
            target_volume = (target_cluster_size / current_size) * transmuted_base.volume
            ideal_edge = target_volume ** (1/3)
            
            scale_a = max(1, int(round(ideal_edge / a)))
            scale_b = max(1, int(round(ideal_edge / b)))
            scale_c = max(1, int(round(ideal_edge / c)))
            
            while (scale_a * scale_b * scale_c * current_size) < target_cluster_size:
                current_edges = [scale_a * a, scale_b * b, scale_c * c]
                min_idx = current_edges.index(min(current_edges))
                if min_idx == 0: scale_a += 1
                elif min_idx == 1: scale_b += 1
                else: scale_c += 1
                
            # Randomize scaling if the flag is passed
            if randomize_scaling:
                # Randomly add 0 or 1 to each dimension to create varying cell shapes
                scale_a += random.randint(0, 1)
                scale_b += random.randint(0, 1)
                scale_c += random.randint(0, 1)
                
            scaling_matrix = [scale_a, scale_b, scale_c]
        else:
            scaling_matrix = [1, 1, 1]
            
        print(f"[DEBUG] Doping: Scaling base {scaling_matrix} to approx {target_cluster_size} atoms.")
        
        supercell = transmuted_base.copy()
        supercell.make_supercell(scaling_matrix)

        # --- EXACT MATCH BYPASS ---
        host_elements = set(major_elements)
        dopants = [e for e in target_comp.elements if e not in host_elements]
        
        if not dopants and supercell.composition.reduced_composition == target_comp.reduced_composition:
            print("[DEBUG] Exact composition match detected. Bypassing targeted substitution.")
            return [{
                'structure': supercell,
                'id': f"doped_from_{base_entry.get('id', 'unknown')}",
                'parent_id': base_entry.get('id'),
                'parent_formula': base_entry.get('reduced_formula'),
                'parent_space_group': base_entry.get('parent_space_group', 'Unknown'),
                'energy_per_atom': None,
                'method': 'Exact Match (Unmodified Supercell)'
            }]
        
        # 3. Targeted Doping & Substitution Logic
        total_sites = supercell.num_sites
        goal_counts = {e: int(round(fractions[e] * total_sites)) for e in target_comp.elements}
        
        # Balance rounding errors to ensure exact site count
        current_sum = sum(goal_counts.values())
        diff = total_sites - current_sum 
        most_abundant = max(fractions, key=fractions.get)
        if most_abundant in host_elements:
            goal_counts[most_abundant] += diff
        else:
            for h in host_elements: 
                goal_counts[h] = goal_counts.get(h, 0) + diff
                break

        # Calculate how many of each host are currently in the supercell
        current_counts = {el: 0 for el in host_elements}
        for site in supercell:
            current_counts[site.specie] += 1
            
        excess_hosts = {el: current_counts[el] - goal_counts.get(el, 0) for el in host_elements if current_counts[el] > goal_counts.get(el, 0)}

        # 4. Radii and Chemical Similarity Check
        host_radii = [e.atomic_radius for e in host_elements if e.atomic_radius]
        host_radius = np.mean(host_radii) if host_radii else 1.5
        
        substitution_tasks = []
        interstitial_tasks = []
        
        for dopant in dopants:
            count = goal_counts.get(dopant, 0)
            if count == 0: continue
            
            d_radius = dopant.atomic_radius or host_radius
            ratio = d_radius / host_radius
            
            # Priority 1: Check for same periodic group
            same_group_hosts = [h for h in excess_hosts.keys() if getattr(h, 'group', -1) == getattr(dopant, 'group', -2)]
            
            if same_group_hosts:
                target_host = same_group_hosts[0]
                is_subst = True
                print(f"[DEBUG] Doping: {dopant.symbol} shares a group with {target_host.symbol}. Forcing SUBSTITUTIONAL.")
            elif ratio < 0.65:
                is_subst = False
                print(f"[DEBUG] Doping: Element {dopant.symbol} (r={d_radius:.3f} Å) is INTERSTITIAL (Ratio: {ratio:.2f})")
            else:
                is_subst = True
                # Fallback: Substitute the host with the closest electronegativity
                target_host = min(excess_hosts.keys(), key=lambda h: abs(h.X - dopant.X))
                print(f"[DEBUG] Doping: Element {dopant.symbol} is SUBSTITUTIONAL. Target: {target_host.symbol} (closest electronegativity).")

            if is_subst:
                excess_hosts[target_host] -= count
                substitution_tasks.append((dopant, target_host, count))
            else:
                interstitial_tasks.extend([dopant] * count)

        # 5. Targeted Substitution (Symmetrized or Farthest-Point)
        if substitution_tasks:
            symmetrize_success = False
            
            if symmetrize:
                print(f"[DEBUG] Doping: Applying sequential symmetry-preserving enumeration (symmetrize=True)...")
                
                # Group substitutions by host to handle multiple dopants on the same site type
                host_to_mix = defaultdict(dict)
                for dopant, host, count in substitution_tasks:
                    host_to_mix[host][dopant] = count

                # --- UPFRONT COMBINATORIAL CHECK ---
                # Impede it to hang for minutes/hours.
                MAX_COMBINATIONS = 200_000 
                space_too_large = False
                
                for host, dopant_counts in host_to_mix.items():
                    # Count how many sites this host occupies in the pristine supercell
                    n_sites = sum(1 for site in supercell if max(site.species, key=site.species.get) == host)
                    
                    # Calculate the multinomial combinations
                    comb = 1
                    n_rem = n_sites
                    for dopant, count in dopant_counts.items():
                        if count > 0 and n_rem >= count:
                            comb *= math.comb(n_rem, count)
                            n_rem -= count
                            
                    print(f"[DEBUG] Doping: Sublattice {host.symbol} has ~{comb:,} theoretical combinations.")
                    
                    if comb > MAX_COMBINATIONS:
                        print(f"[WARNING] Sublattice {host.symbol} exceeds enumeration threshold ({comb:,} > {MAX_COMBINATIONS:,}).")
                        space_too_large = True
                        break
                
                # Decide whether to enumerate or immediately bail to FPS
                if space_too_large:
                    print("[DEBUG] Bypassing enumlib to prevent hanging. Forcing Farthest-Point Sampling fallback.")
                    symmetrize_success = False
                else:
                    # Start with a list containing just the pristine ordered supercell
                    current_candidates = [supercell.copy()]
                    
                    try:
                        # Process one host sublattice at a time (Sequential Doping)
                        for host_idx, (host, dopant_counts) in enumerate(host_to_mix.items()):
                            print(f"[DEBUG] Doping: Enumerating substitutions for host {host.symbol}...")
                            next_candidates = []

                            # Keep a small beam width for intermediate steps to prevent branching explosion.
                            # Only expand to the full top_k on the final sublattice.
                            is_last_host = (host_idx == len(host_to_mix) - 1)
                            beam_width = top_k if is_last_host else min(3, top_k)

                            for current_struct in current_candidates:
                                disordered_step = current_struct.copy()

                                # 1. Gather sites for the CURRENT host only
                                candidate_sites = [
                                    i for i, site in enumerate(disordered_step) 
                                    if max(site.species, key=site.species.get) == host
                                ]
                                total_sites = len(candidate_sites)

                                if total_sites == 0:
                                    next_candidates.append(current_struct)
                                    continue

                                total_dopants = sum(dopant_counts.values())
                                remaining_host = max(0, total_sites - total_dopants)

                                # 2. Create fractional composition for these specific sites
                                mix_dict = {host.symbol: remaining_host / total_sites}
                                for dopant, count in dopant_counts.items():
                                    mix_dict[dopant.symbol] = count / total_sites

                                # 3. Apply fractions ONLY to the current host's sites
                                for site_idx in candidate_sites:
                                    disordered_step.replace(site_idx, mix_dict)

                                # 4. Enumerate this specific sublattice
                                est = EnumerateStructureTransformation(max_cell_size=1, min_cell_size=1)
                                ordered_list = est.apply_transformation(disordered_step, return_ranked_list=beam_width)

                                if ordered_list:
                                    for d in ordered_list:
                                        next_candidates.append(d['structure'])
                                else:
                                    next_candidates.append(current_struct)

                            # Keep only the best structures for the next sublattice step
                            current_candidates = next_candidates[:top_k]

                        # Finalize the sequentially enumerated candidates
                        if current_candidates:
                            symmetrize_success = True
                            enumerated_candidates = []
                            for rank, struct in enumerate(current_candidates):
                                enumerated_candidates.append({
                                    'structure': struct,
                                    'id': f"doped_from_{base_entry.get('id', 'unknown')}_enum_{rank+1}",
                                    'parent_id': base_entry.get('id'),
                                    'parent_formula': base_entry.get('reduced_formula'),
                                    'parent_space_group': base_entry.get('parent_space_group', 'Unknown'),
                                    'energy_per_atom': None,
                                    'method': 'Sequential Symmetrized Enumeration'
                                })
                            return enumerated_candidates
                        else:
                            raise ValueError("Sequential enumeration returned an empty list.")

                    except Exception as e:
                        print(f"[WARNING] Sequential enumeration failed: {e}. Falling back to Farthest-Point Sampling.")
                        symmetrize_success = False

            # Farthest-Point Sampling Fallback (runs if --symmetrize is omitted OR if enumeration fails)
            if not symmetrize_success:
                print(f"[DEBUG] Doping: Applying targeted farthest-point substitution to generate {top_k} candidates...")
                fps_candidates = []
                
                # Generate 'top_k' distinct structures
                for attempt in range(top_k):
                    # ALWAYS START WITH A FRESH COPY OF THE PRISTINE SUPERCELL
                    current_supercell = supercell.copy()
                    dist_matrix = current_supercell.distance_matrix
                    
                    for dopant, host, count in substitution_tasks:
                        # Safely identify candidate sites
                        candidate_sites = [
                            i for i, site in enumerate(current_supercell) 
                            if max(site.species, key=site.species.get) == host
                        ]
                        placed_sites = []
                        
                        for _ in range(count):
                            if not candidate_sites: 
                                print(f"[WARNING] Not enough {host.symbol} sites to substitute with {dopant.symbol}!")
                                break
                                
                            if not placed_sites:
                                # Randomizing the first site ensures every 'attempt' yields a totally different overall configuration
                                chosen_site = random.choice(candidate_sites)
                            else:
                                best_site = candidate_sites[0]
                                max_min_dist = -1
                                for site_idx in candidate_sites:
                                    min_dist = min([dist_matrix[site_idx][p] for p in placed_sites])
                                    if min_dist > max_min_dist:
                                        max_min_dist = min_dist
                                        best_site = site_idx
                                chosen_site = best_site
                                
                            current_supercell.replace(chosen_site, dopant)
                            placed_sites.append(chosen_site)
                            candidate_sites.remove(chosen_site)        

                    # Handle Interstitials if they exist
                    if interstitial_tasks:
                        current_supercell = self._insert_interstitials(current_supercell, interstitial_tasks)

                    # Save the candidate
                    fps_candidates.append({
                        'structure': current_supercell,
                        'id': f"doped_from_{base_entry.get('id', 'unknown')}_fps_{attempt+1}",
                        'parent_id': base_entry.get('id'),
                        'parent_formula': base_entry.get('reduced_formula'),
                        'parent_space_group': base_entry.get('parent_space_group', 'Unknown'),
                        'energy_per_atom': None,
                        'method': 'Doping (Interstitial FPS)' if interstitial_tasks else 'Doping (Substitutional FPS)'
                    })
                    
                return fps_candidates
        # 6. Interstitial-Only Insertion (Runs if there were NO substitution tasks)
        if interstitial_tasks:
            print(f"[DEBUG] Doping: Applying interstitial insertion to generate {top_k} candidates...")
            interstitial_candidates = []
            
            for attempt in range(top_k):
                current_supercell = supercell.copy()
                current_supercell = self._insert_interstitials(current_supercell, interstitial_tasks)
                
                interstitial_candidates.append({
                    'structure': current_supercell,
                    'id': f"doped_from_{base_entry.get('id', 'unknown')}_interstitial_{attempt+1}",
                    'parent_id': base_entry.get('id'),
                    'parent_formula': base_entry.get('reduced_formula'),
                    'parent_space_group': base_entry.get('parent_space_group', 'Unknown'),
                    'energy_per_atom': None,
                    'method': 'Doping (Interstitial Only)'
                })
            return interstitial_candidates
            
        # 7. Failsafe Return (If no dopants or interstitials were somehow processed)
        return [{
            'structure': supercell,
            'id': f"doped_from_{base_entry.get('id', 'unknown')}",
            'parent_id': base_entry.get('id'),
            'parent_formula': base_entry.get('reduced_formula'),
            'parent_space_group': base_entry.get('parent_space_group', 'Unknown'),
            'energy_per_atom': None,
            'method': 'Doping (Unmodified)'
        }] 
    
    def _insert_interstitials(self, struct: Structure, species_list: List[Element]) -> Structure:
        """
        Inserts atoms into voids using Monte Carlo sampling with decaying distance thresholds.
        
        This method attempts to place interstitial atoms in void spaces within the crystal structure.
        It uses a hierarchical approach with progressively smaller distance thresholds to find suitable
        insertion points, falling back to random placement if no void is found.
        
        Args:
            struct: The pymatgen Structure object to modify
            species_list: List of Element objects to insert as interstitials
            
        Returns:
            The modified Structure with interstitial atoms added
            
        Raises:
            No explicit exceptions raised, but may propagate pymatgen errors
        """
        print(f"[DEBUG] Interstitial Insertion: Starting with {len(species_list)} species to insert into structure with {struct.num_sites} sites.")
        
        # Input validation
        if not species_list:
            print("[DEBUG] Interstitial Insertion: No species to insert, returning original structure.")
            return struct
        
        if struct.num_sites == 0:
            print("[WARNING] Interstitial Insertion: Structure has no sites, cannot determine voids.")
            # Still proceed with fallback insertion
        
        matrix = struct.lattice.matrix
        if matrix is None:
            raise ValueError("Structure lattice matrix is None")
        
        # Thresholds: Ideal -> Tight -> Forced (progressively smaller voids)
        THRESHOLDS = [1.1, 0.9, 0.7, 0.5]
        MAX_ATTEMPTS_PER_THRESHOLD = 100
        
        print(f"[DEBUG] Interstitial Insertion: Using thresholds {THRESHOLDS} for void detection.")
        
        for i, specie in enumerate(species_list):
            print(f"[DEBUG] Interstitial Insertion: Attempting to insert specie {specie.symbol} ({i+1}/{len(species_list)})")
            inserted = False
            
            for threshold in THRESHOLDS:
                print(f"[DEBUG] Interstitial Insertion: Trying threshold {threshold} Å for {specie.symbol}")
                attempts = 0
                for _ in range(MAX_ATTEMPTS_PER_THRESHOLD):
                    attempts += 1
                    rand_frac = np.random.rand(3)
                    cart_point = np.dot(rand_frac, matrix)
                    
                    try:
                        neighbors = struct.get_sites_in_sphere(cart_point, threshold)
                        if not neighbors:
                            struct.append(specie, rand_frac)
                            inserted = True
                            print(f"[DEBUG] Interstitial Insertion: Successfully inserted {specie.symbol} at fractional coords {rand_frac} after {attempts} attempts with threshold {threshold} Å.")
                            break
                    except Exception as e:
                        print(f"[WARNING] Interstitial Insertion: Error during neighbor search: {e}")
                        continue
                        
                if inserted:
                    break
                else:
                    print(f"[DEBUG] Interstitial Insertion: Failed to insert {specie.symbol} with threshold {threshold} Å after {MAX_ATTEMPTS_PER_THRESHOLD} attempts.")
            
            if not inserted:
                # Absolute fallback - random placement regardless of neighbors
                fallback_frac = np.random.rand(3)
                try:
                    struct.append(specie, fallback_frac)
                    print(f"[DEBUG] Interstitial Insertion: Absolute fallback - inserted {specie.symbol} at random fractional coords {fallback_frac}.")
                except Exception as e:
                    print(f"[ERROR] Interstitial Insertion: Failed to append {specie.symbol}: {e}")
                    raise
        
        print(f"[DEBUG] Interstitial Insertion: Completed. Structure now has {struct.num_sites} sites.")
        return struct

    def _find_base_entry(self, target_comp: Composition) -> Tuple[Optional[Dict[str, Any]], List[Element]]:
        """
        Finds chemically relevant base entry by stripping minority elements.
        Enhanced for steel compositions with better hierarchical reduction.
        Includes logic to prioritize mass coverage to prevent over-stripping.
        """
        elements_sorted = sorted(target_comp.elements, key=lambda e: target_comp.get_atomic_fraction(e))
        amounts = {e: target_comp.get(e) for e in target_comp.elements}

        # Get total atoms for mass coverage calculation
        total_atoms_initial = sum(amounts.values())
        element_fractions = {e.symbol: amounts[e]/total_atoms_initial for e in amounts.keys()}

        best_entry = None
        best_elements = []
        min_rmsd = float('inf')
        best_mass_coverage = 0.0

        print(f"[DEBUG] Reduction: Starting with composition {target_comp.reduced_formula}")
        print(f"[DEBUG] Reduction: Element fractions: {', '.join([f'{sym}={frac:.3f}' for sym, frac in sorted(element_fractions.items(), key=lambda x: -x[1])])}")
        print(f"[DEBUG] Reduction: Hierarchical reduction order (smallest first): {[e.symbol for e in elements_sorted]}")

        # Try removing elements one by one (smallest first)
        for i in range(len(elements_sorted) + 1):
            if not amounts: break

            # Current State
            current_elements = list(amounts.keys())
            current_subset_comp = Composition(amounts)
            
            # Calculate how much of the original alloy is represented here
            current_total = sum(amounts.values())
            current_coverage = current_total / total_atoms_initial

            # Create dummy to get Anonymous Formula (e.g. "AB")
            dummy_comp = Composition({e: 1.0 for e in amounts.keys()})
            anon_formula = dummy_comp.anonymized_formula

            print(f"[DEBUG] Reduction Step {i}: Testing subset {current_subset_comp.reduced_formula} -> Class {anon_formula} (Coverage: {current_coverage:.1%})")

            entries = self._get_entries(anon_formula)

            if entries:
                print(f"[DEBUG]   -> Found {len(entries)} prototypes for class {anon_formula}")

                # Rank by chemical similarity to the CURRENT SUBSET
                ranked = self._rank_prototypes_by_similarity(entries, current_subset_comp)

                # Check top 5 for RMSD
                prototypes_to_check = []
                for e in ranked[:5]:
                    s = self._hydrate_entry(e)
                    if s: prototypes_to_check.append((e, s))

                if prototypes_to_check:
                    proto_ratios = self._get_ratios_from_anon(anon_formula)
                    current_values = list(amounts.values())
                    rmsd = self._calculate_rmsd(sorted(proto_ratios), sorted(current_values))

                    top_formula = prototypes_to_check[0][0].get('reduced_formula')
                    
                    # --- NEW DECISION LOGIC ---
                    # We update the best entry if:
                    # 1. First valid match found
                    # 2. OR RMSD is significantly better (improved by > 0.1)
                    # 3. OR RMSD is effectively better (lower) AND we haven't lost too much mass (< 10% loss)
                    
                    # Heuristics
                    is_significant_improvement = (rmsd < min_rmsd - 0.1)
                    # We allow a small coverage drop (0.1) for a better RMSD, but prevent huge drops (e.g. 0.98 -> 0.50)
                    is_acceptable_mass_loss = (best_mass_coverage - current_coverage < 0.1)
                    
                    update = False
                    
                    if best_entry is None:
                         update = True
                         print(f"[DEBUG]   -> First match: {top_formula} (RMSD: {rmsd:.3f}). Setting as baseline.")
                    elif is_significant_improvement:
                         update = True
                         print(f"[DEBUG]   -> Significant RMSD improvement ({min_rmsd:.3f} -> {rmsd:.3f}). Updating despite coverage change.")
                    elif (rmsd < min_rmsd) and is_acceptable_mass_loss:
                         update = True
                         print(f"[DEBUG]   -> Better RMSD ({rmsd:.3f}) with acceptable mass loss. Updating.")
                    elif (rmsd < min_rmsd) and not is_acceptable_mass_loss:
                         update = False
                         print(f"[DEBUG]   -> Lower RMSD found ({rmsd:.3f}), but mass coverage dropped too much ({best_mass_coverage:.1%} -> {current_coverage:.1%}). Keeping previous base.")
                    else:
                         print(f"[DEBUG]   -> RMSD {rmsd:.3f} not better than current best {min_rmsd:.3f}, skipping")

                    if update:
                        min_rmsd = rmsd
                        best_elements = list(amounts.keys())
                        best_mass_coverage = current_coverage
                        best_entry = prototypes_to_check[0][0]

                        try:
                            # Get the hydrated structure we just checked
                            base_struct = prototypes_to_check[0][1]['structure']
                            # Calculate the initial space group 
                            # Symprec to match MP classifications
                            sga = SpacegroupAnalyzer(base_struct, symprec=0.1)
                            parent_sg = sga.get_space_group_symbol()
                            
                            best_entry['parent_space_group'] = parent_sg
                        except:
                            best_entry['parent_space_group'] = "Unknown"

                        print(f"[DEBUG]   -> *** New Best Base Found! {best_entry.get('reduced_formula')} ***")
                else:
                    print(f"[DEBUG]   -> No valid prototypes could be hydrated")
            else:
                print(f"[DEBUG]   -> No prototypes found for class {anon_formula}")

            # Remove smallest element for next iteration
            remaining = [e for e in elements_sorted if e in amounts]
            if not remaining: break

            removed_el = remaining[0]
            removed_fraction = amounts[removed_el] / sum(amounts.values())
            del amounts[removed_el]
            print(f"[DEBUG]   -> Removing minority element: {removed_el.symbol} (fraction: {removed_fraction:.3f})")

        if best_entry:
            print(f"[DEBUG] Reduction: Final best base structure: {best_entry.get('reduced_formula')} with elements {[e.symbol for e in best_elements]}")
        else:
            print(f"[DEBUG] Reduction: WARNING - No suitable base structure found after trying all reduction paths!")
            print(f"[DEBUG] Reduction: This likely means the database lacks simple structures for these elements.")

        return best_entry, best_elements
    
    def _find_base_entries(self, target_comp: Composition, num_bases: int = 3) -> Tuple[List[Dict[str, Any]], List[Element]]:
        """
        Finds multiple chemically relevant base entries by stripping minority elements.
        Returns a list of the top `num_bases` entries and the major elements used.
        """
        elements_sorted = sorted(target_comp.elements, key=lambda e: target_comp.get_atomic_fraction(e))
        amounts = {e: target_comp.get(e) for e in target_comp.elements}

        total_atoms_initial = sum(amounts.values())
        element_fractions = {e.symbol: amounts[e]/total_atoms_initial for e in amounts.keys()}

        best_entries = []
        best_elements = []
        min_rmsd = float('inf')
        best_mass_coverage = 0.0

        print(f"[DEBUG] Reduction: Starting with composition {target_comp.reduced_formula}")

        for i in range(len(elements_sorted) + 1):
            if not amounts: break

            current_subset_comp = Composition(amounts)
            current_coverage = sum(amounts.values()) / total_atoms_initial

            dummy_comp = Composition({e: 1.0 for e in amounts.keys()})
            anon_formula = dummy_comp.anonymized_formula

            print(f"[DEBUG] Reduction Step {i}: Testing subset {current_subset_comp.reduced_formula} -> Class {anon_formula} (Coverage: {current_coverage:.1%})")

            entries = self._get_entries(anon_formula)

            if entries:
                ranked = self._rank_prototypes_by_similarity(entries, current_subset_comp)
                
                # Check top N for RMSD and hydration
                prototypes_to_check = []
                for e in ranked[:num_bases * 2]: # Look at a slightly larger pool to ensure we get enough hydrated ones
                    s = self._hydrate_entry(e)
                    if s: prototypes_to_check.append((e, s))

                if prototypes_to_check:
                    proto_ratios = self._get_ratios_from_anon(anon_formula)
                    current_values = list(amounts.values())
                    rmsd = self._calculate_rmsd(sorted(proto_ratios), sorted(current_values))
                    
                    is_significant_improvement = (rmsd < min_rmsd - 0.1)
                    is_acceptable_mass_loss = (best_mass_coverage - current_coverage < 0.1)
                    
                    update = False
                    
                    if not best_entries:
                         update = True
                         print(f"[DEBUG]   -> First match found (RMSD: {rmsd:.3f}). Setting as baseline.")
                    elif is_significant_improvement:
                         update = True
                         print(f"[DEBUG]   -> Significant RMSD improvement ({min_rmsd:.3f} -> {rmsd:.3f}). Updating.")
                    elif (rmsd < min_rmsd) and is_acceptable_mass_loss:
                         update = True
                         print(f"[DEBUG]   -> Better RMSD ({rmsd:.3f}) with acceptable mass loss. Updating.")

                    if update:
                        min_rmsd = rmsd
                        best_elements = list(amounts.keys())
                        best_mass_coverage = current_coverage
                        
                        # Collect the top N valid entries
                        best_entries = []
                        for entry_dict, hydrated_dict in prototypes_to_check[:num_bases]:
                            entry = entry_dict.copy()
                            try:
                                sga = SpacegroupAnalyzer(hydrated_dict['structure'], symprec=0.1)
                                entry['parent_space_group'] = sga.get_space_group_symbol()
                            except:
                                entry['parent_space_group'] = "Unknown"
                            best_entries.append(entry)
                            
                        formulas = [e.get('reduced_formula') for e in best_entries]
                        print(f"[DEBUG]   -> *** New Best Bases Found! {formulas} ***")

            # Remove smallest element for next iteration
            remaining = [e for e in elements_sorted if e in amounts]
            if not remaining: break
            del amounts[remaining[0]]

        return best_entries, best_elements
    def _attempt_generalized_grouping(self, target_comp: Composition) -> List[Dict[str, Any]]:
        """Handles complex mixing. Returns list of hydrated entries with METADATA."""
        
        total_atoms = sum(target_comp.values())
        target_num_sites = round(total_atoms)
        
        print(f"[DEBUG] Grouping: Target sums to {total_atoms:.2f} atoms (~{target_num_sites} sites).")
        
        search_keys = []
        if 1.9 < target_num_sites < 2.1: search_keys = ["AB"]
        elif 2.9 < target_num_sites < 3.1: search_keys = ["AB2", "ABC"]
        elif 3.9 < target_num_sites < 4.1: search_keys = ["ABC2", "AB3"]
        elif 4.9 < target_num_sites < 5.1: search_keys = ["ABC3", "A2B3", "AB2C2"]
        elif 6.9 < target_num_sites < 7.1: search_keys = ["AB2C4", "A2B2C3", "ABC5"]
        
        candidates = []
        target_elements_sorted = sorted(target_comp.elements, key=lambda e: e.X)

        for key in search_keys:
            entries = self._get_entries(key)
            if not entries: continue
            
            # --- NEW: RANK BY SIMILARITY ---
            ranked = self._rank_prototypes_by_similarity(entries, target_comp)
            prototypes_to_try = ranked[:3]
            
            if prototypes_to_try:
                print(f"[DEBUG] Grouping: Selected top 3 parents for {key}: {[p.get('reduced_formula') for p in prototypes_to_try]}")

            for e in prototypes_to_try:
                proto_entry = self._hydrate_entry(e)
                if not proto_entry: continue
                proto = proto_entry['structure']
                # Calculate the scaling factor between Parent and Target
                parent_num_sites = proto.num_sites
                target_total_atoms = sum(target_comp.values())
                
                if target_total_atoms == 0: continue
                scale_factor = parent_num_sites / target_total_atoms
                
                # Create local counts scaled to fill THIS specific parent
                # e.g., if Parent=20 atoms and Target=5 atoms, we multiply all target counts by 4.
                local_target_counts = {
                    el: target_comp.get(el) * scale_factor 
                    for el in target_elements_sorted
                }
                # Capture Parent Metadata
                parent_id = proto_entry.get('id', 'unknown')
                parent_form = proto_entry.get('reduced_formula', 'unknown')
                try:
                    sga = SpacegroupAnalyzer(proto, symprec=0.1)
                    parent_sg = sga.get_space_group_symbol()
                except Exception:
                    parent_sg = "Unknown"
                print(f"[DEBUG] Grouping: Trying parent {parent_form} (ID: {parent_id})")
                # BUCKET FILL
                proto_unique_elements = sorted(proto.composition.elements, key=lambda e: e.X)
                mapping = {}
                t_idx = 0
                current_target_el = target_elements_sorted[t_idx]
                current_target_rem = local_target_counts[current_target_el]
                success = True
                
                for p_el in proto_unique_elements:
                    p_amt = proto.composition.get(p_el)
                    bucket_content = defaultdict(float)
                    filled = 0.0
                    while filled < p_amt - 0.001:
                        if current_target_rem <= 0.001:
                            t_idx += 1
                            if t_idx >= len(target_elements_sorted):
                                success = False; break
                            current_target_el = target_elements_sorted[t_idx]
                            current_target_rem = local_target_counts[current_target_el]
                        take = min(p_amt - filled, current_target_rem)
                        bucket_content[current_target_el] += take
                        filled += take
                        current_target_rem -= take
                    final_mix = {el: amt/p_amt for el, amt in bucket_content.items()}
                    mapping[p_el] = final_mix
                    if not success: 
                        print(f"[DEBUG] Grouping: Failed to fill bucket for {p_el} in {parent_form}")
                        break
                
                if success:
                    new_struct = proto.copy()
                    for site in new_struct:
                        site.species = mapping[max(site.species, key=site.species.get)]
                    
                    ordered_list = self._order_disordered_structure(new_struct)
                    
                    for s in ordered_list:
                        candidates.append({
                            'structure': s,
                            'id': f"grouped_from_{proto_entry.get('id')}",
                            'parent_id': parent_id,
                            'parent_formula': parent_form, 
                            'parent_space_group': parent_sg,
                            'energy_per_atom': None,
                            'method': f'Generalized Grouping (Base: {parent_form})'
                        })
                        if len(candidates) >= 5: return candidates
        return candidates
   
    def _rank_prototypes_by_similarity(self, entries: List[Dict], target_comp: Composition) -> List[Dict]:
        """
        Sorts entries by chemical relevance.
        UPDATED:
        1. Treats targets with < 10% anions as Alloys (Metal Path).
        2. Prioritizes chemical compatibility (element overlap) over energy.
        3. For oxides, strongly prefers structures with matching cation elements.
        """
        target_elements = set(target_comp.elements)
        target_symbols = {e.symbol for e in target_elements}

        # Define common anions
        common_anions = {'O', 'F', 'S', 'Cl', 'N', 'P', 'Br', 'I', 'Se', 'Te'}
        target_anions = {e.symbol for e in target_elements if e.symbol in common_anions}
        target_cations = {e.symbol for e in target_elements if e.symbol not in common_anions}

        # Calculate Anion Fraction
        total_atoms = sum(target_comp.values())
        anion_atoms = sum(target_comp.get(e) for e in target_elements if e.symbol in common_anions)
        anion_fraction = anion_atoms / total_atoms if total_atoms > 0 else 0

        # DECISION: Metal Path vs Oxide Path
        # If anions are present but < 10%, we treat it as an Alloy (Metal Path)
        is_target_oxide_or_salt = (len(target_anions) > 0) and (anion_fraction >= 0.10)

        path_str = "OXIDE/SALT" if is_target_oxide_or_salt else "METAL/ALLOY"
        # Only print this once per major call (heuristic to avoid spam in loops)
        if len(entries) > 10:
            print(f"[DEBUG] Ranking: Target is {path_str} (Anion Fraction: {anion_fraction:.2%})")

        scored_entries = []

        for entry in entries:
            f_str = entry.get('reduced_formula', '')
            energy = entry.get('energy_per_atom')
            if energy is None: energy = float('inf')

            # Parse prototype composition to get its elements
            try:
                proto_comp = Composition(f_str)
                proto_elements = {e.symbol for e in proto_comp.elements}
                proto_cations = {e.symbol for e in proto_comp.elements if e.symbol not in common_anions}
                proto_anions = {e.symbol for e in proto_comp.elements if e.symbol in common_anions}
            except:
                proto_elements = set()
                proto_cations = set()
                proto_anions = set()

            proto_has_anion = len(proto_anions) > 0

            score = 0

            # --- CRITERIA 1: CHEMICAL COMPATIBILITY (HIGHEST PRIORITY) ---
            # For oxides/salts: prioritize matching cation elements
            if is_target_oxide_or_salt:
                # Count how many cations match
                cation_overlap = len(target_cations & proto_cations)
                # CRITICAL: This is the most important factor for perovskites
                score += (cation_overlap * 500)  # Increased from 50 to 500

                # Bonus if ALL target cations are present
                if target_cations and target_cations.issubset(proto_cations):
                    score += 1000

                # Penalize if prototype has cations not in target (less relevant)
                extra_cations = len(proto_cations - target_cations)
                score -= (extra_cations * 100)

                # Check anion match
                target_main_anion = list(target_anions)[0] if target_anions else ''
                if target_main_anion and target_main_anion in proto_anions:
                    score += 300  # Increased from 200
                elif proto_has_anion:
                    score += 50
                else:
                    # Prototype has no anions but target does - bad match
                    score -= 2000
            else:
                # Metal Path: We WANT metallic parents with matching elements
                if proto_has_anion:
                    score -= 2000  # Heavily penalize oxides (increased from 1000)
                else:
                    score += 200   # Reward pure metals/intermetallics

                # Count element overlap (excluding anions)
                metal_overlap = len(target_cations & proto_cations)
                score += (metal_overlap * 500)  # Increased from 50 to 500

                # Bonus if ALL target metals are present
                if target_cations and target_cations.issubset(proto_cations):
                    score += 1000

            # --- CRITERIA 2: ENERGY (SECONDARY) ---
            # Energy is now much less important than chemical compatibility
            # Scale energy contribution to be smaller
            if energy != float('inf'):
                score -= (energy * 0.1)  # Reduced weight from 1.0 to 0.1

            scored_entries.append((score, entry))

        # Sort descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        # VERBOSE: Print top 3 winners to see why they won
        if len(scored_entries) > 0:
            top_3 = scored_entries[:3]
            print('[DEBUG] Ranking: Top 10 candidates (score, formula, id, energy) -> ',[[x[0], x[1]['reduced_formula'], x[1]['id'], x[1]['energy_per_atom']] for x in scored_entries[:10]])
            debug_strs = [f"{x[1].get('reduced_formula')}-{x[1].get('id')}(Score={x[0]:.1f})" for x in top_3]
            print(f"[DEBUG] Ranking: Top candidates -> {', '.join(debug_strs)}")

        return [x[1] for x in scored_entries]
        
    
    def _order_disordered_structure(self, disordered_struct: Structure) -> List[Structure]:
        import math
        import random
        from collections import defaultdict
        
        # --- Attempt 1: Rigorous Enumeration ---
        try:
            est = EnumerateStructureTransformation(max_cell_size=8, min_cell_size=1)
            ordered_list = est.apply_transformation(disordered_struct, return_ranked_list=3)
            if ordered_list:
                return [d['structure'] for d in ordered_list]
        except Exception:
            pass # Fail silently to fallback

        # --- Attempt 2: Manual Python Fallback ---
        try:
            min_fraction = 1.0
            for site in disordered_struct:
                for occ in site.species.values():
                    if occ < 0.999: min_fraction = min(min_fraction, occ)
            
            if min_fraction > 0.99: return [disordered_struct]

            expansion = int(round(1.0 / min_fraction))
            expansion = min(expansion, 8) 
            
            scaling_matrix = [expansion, 1, 1]
            if expansion == 4: scaling_matrix = [2, 2, 1]
            elif expansion >= 8: scaling_matrix = [2, 2, 2]
            
            s_super = disordered_struct.copy()
            s_super.make_supercell(scaling_matrix)
            
            s_ordered = s_super.copy()
            
            site_groups = defaultdict(list)
            for i, site in enumerate(s_super):
                comp_key = tuple(sorted([(el.symbol, occ) for el, occ in site.species.items()]))
                site_groups[comp_key].append(i)
                
            for comp_key, indices in site_groups.items():
                if len(comp_key) == 1: continue 
                
                n_sites = len(indices)
                atom_pool = []
                
                for el_sym, occ in comp_key:
                    count = int(round(occ * n_sites))
                    atom_pool.extend([el_sym] * count)
                
                while len(atom_pool) < n_sites: atom_pool.append(atom_pool[0]) 
                while len(atom_pool) > n_sites: atom_pool.pop()
                
                # --- SORT instead of SHUFFLE for better symmetry ---
                atom_pool.sort() 
                
                for i, el_sym in zip(indices, atom_pool):
                    s_ordered.replace(i, el_sym)
            
            return [s_ordered]

        except Exception as e:
            print(f"[ERROR] Manual ordering failed: {e}. Returning disordered structure.")
            return [disordered_struct]
    

    def _scale_lattice(self, struct: Structure) -> Structure:
        """Scales lattice using average atomic radius. Handles disordered structures safely."""
        radii = []
        for s in struct:
            if s.is_ordered:
                if s.specie.atomic_radius: radii.append(s.specie.atomic_radius)
            else:
                try:
                    avg_r = sum(el.atomic_radius * amt for el, amt in s.species.items() if el.atomic_radius)
                    total_occ = sum(s.species.values())
                    if total_occ > 0: radii.append(avg_r / total_occ)
                except Exception: continue

        if not radii: return struct
        
        avg_radius = np.mean(radii)
        estimated_packing = 0.60 
        ideal_vol_per_atom = (4/3 * np.pi * (avg_radius ** 3)) / estimated_packing
        ideal_total_vol = ideal_vol_per_atom * struct.num_sites
        
        if ideal_total_vol > struct.volume:
            struct.scale_lattice(ideal_total_vol)
            
        distances = struct.distance_matrix.flatten()
        non_zero_dists = distances[distances > 1e-5]
        
        if len(non_zero_dists) > 0:
            min_dist = np.min(non_zero_dists)
            min_allowed_dist = max(0.8, avg_radius * 0.5)
            if min_dist < min_allowed_dist:
                scale_factor = min_allowed_dist / min_dist
                scale_factor = min(scale_factor, 2.0)
                new_volume = struct.volume * (scale_factor ** 3)
                struct.scale_lattice(new_volume)
        return struct

    def _sanity_check(self, struct: Structure) -> bool:
        """Checks for density and atomic overlaps."""
        
        # 1. Density Check
        if struct.density < 0.1: 
            print(f"[DEBUG] Sanity Fail: Density too low ({struct.density:.2f})")
            return False
        if struct.density > 60:
            print(f"[DEBUG] Sanity Fail: Density too high ({struct.density:.2f})")
            return False
        
        if struct.num_sites <= 1: return True
            
        distances = struct.distance_matrix.flatten()
        non_zero_dists = distances[distances > 1e-5]
        
        if len(non_zero_dists) == 0: return True
            
        min_dist = np.min(non_zero_dists)
        
        # 2. Overlap Check
        if min_dist < 0.5: 
            print(f"[DEBUG] Sanity Fail: Atoms overlap too closely ({min_dist:.3f} < 0.5A)")
            return False
        
        return True
    def _get_ratios_from_anon(self, anon_str: str) -> List[float]:
        matches = re.findall(r'([A-Z])(\d*)', anon_str)
        counts = []
        for _, count in matches:
            counts.append(float(count) if count else 1.0)
        total = sum(counts)
        return [c / total for c in counts]

    def _calculate_rmsd(self, list1: List[float], list2: List[float]) -> float:
        total2 = sum(list2)
        norm2 = [x / total2 for x in list2]
        if len(list1) != len(norm2): return 999.0
        sq_diff = [(a - b)**2 for a, b in zip(list1, norm2)]
        return math.sqrt(sum(sq_diff) / len(list1))

    def _select_best_prototype(self, prototypes: List[Dict[str, Any]], current_comp: Composition) -> Optional[Dict[str, Any]]:
        for proto in prototypes:
            return proto 
        return prototypes[0] if prototypes else None
# Helper functions for data loading
def clean_array_string(s: str) -> str:
    if pd.isna(s): return "[]"
    s = str(s)
    s = re.sub(r'\)\s+array\(', '), array(', s)
    s = s.replace("array(", "").replace(")", "")
    s = s.replace("\n", ",").replace("\r", ",")
    while ",," in s: s = s.replace(",,", ",")
    s = re.sub(r',\s+,', ',', s)
    return s

def clean_species_string(s: str) -> str:
    if pd.isna(s): return "[]"
    s = str(s)
    s = re.sub(r"'\s+'", "', '", s)
    return s

def reconstruct_structure(row: pd.Series) -> Optional[Dict[str, Any]]:
    try:
        lattice_str = clean_array_string(row.get('lattice_vectors', '[]'))
        coords_str = clean_array_string(row.get('cartesian_site_positions', '[]'))
        species_str = clean_species_string(row.get('species_at_sites', '[]'))
        lattice_matrix = list(ast.literal_eval(lattice_str))
        coords = list(ast.literal_eval(coords_str))
        species = ast.literal_eval(species_str)
        if lattice_matrix and coords and species:
            struct = Structure(lattice=lattice_matrix, species=species, coords=coords, coords_are_cartesian=True)
            total_energy = row.get('energy', None)
            nsites = row.get('nsites', len(species))
            e_per_atom = None
            if total_energy is not None and nsites > 0:
                e_per_atom = float(total_energy) / float(nsites)
            
            # The structure in this dict does NOT have properties attached.
            # Metadata is in the dict keys.
            return {
                'structure': struct,
                'id': row.get('immutable_id', None),
                'reduced_formula': row.get('chemical_formula_reduced', struct.composition.reduced_formula),
                'energy_per_atom': e_per_atom
            }
    except Exception: return None
    return None

def build_library_index(csv_dir: str, limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    print(f"Loading dataset from {csv_dir}...")
    try:
        csv_files = sorted([os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')])
    except FileNotFoundError:
        print(f"Error: CSV directory '{csv_dir}' not found.")
        sys.exit(1)
    library_index = defaultdict(list)
    total_processed = 0
    print("Indexing structures by stoichiometry...")
    for file_path in tqdm(csv_files, desc="Processing CSV files"):
        try:
            chunk_iter = pd.read_csv(file_path, on_bad_lines='warn', quoting=csv.QUOTE_MINIMAL, chunksize=2000)
            for chunk in chunk_iter:
                for idx, row in chunk.iterrows():
                    try:
                        entry = reconstruct_structure(row)
                        if entry:
                            struct = entry['structure']
                            anon_formula = struct.composition.anonymized_formula
                            library_index[anon_formula].append(entry)
                            total_processed += 1
                            if limit and total_processed >= limit: break
                    except Exception: continue
                if limit and total_processed >= limit: break
            gc.collect()
            if limit and total_processed >= limit: break
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    print(f"Index built! Found {len(library_index)} unique stoichiometries with {total_processed} structures.")
    return library_index

if __name__ == "__main__":
    csv_dir = "../lemat_unique_csv_500_parts"

    # Build index (this may take time for full dataset)
    # library = build_library_index(csv_dir, limit=100)

    # Initialize guesser
    # guesser = ProtoCSP(library)

    # Test with a composition
    # candidates = guesser.guess("SrTiO3")
    # print(f"Generated {len(candidates)} candidates for SrTiO3")