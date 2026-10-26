#!/usr/bin/env python3
"""
TensorGenerator: High-throughput combinatorial structure generation for ProtoCSP.

This module provides a fast-path pipeline for fixed-topology scenarios, bypassing
the pymatgen CPU bottleneck for GPU-ready tensor generation. It decouples geometry
from chemistry to enable rapid combinatorial exploration of solid solutions and HEAs.

Usage:
    python main.py config.json --fast

The pipeline is triggered via the --fast CLI flag and ingests a JSON configuration
defining the grid search parameters.
"""

import os
import sys
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from itertools import product
import numpy as np

try:
    from pymatgen.core import Structure, Composition, Element
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:
    print("Error: 'pymatgen' library not found. Please run 'pip install pymatgen'")
    sys.exit(1)

try:
    from ase import Atoms
    from ase.io import write
    HAVE_ASE = True
except ImportError:
    HAVE_ASE = False
    print("Warning: ASE not found. Some export features may be limited.")


class TensorGenerator:
    """
    Fast tensor-based structure generator for fixed-topology scenarios.
    
    This class decouples geometry from chemistry to enable high-throughput
    combinatorial generation. Instead of creating unique structural objects for
    each variation, it parses geometry once and generates chemical diversity by
    shuffling 1D integer tensors representing atomic numbers (Z).
    
    Attributes:
        config (Dict): JSON configuration dictionary
        pos_tensor (np.ndarray): Frozen positions tensor [N, 3]
        cell_tensor (np.ndarray): Frozen cell tensor [3, 3]
        base_structure (Structure): The scaled base pymatgen Structure
        num_atoms (int): Total number of atoms in the supercell (N)
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the TensorGenerator with a JSON configuration file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config = self._load_config(config_path)
        self.pos_tensor = None
        self.cell_tensor = None
        self.base_structure = None
        self.num_atoms = 0
        self._element_to_Z = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate the JSON configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required = ['base_structure', 'samples_per_composition']
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        # Must have either composition_ranges OR site_substitutions
        has_comp_ranges = 'composition_ranges' in config
        has_site_subst = 'site_substitutions' in config
        
        if not has_comp_ranges and not has_site_subst:
            raise ValueError("Config must contain either 'composition_ranges' or 'site_substitutions'")
        
        if has_comp_ranges:
            # Validate composition_ranges structure
            comp_ranges = config['composition_ranges']
            if 'host' not in comp_ranges and 'substitutional' not in comp_ranges:
                raise ValueError("composition_ranges must contain 'host' and/or 'substitutional' keys")
        
        return config
    
    def parse_and_freeze_geometry(self) -> None:
        """
        Parse the base structure and freeze geometry into tensors.
        
        This method:
        1. Loads the base_structure CIF file
        2. Scales it to supercell_target_atoms (if specified)
        3. Extracts Cartesian coordinates into pos_tensor [N, 3]
        4. Extracts lattice vectors into cell_tensor [3, 3]
        
        The geometry is parsed exactly ONCE and then frozen for all compositions.
        """
        base_path = self.config['base_structure']
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base structure file not found: {base_path}")
        
        print(f"[TensorGen] Loading base structure from {base_path}...")
        
        # Load the base structure
        base_struct = Structure.from_file(base_path)
        base_struct.remove_oxidation_states()
        
        # --- NEW: Check for explicit dimensions first ---
        if 'supercell_dimensions' in self.config:
            scaling_matrix = self.config['supercell_dimensions']
            if isinstance(scaling_matrix, list) and len(scaling_matrix) == 3:
                print(f"[TensorGen] Applying explicit supercell dimensions: {scaling_matrix}")
                base_struct.make_supercell(scaling_matrix)
            else:
                raise ValueError("supercell_dimensions must be a list of 3 integers, e.g., [2, 2, 2]")
        else:
            # Fallback to the old target_atoms logic if dimensions aren't provided
            target_atoms = self.config.get('supercell_target_atoms', base_struct.num_sites)
            if target_atoms > base_struct.num_sites:
                base_struct = self._scale_structure_to_target(base_struct, target_atoms)
        # ----------------------------------------------

        self.base_structure = base_struct
        self.num_atoms = base_struct.num_sites
        
        # Extract frozen tensors
        # Position tensor: [N, 3] Cartesian coordinates
        self.pos_tensor = np.array(base_struct.cart_coords, dtype=np.float64)
        
        # Cell tensor: [3, 3] Lattice vectors
        self.cell_tensor = np.array(base_struct.lattice.matrix, dtype=np.float64)
        
        # --- NEW: Record original host element for Sublattice Masking ---
        host_elements_list = []
        for site in base_struct:
            if site.is_ordered:
                elem = str(site.specie)
            else:
                elem = str(max(site.species.keys(), key=lambda e: site.species[e]))
            host_elements_list.append(elem)
        self.host_elements = np.array(host_elements_list)
        # ---------------------------------------------------------------
        
        print(f"[TensorGen] Geometry frozen: {self.num_atoms} atoms")
        print(f"[TensorGen] Cell volume: {base_struct.volume:.2f} Å³")
        
    def _scale_structure_to_target(self, struct: Structure, target_atoms: int) -> Structure:
        """
        Scale a structure to approximately target_atoms using smart cubic scaling.
        
        Args:
            struct: Input pymatgen Structure
            target_atoms: Target number of atoms in the supercell
            
        Returns:
            Scaled pymatgen Structure
        """
        current_size = struct.num_sites
        
        if target_atoms <= current_size:
            return struct.copy()
        
        # Smart cubic scaling
        a, b, c = struct.lattice.abc
        target_volume = (target_atoms / current_size) * struct.volume
        ideal_edge = target_volume ** (1/3)
        
        scale_a = max(1, int(round(ideal_edge / a)))
        scale_b = max(1, int(round(ideal_edge / b)))
        scale_c = max(1, int(round(ideal_edge / c)))
        
        # Ensure we have enough atoms
        while (scale_a * scale_b * scale_c * current_size) < target_atoms:
            current_edges = [scale_a * a, scale_b * b, scale_c * c]
            min_idx = current_edges.index(min(current_edges))
            if min_idx == 0:
                scale_a += 1
            elif min_idx == 1:
                scale_b += 1
            else:
                scale_c += 1
        
        scaling_matrix = [scale_a, scale_b, scale_c]
        print(f"[TensorGen] Scaling structure by {scaling_matrix} to reach ~{target_atoms} atoms")
        
        supercell = struct.copy()
        supercell.make_supercell(scaling_matrix)
        
        return supercell
    
    def compute_discrete_compositions(self) -> List[Dict[str, int]]:
        """
        Compute all valid discrete compositions using the PARTIAL SUBSTITUTION model.
        
        In this model:
        - host fraction = percentage of host sites to SUBSTITUTE (not overall composition)
        - substitutional elements fill the substituted sites proportionally
        - remaining sites stay as the host element
        
        Example: host={Fe: 0.35}, substitutional={Cr: 0.4, Ni: 0.4, Co: 0.2}
        Means: 35% of all sites are substituted. Of those substituted sites:
        - 40% become Cr, 40% become Ni, 20% become Co
        - 65% of sites remain as Fe
        
        For N=100 atoms: 35 substituted (14 Cr, 14 Ni, 7 Co) + 65 Fe = 100 total
        
        Returns:
            List of dictionaries mapping element symbol to atom count
        """
        composition_ranges = self.config.get('composition_ranges', {})
        site_substitutions = self.config.get('site_substitutions', {})
        N = self.num_atoms
        
        # Determine mode based on config structure
        if site_substitutions:
            # MODE 3: SITE-SPECIFIC SUBSTITUTION (e.g., perovskites with multiple Wyckoff sites)
            # Each site type (K, Na, Al, F) has its own substitution pattern
            site_subst = {k: v for k, v in site_substitutions.items() if not k.startswith('_')}
            return self._compute_site_specific_compositions(site_subst, N)
        
        # MODE 1 & 2: Use composition_ranges
        if not composition_ranges:
            raise ValueError("Config must contain either 'composition_ranges' or 'site_substitutions'")
        
        # Extract host and substitutional ranges (filter out comment keys starting with _)
        host_ranges = {k: v for k, v in composition_ranges.get('host', {}).items() if not k.startswith('_')}
        subst_ranges = {k: v for k, v in composition_ranges.get('substitutional', {}).items() if not k.startswith('_')}
        
        # Determine mode: PARTIAL SUBSTITUTION vs FIXED COMPOSITION
        if len(host_ranges) == 1 and len(subst_ranges) > 0:
            # MODE 1: PARTIAL SUBSTITUTION (e.g., Fe-based HEA)
            # Single host element, substitutional elements fill substituted sites
            return self._compute_partial_substitution_compositions(host_ranges, subst_ranges, N)
        else:
            # MODE 2: FIXED COMPOSITION (e.g., perovskite ordering)
            # Multiple elements with fixed ratios, just shuffle positions
            return self._compute_fixed_composition_compositions(host_ranges, subst_ranges, N)
    
    def _compute_partial_substitution_compositions(self, host_ranges: Dict, subst_ranges: Dict, N: int) -> List[Dict[str, int]]:
        """
        Partial substitution model: host fraction = % of sites to substitute.
        
        Example: host={Fe: 0.35}, substitutional={Cr: 0.4, Ni: 0.4, Co: 0.2}
        Means: 35% of sites substituted. Of substituted sites: 40% Cr, 40% Ni, 20% Co.
        Result: 65% Fe + (35% * 40% Cr + 35% * 40% Ni + 35% * 20% Co)
        """
        host_elem = list(host_ranges.keys())[0]
        host_substitution_fractions = self._parse_range_spec(host_ranges[host_elem])
        
        # Parse substitutional ranges (fractions OF THE SUBSTITUTED SITES)
        subst_elements = []
        subst_element_ranges = {}
        
        for elem, range_spec in subst_ranges.items():
            subst_elements.append(elem)
            subst_element_ranges[elem] = self._parse_range_spec(range_spec)
        
        valid_compositions = []
        
        for host_subst_frac in host_substitution_fractions:
            if host_subst_frac < 0 or host_subst_frac > 1:
                continue
            
            n_substituted = int(round(N * host_subst_frac))
            n_remaining_host = N - n_substituted
            
            if n_substituted == 0:
                valid_compositions.append({host_elem: N})
                continue
            
            # Generate all substitutional element distributions
            subst_range_values = [subst_element_ranges[elem] for elem in subst_elements]
            
            for subst_fractions in product(*subst_range_values):
                subst_comp_frac = {elem: frac for elem, frac in zip(subst_elements, subst_fractions)}
                
                # Substitutional fractions must sum to 1.0 (fill all substituted sites)
                total_subst_frac = sum(subst_comp_frac.values())
                if abs(total_subst_frac - 1.0) > 0.01:
                    continue
                
                subst_discrete = self._fractions_to_discrete(subst_comp_frac, n_substituted)
                
                if not subst_discrete or sum(subst_discrete.values()) != n_substituted:
                    continue
                
                # Build full composition: remaining host + substituted elements
                full_composition = {host_elem: n_remaining_host}
                full_composition.update(subst_discrete)
                
                if sum(full_composition.values()) == N:
                    valid_compositions.append(full_composition)
        
        print(f"[TensorGen] Partial Substitution mode: {len(valid_compositions)} compositions")
        print(f"[TensorGen] Host: {host_elem}, substitution levels: {host_substitution_fractions}")
        if valid_compositions:
            print(f"[TensorGen] Example: {valid_compositions[0]}")
        
        return valid_compositions
    
    def _compute_fixed_composition_compositions(self, host_ranges: Dict, subst_ranges: Dict, N: int) -> List[Dict[str, int]]:
        """
        Fixed composition mode: All element fractions sum to 1.0 directly.
        
        Example: host={K: 0.33, Na: 0.17, Al: 0.17, F: 0.33}, substitutional={}
        Used for cation ordering studies where composition is fixed, only arrangement varies.
        """
        # Combine all elements
        all_elements = list(host_ranges.keys()) + list(subst_ranges.keys())
        all_ranges = {**host_ranges, **subst_ranges}
        
        # Parse ranges for all elements
        element_ranges = {}
        for elem, range_spec in all_ranges.items():
            element_ranges[elem] = self._parse_range_spec(range_spec)
        
        # Generate all combinations using Cartesian product
        range_values = [element_ranges[elem] for elem in all_elements]
        
        valid_compositions = []
        
        for fractions in product(*range_values):
            comp_frac = {elem: frac for elem, frac in zip(all_elements, fractions)}
            
            # All fractions must sum to ~1.0
            total_frac = sum(comp_frac.values())
            if abs(total_frac - 1.0) > 0.01:
                continue
            
            discrete_comp = self._fractions_to_discrete(comp_frac, N)
            
            if discrete_comp and sum(discrete_comp.values()) == N:
                valid_compositions.append(discrete_comp)
        
        print(f"[TensorGen] Fixed Composition mode: {len(valid_compositions)} compositions")
        if valid_compositions:
            print(f"[TensorGen] Example: {valid_compositions[0]}")
        
        return valid_compositions
    
    def _compute_site_specific_compositions(self, site_subst: Dict, N: int) -> List[Dict[str, int]]:
        """
        Site-specific substitution model for structures with multiple Wyckoff positions.
        
        Each site type (e.g., K sites, Al sites, F sites in perovskite) can be substituted
        independently. The base structure determines how many sites of each type exist.
        
        Example for K2NaAlF6:
        - K sites: substitution_fraction=[0.0, 0.5], substituents={Cs: 0.6, Rb: 0.4}
          At 50% substitution: half K sites become Cs/Rb mix (60% Cs, 40% Rb of substituted)
        - Al sites: substitution_fraction=[0.0, 1.0], substituents={Ti: 1.0}
          At 100% substitution: all Al sites become Ti
        
        Args:
            site_subst: Dict mapping site element to substitution spec
            N: Total number of atoms
            
        Returns:
            List of composition dictionaries
        """
        if not self.base_structure:
            raise ValueError("Base structure required for site-specific substitution")
        
        # Count sites of each type in the base structure
        site_counts = self._count_sites_by_type()
        
        # Verify all site types in config exist in structure
        for site_type in site_subst.keys():
            if site_type not in site_counts:
                available = list(site_counts.keys())
                raise ValueError(f"Site type '{site_type}' not found in structure. Available: {available}")
        
        # Build substitution options for each site type
        # Each site type can have multiple substitution levels
        site_options = {}
        
        for site_type, spec in site_subst.items():
            site_options[site_type] = []
            
            # Parse substitution fractions (levels of substitution)
            subst_fractions = self._parse_range_spec(spec['substitution_fraction'])
            
            # Parse substituent ratios
            substituents = {k: v for k, v in spec['substituents'].items() if not k.startswith('_')}
            
            total_sites = site_counts[site_type]
            
            for subst_frac in subst_fractions:
                if subst_frac < 0 or subst_frac > 1:
                    continue
                
                n_substituted = int(round(total_sites * subst_frac))
                n_remaining = total_sites - n_substituted
                
                if n_substituted == 0:
                    # No substitution, keep original element
                    site_options[site_type].append({site_type: total_sites})
                    continue
                
                # Generate substituent distributions
                subst_elem_ranges = {elem: self._parse_range_spec(ratio) for elem, ratio in substituents.items()}
                subst_elem_names = list(subst_elem_ranges.keys())
                subst_range_values = [subst_elem_ranges[elem] for elem in subst_elem_names]
                
                for subst_elem_fracs in product(*subst_range_values):
                    # Check if substituent fractions sum to 1.0
                    if abs(sum(subst_elem_fracs) - 1.0) > 0.01:
                        continue
                    
                    # Convert to discrete counts
                    subst_frac_dict = {elem: frac for elem, frac in zip(subst_elem_names, subst_elem_fracs)}
                    subst_discrete = self._fractions_to_discrete(subst_frac_dict, n_substituted)
                    
                    if not subst_discrete or sum(subst_discrete.values()) != n_substituted:
                        continue
                    
                    # Build composition for this site type
                    site_comp = {site_type: n_remaining}
                    site_comp.update(subst_discrete)
                    site_options[site_type].append(site_comp)
        
        # Generate all combinations across site types using Cartesian product
        site_types = list(site_options.keys())
        site_option_lists = [site_options[st] for st in site_types]
        
        valid_compositions = []
        
        for combination in product(*site_option_lists):
            # Keep compositions grouped by their sublattice
            sublattice_comp = {}
            for st, site_comp in zip(site_types, combination):
                sublattice_comp[st] = site_comp
            
            # Validate total atoms
            total_atoms = sum(sum(sc.values()) for sc in sublattice_comp.values())
            if total_atoms == N:
                valid_compositions.append(sublattice_comp)
        
        print(f"[TensorGen] Site-Specific Substitution mode: {len(valid_compositions)} compositions")
        print(f"[TensorGen] Site types: {list(site_counts.keys())}")
        
        return valid_compositions
    
    def _count_sites_by_type(self) -> Dict[str, int]:
        """
        Count number of sites for each element type in the base structure.
        
        Returns:
            Dict mapping element symbol to count
        """
        site_counts = {}
        for site in self.base_structure:
            # Get the dominant species (for disordered sites, take max)
            if site.is_ordered:
                elem = str(site.specie)
            else:
                # For disordered sites, take the element with highest occupancy
                elem = max(site.species.keys(), key=lambda e: site.species[e])
                elem = str(elem)
            
            site_counts[elem] = site_counts.get(elem, 0) + 1
        
        return site_counts
    
    def _parse_range_spec(self, range_spec: Union[List[float], float]) -> List[float]:
        """
        Parse a range specification into a list of values.
        
        Args:
            range_spec: Either a single value, [min, max, step] range, or list of explicit values
            
        Returns:
            List of float values
        """
        if isinstance(range_spec, (int, float)):
            return [float(range_spec)]
        
        if len(range_spec) == 3:
            min_val, max_val, step = range_spec
            
            # Check if this is a valid [min, max, step] or an explicit 3-value list
            # If step > (max - min), it's likely an explicit list, not a range
            range_size = max_val - min_val
            
            if step > range_size and step > 0:
                # Step is larger than the range - treat as explicit list
                # e.g., [0.0, 0.5, 1.0] where 1.0 > (0.5 - 0.0)
                return [float(v) for v in range_spec]
            else:
                # Valid [min, max, step] range
                values = []
                current = min_val
                while current <= max_val + 1e-9:
                    values.append(round(current, 6))
                    current += step
                return values
        else:
            # Interpret as explicit list of values
            return [float(v) for v in range_spec]
    
    def _fractions_to_discrete(self, fractions: Dict[str, float], N: int) -> Optional[Dict[str, int]]:
        """
        Convert fractional composition to discrete atom counts.
        
        Uses rounding with remainder distribution to ensure exact sum to N.
        
        Args:
            fractions: Dictionary mapping element to fraction
            N: Total number of atoms
            
        Returns:
            Dictionary mapping element to atom count, or None if invalid
        """
        # Initial rounding
        discrete = {}
        remainders = {}
        
        for elem, frac in fractions.items():
            exact = frac * N
            discrete[elem] = int(round(exact))
            remainders[elem] = abs(exact - discrete[elem])
        
        # Adjust to ensure exact sum
        current_sum = sum(discrete.values())
        diff = N - current_sum
        
        # Distribute difference by largest remainders
        if diff != 0:
            sorted_elems = sorted(remainders.keys(), key=lambda e: remainders[e], reverse=True)
            for i in range(abs(diff)):
                elem = sorted_elems[i % len(sorted_elems)]
                if diff > 0:
                    discrete[elem] += 1
                else:
                    if discrete[elem] > 0:
                        discrete[elem] -= 1
        
        # Validate all counts are non-negative
        if any(count < 0 for count in discrete.values()):
            return None
        
        return discrete
    
    def generate_batched_tensors(self, compositions: List[Union[Dict[str, int], Dict[str, Dict[str, int]]]]) -> Dict[str, np.ndarray]:
        """
        Generate batched Z-tensors for all compositions.
        
        Handles both Global Shuffling (HEA) and Sublattice-Restricted Shuffling (Perovskites).
        
        This method:
        1. For each discrete composition, creates a 1D array with atomic numbers (Z)
        2. Duplicates each array samples_per_composition times
        3. Randomly shuffles each duplicated array (globally or per sublattice)
        4. Stacks all Z arrays into a single batched tensor
        
        Args:
            compositions: List of composition dictionaries (flat for HEA, nested for Perovskite)
            
        Returns:
            Dictionary with keys:
                - 'atomic_numbers': Shape [B, N] where B is total batches
                - 'positions': Shape [B, N, 3] broadcasted from frozen geometry
                - 'cell': Shape [B, 3, 3] broadcasted from frozen cell
                - 'metadata': List of dicts with composition info for each batch
        """
        samples_per_comp = self.config.get('samples_per_composition', 1)
        
        all_z_arrays = []
        metadata = []
        
        for comp_idx, composition in enumerate(compositions):
            # Check if this is a nested dict (Sublattice Mode) or flat dict (Global Mode)
            is_nested = any(isinstance(v, dict) for v in composition.values())
            
            for sample_idx in range(samples_per_comp):
                if is_nested:
                    # --- MODE A: SUBLATTICE-RESTRICTED SHUFFLING ---
                    z_array = np.zeros(self.num_atoms, dtype=np.int32)
                    flat_comp = {}
                    
                    for host_elem, sub_comp in composition.items():
                        # Find indices in the supercell that belong to this host sublattice
                        site_indices = np.where(self.host_elements == host_elem)[0]
                        
                        # Generate Z array purely for this sublattice
                        sub_z_list = []
                        for elem, count in sub_comp.items():
                            Z = self._get_atomic_number(elem)
                            sub_z_list.extend([Z] * count)
                            flat_comp[elem] = flat_comp.get(elem, 0) + count
                            
                        sub_z_array = np.array(sub_z_list, dtype=np.int32)
                        
                        # Random shuffle WITHIN the sublattice bounds only
                        np.random.shuffle(sub_z_array)
                        
                        # Assign back into the full supercell tensor
                        z_array[site_indices] = sub_z_array
                        
                    meta_comp = flat_comp
                else:
                    # --- MODE B: GLOBAL SHUFFLING (HEA Mode) ---
                    z_array = self._composition_to_z_array(composition)
                    np.random.shuffle(z_array)
                    meta_comp = composition.copy()
                
                all_z_arrays.append(z_array)
                
                # Store metadata
                meta = {
                    'composition_idx': comp_idx,
                    'sample_idx': sample_idx,
                    'composition': meta_comp,
                    'total_atoms': self.num_atoms
                }
                metadata.append(meta)
        
        # Stack into batched tensors
        atomic_numbers = np.stack(all_z_arrays, axis=0)  # [B, N]
        
        # Broadcast positions and cell
        B = len(all_z_arrays)
        positions = np.broadcast_to(
            self.pos_tensor[np.newaxis, :, :], 
            (B, self.num_atoms, 3)
        )  # [B, N, 3]
        
        cell = np.broadcast_to(
            self.cell_tensor[np.newaxis, :, :],
            (B, 3, 3)
        )  # [B, 3, 3]
        
        print(f"[TensorGen] Generated {B} batched configurations")
        
        return {
            'atomic_numbers': atomic_numbers,
            'positions': positions,
            'cell': cell,
            'metadata': metadata
        }
    
    def _composition_to_z_array(self, composition: Dict[str, int]) -> np.ndarray:
        """
        Convert a composition dictionary to a 1D Z array.
        
        Args:
            composition: Dictionary mapping element symbol to atom count
            
        Returns:
            1D numpy array of atomic numbers
        """
        z_list = []
        
        for elem, count in composition.items():
            Z = self._get_atomic_number(elem)
            z_list.extend([Z] * count)
        
        return np.array(z_list, dtype=np.int32)
    
    def _get_atomic_number(self, symbol: str) -> int:
        """Get atomic number from element symbol with caching."""
        if symbol not in self._element_to_Z:
            try:
                elem = Element(symbol)
                self._element_to_Z[symbol] = elem.Z
            except:
                raise ValueError(f"Unknown element symbol: {symbol}")
        
        return self._element_to_Z[symbol]
    
    def generate(self) -> Dict[str, np.ndarray]:
        """
        Main entry point for the fast tensor generation pipeline.
        
        This method orchestrates the full pipeline:
        1. Parse and freeze geometry
        2. Compute discrete compositions
        3. Generate batched Z-tensors
        
        Returns:
            Dictionary with batched tensors ready for MLIP evaluation:
                - 'atomic_numbers': Shape [B, N]
                - 'positions': Shape [B, N, 3]
                - 'cell': Shape [B, 3, 3]
                - 'metadata': List of composition metadata
        """
        print("\n" + "=" * 80)
        print("TENSOR GENERATION PIPELINE (Fast Mode)")
        print("=" * 80)
        
        # Step 1: Parse and freeze geometry
        self.parse_and_freeze_geometry()
        
        # Step 2: Compute discrete compositions
        compositions = self.compute_discrete_compositions()
        
        if not compositions:
            print("[TensorGen] Warning: No valid compositions generated!")
            return {
                'atomic_numbers': np.array([]),
                'positions': np.array([]),
                'cell': np.array([]),
                'metadata': []
            }
        
        # Step 3: Generate batched tensors
        result = self.generate_batched_tensors(compositions)
        
        print("=" * 80)
        print(f"[TensorGen] Pipeline complete: {len(result['metadata'])} configurations")
        print("=" * 80 + "\n")
        
        return result
    
    def export_to_cif(self, output_dir: str, batch_idx: Optional[int] = None) -> None:
        """
        Export generated structures to CIF files.
        
        Note: This requires rehydrating the tensors back to pymatgen Structures,
        so it defeats the speed benefit. Use only for verification/debugging.
        
        Args:
            output_dir: Directory to save CIF files
            batch_idx: If specified, only export this batch index
        """
        if not HAVE_ASE:
            print("[TensorGen] Warning: ASE not available, cannot export to CIF")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Need to generate first if not already done
        if self.pos_tensor is None:
            result = self.generate()
        else:
            # Regenerate tensors
            compositions = self.compute_discrete_compositions()
            result = self.generate_batched_tensors(compositions)
        
        atomic_numbers = result['atomic_numbers']
        positions = result['positions']
        cell = result['cell']
        metadata = result['metadata']
        
        indices = [batch_idx] if batch_idx is not None else range(len(metadata))
        
        for i in indices:
            z_array = atomic_numbers[i]
            pos_array = positions[i]
            cell_array = cell[i]
            meta = metadata[i]
            
            # Convert to ASE Atoms
            symbols = [Element.from_Z(int(Z)).symbol for Z in z_array]
            atoms = Atoms(
                symbols=symbols,
                positions=pos_array,
                cell=cell_array,
                pbc=True
            )
            
            # Build filename
            comp_str = '_'.join([f"{elem}{count}" for elem, count in sorted(meta['composition'].items())])
            filename = f"batch_{i:04d}_comp{meta['composition_idx']}_sample{meta['sample_idx']}_{comp_str}.cif"
            filepath = os.path.join(output_dir, filename)
            
            write(filepath, atoms)
        
        print(f"[TensorGen] Exported {len(indices)} structures to {output_dir}")
    
    def to_ase_atoms_list(self) -> List[Atoms]:
        """
        Convert generated tensors to a list of ASE Atoms objects.
        
        Returns:
            List of ASE Atoms objects
        """
        if not HAVE_ASE:
            raise ImportError("ASE is required for this operation")
        
        # Generate if not done
        if self.pos_tensor is None:
            result = self.generate()
        else:
            compositions = self.compute_discrete_compositions()
            result = self.generate_batched_tensors(compositions)
        
        atomic_numbers = result['atomic_numbers']
        positions = result['positions']
        cell = result['cell']
        
        atoms_list = []
        
        for i in range(len(atomic_numbers)):
            z_array = atomic_numbers[i]
            pos_array = positions[i]
            cell_array = cell[i]
            
            symbols = [Element.from_Z(int(Z)).symbol for Z in z_array]
            atoms = Atoms(
                symbols=symbols,
                positions=pos_array,
                cell=cell_array,
                pbc=True
            )
            atoms_list.append(atoms)
        
        return atoms_list


def run_fast_pipeline(config_path: str, output_dir: str, save_cif: bool = False) -> Dict[str, np.ndarray]:
    """
    Convenience function to run the full fast tensor pipeline.
    
    Args:
        config_path: Path to JSON configuration file
        output_dir: Directory for output files
        save_cif: If True, also export to CIF files (slower)
        
    Returns:
        Dictionary with batched tensors
    """
    generator = TensorGenerator(config_path)
    result = generator.generate()
    
    if save_cif:
        generator.export_to_cif(output_dir)
    
    return result


def convert_npz_to_cif(npz_path: str, output_dir: str, metadata_path: Optional[str] = None,
                       indices: Optional[List[int]] = None, max_structures: Optional[int] = None,
                       polyatomic_mapping: Optional[Dict[str, str]] = None,
                       custom_xyz_dir: Optional[str] = None) -> None:
    """
    Convert NPZ tensor data to CIF files.
    
    This standalone utility converts the batched tensor output back to individual
    CIF files that can be viewed in standard crystallography software.
    
    Supports polyatomic ion expansion via dummy atom substitution.
    
    Args:
        npz_path: Path to the .npz file containing atomic_numbers, positions, cell
        output_dir: Directory to save CIF files
        metadata_path: Optional path to metadata JSON for naming
        indices: Optional list of specific batch indices to export (None = all)
        max_structures: Maximum number of structures to export (for quick inspection)
        polyatomic_mapping: Optional dict mapping dummy elements to molecules (e.g., {"Fr": "NH4", "At": "SCN"})
        custom_xyz_dir: Optional directory with custom XYZ molecule files
    """
    if not HAVE_ASE:
        raise ImportError("ASE is required for CIF export")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NPZ data
    print(f"[NPZtoCIF] Loading tensor data from {npz_path}...")
    data = np.load(npz_path)
    atomic_numbers = data['atomic_numbers']
    positions = data['positions']
    cell = data['cell']
    
    B = atomic_numbers.shape[0]
    print(f"[NPZtoCIF] Found {B} configurations")
    
    # Load metadata if provided
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"[NPZtoCIF] Loaded metadata for naming")
    
    # Determine which indices to export
    if indices is not None:
        export_indices = [i for i in indices if 0 <= i < B]
    elif max_structures is not None:
        export_indices = list(range(min(max_structures, B)))
    else:
        export_indices = list(range(B))
    
    print(f"[NPZtoCIF] Exporting {len(export_indices)} structures to {output_dir}...")
    
    # Import polyatomic utils if needed
    if polyatomic_mapping:
        try:
            from .polyatomic_utils import expand_polyatomics_in_structure
        except ImportError:
            from polyatomic_utils import expand_polyatomics_in_structure
        print(f"[NPZtoCIF] Polyatomic expansion enabled: {polyatomic_mapping}")
    
    # Export each structure
    for idx in export_indices:
        z_array = atomic_numbers[idx]
        pos_array = positions[idx]
        cell_array = cell[idx]
        
        # Convert Z to element symbols
        symbols = [Element.from_Z(int(Z)).symbol for Z in z_array]
        
        # Create ASE Atoms
        atoms = Atoms(
            symbols=symbols,
            positions=pos_array,
            cell=cell_array,
            pbc=True
        )
        
        # Apply polyatomic expansion if requested
        if polyatomic_mapping:
            from pymatgen.io.ase import AseAtomsAdaptor
            # Convert ASE to pymatgen
            pymg_structure = AseAtomsAdaptor.get_structure(atoms)
            # Expand polyatomics
            expanded = expand_polyatomics_in_structure(
                pymg_structure, polyatomic_mapping, custom_xyz_dir, random_rotation=True
            )
            # Convert back to ASE
            atoms = AseAtomsAdaptor.get_atoms(expanded)
        
        # Build filename
        if metadata and idx < len(metadata):
            meta = metadata[idx]
            comp_str = '_'.join([f"{elem}{count}" for elem, count in sorted(meta['composition'].items())])
            filename = f"config_{idx:04d}_comp{meta['composition_idx']}_sample{meta['sample_idx']}_{comp_str}.cif"
        else:
            # Generate filename from composition
            unique, counts = np.unique(z_array, return_counts=True)
            comp_str = '_'.join([f"{Element.from_Z(int(Z)).symbol}{count}" for Z, count in zip(unique, counts)])
            filename = f"config_{idx:04d}_{comp_str}.cif"
        
        filepath = os.path.join(output_dir, filename)
        write(filepath, atoms)
    
    print(f"[NPZtoCIF] Successfully exported {len(export_indices)} CIF files to {output_dir}")


def tensors_to_pymatgen(atomic_numbers: np.ndarray, positions: np.ndarray, 
                        cell: np.ndarray, index: int = 0) -> Structure:
    """
    Convert a single configuration from tensors to pymatgen Structure.
    
    Args:
        atomic_numbers: Array of shape [B, N] or [N]
        positions: Array of shape [B, N, 3] or [N, 3]
        cell: Array of shape [B, 3, 3] or [3, 3]
        index: Which batch index to convert (if batched)
        
    Returns:
        pymatgen Structure object
    """
    # Handle batched input
    if atomic_numbers.ndim == 2:
        z_array = atomic_numbers[index]
        pos_array = positions[index]
        cell_array = cell[index]
    else:
        z_array = atomic_numbers
        pos_array = positions
        cell_array = cell
    
    # Convert to element symbols
    species = [Element.from_Z(int(Z)) for Z in z_array]
    
    # Create structure
    structure = Structure(
        lattice=cell_array,
        species=species,
        coords=pos_array,
        coords_are_cartesian=True
    )
    
    return structure


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TensorGenerator: Fast tensor pipeline and NPZ utilities')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate tensors from JSON config')
    gen_parser.add_argument('config', help='Path to JSON configuration file')
    gen_parser.add_argument('--output-dir', default='./tensor_output', help='Output directory')
    gen_parser.add_argument('--save-cif', action='store_true', help='Also export to CIF files')
    
    # Convert command
    conv_parser = subparsers.add_parser('convert', help='Convert NPZ to CIF files')
    conv_parser.add_argument('npz', help='Path to .npz tensor file')
    conv_parser.add_argument('--output-dir', required=True, help='Directory for CIF output')
    conv_parser.add_argument('--metadata', help='Optional metadata JSON for naming')
    conv_parser.add_argument('--max-structures', type=int, help='Export only first N structures')
    conv_parser.add_argument('--indices', type=int, nargs='+', help='Specific indices to export')
    conv_parser.add_argument('--polyatomic', help='Polyatomic mapping (e.g., "Fr:NH4,At:SCN")')
    conv_parser.add_argument('--custom-xyz-dir', help='Directory with custom XYZ molecule files')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        result = run_fast_pipeline(args.config, args.output_dir, args.save_cif)
        print(f"\nGenerated {result['atomic_numbers'].shape[0]} configurations")
        print(f"Atoms per configuration: {result['atomic_numbers'].shape[1]}")
        
    elif args.command == 'convert':
        # Parse polyatomic mapping if provided
        polyatomic_mapping = None
        if args.polyatomic:
            polyatomic_mapping = {}
            for pair in args.polyatomic.split(','):
                if ':' in pair:
                    dummy, mol = pair.split(':')
                    polyatomic_mapping[dummy.strip()] = mol.strip()
                else:
                    print(f"[Warning] Invalid polyatomic mapping format: {pair}")
        
        convert_npz_to_cif(
            args.npz,
            args.output_dir,
            metadata_path=args.metadata,
            indices=args.indices,
            max_structures=args.max_structures,
            polyatomic_mapping=polyatomic_mapping,
            custom_xyz_dir=args.custom_xyz_dir
        )
    else:
        parser.print_help()
