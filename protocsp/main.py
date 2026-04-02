#!/usr/bin/env python3
"""
ProtoCSP Main Script

Usage:
    python main.py [options] <composition>

Options:
    --csv-dir DIR       Directory containing LeMat CSV parts (default: ../lemat_unique_csv_500_parts)
    --output-dir DIR    Directory to save generated structures (default: ./generated_structures)
    --limit LIMIT       Limit number of structures to load for testing (only used if rebuilding index)
    --top-k K           Number of top candidates to generate (default: 5)
    --save-cif          Save generated structures as CIF files
    --verbose           Print detailed progress information
    --index PATH        Path to library folder (JSONs) OR pickle file (default: lemat_formula_indexed)
    --force-rebuild     Force rebuilding a pickle index from CSVs (ignored if using folder mode)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List
import pickle
import json
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from pymatgen.core import Structure
except ImportError:
    print("Error: 'pymatgen' library not found. Please run 'pip install pymatgen'")
    sys.exit(1)

from core import ProtoCSP, build_library_index


def save_structures_as_cif(candidates: List[dict], composition: str, output_dir: str, suffix: str = ""):
    """
    Saves generated structures as CIF files with meaningful filenames.

    Args:
        candidates: List of candidate dictionaries with 'structure', 'id', 'method', etc.
        composition: Chemical formula
        output_dir: Output directory
        suffix: Optional string to append to filename (e.g., '_unrelaxed', '_relaxed')
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving {len(candidates)} structures to {output_dir}/ (Suffix: '{suffix}')")

    for i, entry in enumerate(candidates):
        try:
            struct = entry['structure']

            # Clean composition string (remove spaces, special chars)
            clean_comp = composition.replace(' ', '').replace('(', '').replace(')', '')

            # Get method and source ID
            method = entry.get('method', 'unknown')
            source_id = entry.get('id', 'unknown')

            # Simplify method name for filename
            method_short = method.split('(')[0].strip().replace(' ', '_').lower()
            clean_source = source_id.replace('/', '_').replace(':', '_')

            # Build filename: composition_method_source_rank_N_suffix.cif
            filename = f"{clean_comp}_{method_short}_{clean_source}_rank_{i+1}{suffix}.cif"
            filepath = os.path.join(output_dir, filename)

            struct.to(filename=filepath, fmt="cif")
            print(f"  [{i+1}] Saved: {filename}")
        except Exception as e:
            print(f"  [ERROR] Failed to save structure {i+1}: {e}")

def compute_reference_energies(comp_str: str, generator: ProtoCSP, calc, engine: str, fmax: float, output_dir: str, do_relax: bool) -> dict:
    """
    Finds the lowest energy pure element phases from the database, evaluates/relaxes 
    them with the MLIP, and returns their energy per atom (chemical potential).
    """
    from pymatgen.core import Composition
    from pymatgen.io.ase import AseAtomsAdaptor
    from mlip_utils import relax_structure

    comp = Composition(comp_str)
    elements = [e.symbol for e in comp.elements]
    
    adapter = AseAtomsAdaptor()
    mu_dict = {}
    
    entries = generator._get_entries("A")
    
    print("\n" + "-"*80)
    print(f"[INFO] Computing MLIP Reference Energies for Formation Energy")
    print("-" * 80)

    for el in elements:
        print(f"[INFO] Finding reference state for {el}...")
        el_entries = [e for e in entries if e.get('reduced_formula') == el]
        
        if not el_entries:
            print(f"  [WARNING] No pure phase found for {el} in database. Using 0.0 eV/atom.")
            mu_dict[el] = 0.0
            continue
            
        # Sort by database energy (lowest first) to get the ground state polymorph
        el_entries = [e for e in el_entries if e.get('energy_per_atom') is not None]
        if not el_entries:
            print(f"  [WARNING] No pure phase with valid energy found for {el}. Using 0.0 eV/atom.")
            mu_dict[el] = 0.0
            continue

        el_entries.sort(key=lambda x: x.get('energy_per_atom'))
        
        best_entry = generator._hydrate_entry(el_entries[0])
        if not best_entry:
            print(f"  [WARNING] Could not hydrate reference for {el}. Using 0.0 eV/atom.")
            mu_dict[el] = 0.0
            continue

        atoms = adapter.get_atoms(best_entry['structure'])
        atoms.calc = calc
        
        if do_relax:
            # Create a dedicated folder so logs don't overwrite
            ref_dir = os.path.join(output_dir, f"ref_{el}")
            print(f"  [{engine.upper()}] Relaxing reference {el}...")
            relaxed_atoms = relax_structure(atoms, calc, engine, fmax, ref_dir)
            if relaxed_atoms is not None:
                atoms = relaxed_atoms
                atoms.calc = calc # Ensure calculator is preserved
                
        try:
            energy = atoms.get_potential_energy()
            mu = energy / len(atoms)
            mu_dict[el] = mu
            print(f"  [INFO] Reference energy (\u03bc) for {el}: {mu:.4f} eV/atom")
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate reference {el}: {e}")
            mu_dict[el] = 0.0
        
    return mu_dict

def evaluate_candidates_with_mlip(candidates: List[dict], calc, engine: str, do_relax: bool, fmax: float, output_dir: str, mu_dict: dict) -> List[dict]:
    """
    Evaluates, optionally relaxes, and calculates the formation energy of each candidate.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from mlip_utils import relax_structure

    adapter = AseAtomsAdaptor()
    
    print(f"\n[INFO] Computing energies for {len(candidates)} candidates (Relaxation: {do_relax})...")
    for i, entry in enumerate(candidates):
        try:
            # Convert pymatgen Structure to ASE Atoms
            atoms = adapter.get_atoms(entry['structure'])
            atoms.calc = calc
            
            # --- RELAXATION STEP ---
            if do_relax:
                print(f"  [{engine.upper()}] Relaxes Candidate {i+1}...")
                # Print general information of the candidate
                
                cand_dir = os.path.join(output_dir, f"cand_{i+1}")
                relaxed_atoms = relax_structure(atoms, calc, engine, fmax, cand_dir)
                
                if relaxed_atoms is None:
                    print(f"  [WARNING] Relaxation failed or exploded for Candidate {i+1}. Skipping.")
                    entry['total_energy'] = None
                    entry['energy_per_atom'] = None
                    entry['formation_energy_per_atom'] = None
                    continue
                
                atoms = relaxed_atoms
                atoms.calc = calc
                entry['structure'] = adapter.get_structure(atoms)
            # -----------------------

            # Calculate total energy
            total_energy = atoms.get_potential_energy()
            energy_per_atom = total_energy / len(atoms)
            
            # Calculate Formation Energy
            comp = entry['structure'].composition
            frac_dict = {e.symbol: comp.get_atomic_fraction(e) for e in comp.elements}
            ref_energy = sum(frac_dict[el] * mu_dict.get(el, 0.0) for el in frac_dict)
            formation_energy = energy_per_atom - ref_energy
            
            # Update dictionary
            entry['total_energy'] = float(total_energy)
            entry['energy_per_atom'] = float(energy_per_atom)
            entry['formation_energy_per_atom'] = float(formation_energy)
            
            print(f"  [{engine.upper()}] Candidate {i+1} E_form: {formation_energy:.4f} eV/atom (E_tot: {energy_per_atom:.4f})")
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate Candidate {i+1}: {e}")
            entry['total_energy'] = None
            entry['energy_per_atom'] = None
            entry['formation_energy_per_atom'] = None
            
    return candidates

def main():
    parser = argparse.ArgumentParser(description='ProtoCSP')
    parser.add_argument('composition', help='Chemical composition')
    parser.add_argument('--csv-dir', default='../lemat_unique_csv_500_parts', help='CSV Directory')
    parser.add_argument('--output-dir', default='./generated_structures', help='Output Directory')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max-bases', type=int, default=3, help='Maximum number of different parent structures to use')
    parser.add_argument('--base-cif', type=str, default=None, help='Path to a manual CIF file to use as the structural base.')

    parser.add_argument('--min-atoms', type=int, default=20, help='Minimum atoms in supercell to ensure stoichiometric accuracy')
    parser.add_argument('--randomize-scaling', action='store_true', help='Slightly randomize supercell dimensions for more diversity')
    parser.add_argument('--symmetrize', action='store_true', help='Use rigorous enumeration to find high-symmetry ordered structures instead of farthest-point sampling')

    parser.add_argument('--sqs', action='store_true', help='Use SQS generation for solid solutions')
    parser.add_argument('--sqs-mode', type=str, default='random', choices=['random', 'systematic', 'cluster'], help='SQS generation mode (default: random)')
    parser.add_argument('--sqs-iterations', type=int, default=100000, help='Iterations for SQS (default: 100000)')

    parser.add_argument('--save-cif', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--index', default='lemat_formula_indexed', help='Library path')
    parser.add_argument('--force-rebuild', action='store_true')

    # --- MLIP ARGUMENTS ---
    parser.add_argument('--mlip', action='store_true', help='Activate MLIP energy evaluation for final candidates')
    parser.add_argument('--no-relax', action='store_true', help='Skip MLIP relaxation and only perform single-point energy evaluation')
    parser.add_argument('--fmax', type=float, default=0.01, help='Force tolerance for relaxation in eV/A (default: 0.05)')
    parser.add_argument('--engine', type=str, default='mace', choices=['mace', 'esen', 'uma', 'm3gnet', 'gpumd', 'nep', 'calorine'], help='MLIP engine to use (default: mace)')
    parser.add_argument('--model-name', type=str, default='medium-omat-0', help='Model name for MACE (default: medium-omat-0)')
    parser.add_argument('--checkpoint-model', type=str, default=None, help='Path to checkpoint file (Required for eSEN/UMA/NEP)')
    args = parser.parse_args()

    print("=" * 80)
    print("PROTOCSP: Prototype-based Crystal Structure Prediction")
    print("=" * 80)
    print(f"Target Composition: {args.composition}")
    print(f"Index Path: {args.index}")

    start_time = time.time()
    library_source = None

    if os.path.isdir(args.index):
        library_source = args.index
    elif os.path.isfile(args.index) and not args.force_rebuild:
        try:
            with open(args.index, 'rb') as f:
                library_source = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle: {e}"); sys.exit(1)
    else:
        print("Rebuilding index..."); library_source = build_library_index(args.csv_dir, limit=args.limit)
        if not args.index.endswith('/') and '.' in args.index:
            with open(args.index, 'wb') as f: pickle.dump(library_source, f)

    generator = ProtoCSP(library_source)

    manual_base_struct = None
    manual_base_id = None
    if args.base_cif:
        if not os.path.exists(args.base_cif):
            print(f"[ERROR] Base CIF file not found at {args.base_cif}")
            sys.exit(1)
        manual_base_struct = Structure.from_file(args.base_cif)
        manual_base_struct.remove_oxidation_states() 
        manual_base_id = os.path.basename(args.base_cif).replace('.cif', '')
        print(f"[INFO] Loaded manual base structure from {args.base_cif}")

    candidates = generator.generate(
        args.composition, 
        max_bases=args.max_bases,
        top_k=args.top_k, 
        min_atoms=args.min_atoms, 
        randomize_scaling=args.randomize_scaling,
        manual_base_struct=manual_base_struct,
        manual_base_id=manual_base_id,
        symmetrize=args.symmetrize,
        sqs=args.sqs,
        sqs_mode=args.sqs_mode,
        sqs_iterations=args.sqs_iterations
    )
    generation_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"GENERATION COMPLETE ({len(candidates)} candidates in {generation_time:.2f}s)")
    print()

    # --- SAVE UNRELAXED CIFs BEFORE MLIP OVERWRITES THEM ---
    if args.save_cif and candidates:
        suffix = "_unrelaxed" if (args.mlip and not args.no_relax) else ""
        save_structures_as_cif(candidates, args.composition, args.output_dir, suffix)

    # --- RUN MLIP EVALUATION ---
    if args.mlip and candidates:
        mlip_start = time.time()
        
        try:
            from mlip_utils import initialize_calculator
        except ImportError as e:
            print(f"\n[ERROR] Could not import MLIP utilities: {e}")
            sys.exit(1)

        print(f"\n[INFO] Initializing {args.engine.upper()} MLIP calculator...")
        calc = initialize_calculator(
            engine=args.engine, 
            model_name=args.model_name, 
            checkpoint_model_path=args.checkpoint_model
        )
        
        if calc is not None:
            # 1. Calculate Reference Energies (mu)
            mu_dict = compute_reference_energies(
                comp_str=args.composition,
                generator=generator,
                calc=calc,
                engine=args.engine,
                fmax=args.fmax,
                output_dir=args.output_dir,
                do_relax=not args.no_relax
            )
            
            # 2. Evaluate and Relax Candidates
            candidates = evaluate_candidates_with_mlip(
                candidates=candidates, 
                calc=calc,
                engine=args.engine, 
                do_relax=not args.no_relax,
                fmax=args.fmax,
                output_dir=args.output_dir,
                mu_dict=mu_dict
            )
            
            # 3. Re-sort candidates by Formation Energy! (Lowest/most negative is best)
            # Handle None values gracefully
            candidates.sort(key=lambda x: x.get('formation_energy_per_atom') if x.get('formation_energy_per_atom') is not None else 999.0)

            # --- SAVE RELAXED CIFs ---
            if args.save_cif and not args.no_relax:
                save_structures_as_cif(candidates, args.composition, args.output_dir, suffix="_relaxed")

        print(f"[INFO] MLIP Evaluation finished in {time.time() - mlip_start:.2f}s")
    # --------------------------------

    print()

    if candidates:
        print("Generated Structures (Ranked):")
        # --- OUTPUT LOOP ---
        for i, entry in enumerate(candidates):
            struct = entry['structure']
            e_tot = entry.get('energy_per_atom')
            e_form = entry.get('formation_energy_per_atom')
            
            e_str = f"{e_tot:.4f} eV/atom" if e_tot is not None else "N/A"
            form_str = f"{e_form:.4f} eV/atom" if e_form is not None else "N/A"
            
            src_id = entry.get('id') or entry.get('parent_id', 'Unknown')
            
            print(f"[{i+1}] Formula: {struct.composition.reduced_formula}")
            print(f"    Source ID: {src_id}")
            parent_sg = entry.get('parent_space_group', 'N/A')
            print(f"    Initial Space Group (Parent): {parent_sg}")
            
            if args.mlip:
                print(f"    Formation Energy: {form_str}")
                print(f"    Total Energy: {e_str}")
            
            try:
                # Store space group on the entry dict for JSON export later
                sg = struct.get_space_group_info()[0]
                entry['space_group'] = sg
            except:
                sg = "Unknown"
                entry['space_group'] = sg
            print(f"    Space Group: {sg}")
            print(f"    Volume: {struct.volume:.3f} Å³")
            print()

        # --- SAVE JSON FILE WITH ENERGIES ---
        if args.mlip:
            os.makedirs(args.output_dir, exist_ok=True)
            clean_comp = args.composition.replace(' ', '').replace('(', '').replace(')', '')
            json_path = os.path.join(args.output_dir, f"{clean_comp}_mlip_energies.json")
            
            json_data = []
            for i, entry in enumerate(candidates):
                json_data.append({
                    "rank": i + 1,
                    "formula": entry['structure'].composition.reduced_formula,
                    "source_id": entry.get('id') or entry.get('parent_id', 'Unknown'),
                    "parent_formula": entry.get('parent_formula', 'Unknown'),
                    "method": entry.get('method', 'Unknown'),
                    "space_group": entry.get('space_group', 'Unknown'),
                    "num_sites": entry['structure'].num_sites,
                    "volume_A3": round(entry['structure'].volume, 3),
                    "formation_energy_per_atom_eV": entry.get('formation_energy_per_atom'),
                    "total_energy_per_atom_eV": entry.get('energy_per_atom'),
                    "total_energy_eV": entry.get('total_energy')
                })
                
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            print(f"[INFO] MLIP evaluation summary saved to {json_path}")
    else:
        print("No valid candidates generated.")

if __name__ == "__main__":
    main()