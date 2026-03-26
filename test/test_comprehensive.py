#!/usr/bin/env python3
"""
Comprehensive Diagnostic Test Suite for ProtoCSP

Tests the tool with:
1. Simple baseline structures (sanity check)
2. Simple alloys, carbides, and nitrides (FeNi, TiC, etc.)
3. Fractional perovskite compositions
4. Complex steel alloy compositions from CSV

Features:
- Verbose failure analysis (checks if prototypes exist in the library).
- Detailed timing and candidate counts.
- Progression from simple to complex structures.
"""

import os
import sys
import pandas as pd
import subprocess
import time
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.append('.')

# Try importing pymatgen for the analysis logic
try:
    from pymatgen.core import Composition
except ImportError:
    print("Error: Pymatgen not installed.")
    sys.exit(1)

from lemat_unique_dataset.ProtoCSP.core import ProtoCSP


def analyze_failure(library_source, comp_str):
    """
    Diagnoses why no candidates were generated for a specific composition.
    Returns a verbose string explanation.
    """
    try:
        target_comp = Composition(comp_str)
        
        # Helper to check existence in Dict or Folder
        def check_exists(anon_form):
            if isinstance(library_source, dict):
                return anon_form in library_source
            elif isinstance(library_source, str) and os.path.isdir(library_source):
                clean_name = re.sub(r'[\\/*?:"<>|]', "", anon_form)
                return os.path.exists(os.path.join(library_source, f"{clean_name}.json"))
            return False

        # 1. Check Exact Match / Anonymized Formula
        anon_formula = target_comp.anonymized_formula
        if check_exists(anon_formula):
            return (f"Prototype '{anon_formula}' found in library, "
                    f"but element substitution failed (likely due to electronegativity mismatch).")

        # 2. Check Recursive Reduction Path (The 'Base Structure' logic)
        # We simulate the logic used in ProtoCSP._find_base_structure
        elements_sorted = sorted(target_comp.elements, key=lambda e: target_comp.get_atomic_fraction(e))
        amounts = {e: target_comp.get(e) for e in target_comp.elements}
        
        reduction_trace = []
        found_base = False
        found_proto = None

        # Iterate removing elements to see if we hit a valid prototype
        for _ in range(len(elements_sorted) + 1):
            if not amounts: break
            
            # Create dummy composition to check formula class (e.g. "A", "AB")
            dummy_comp = Composition({e: 1.0 for e in amounts.keys()})
            curr_anon = dummy_comp.anonymized_formula
            reduction_trace.append(curr_anon)
            
            if check_exists(curr_anon):
                found_base = True
                found_proto = curr_anon
                break 

            # Remove smallest
            remaining = [e for e in elements_sorted if e in amounts]
            if not remaining: break
            del amounts[remaining[0]]

        if found_base:
            return (f"Logic found base prototype class '{found_proto}' after reduction path {reduction_trace}. "
                    f"Failure likely occurred during Supercell Generation or Doping (check element counts or radius scaling).")
        else:
            return (f"NO suitable prototype found. \n"
                    f"       Checked reduction path: { ' -> '.join(reduction_trace) } \n"
                    f"       The formula database '{library_source}' does not contain these stoichiometries.")
            
    except Exception as e:
        return f"Analysis crashed: {e}"

def test_simple_structures():
    """Test simple, well-known structures as a sanity check."""
    print("=" * 80)
    print("TESTING SIMPLE BASELINE STRUCTURES")
    print("=" * 80)

    test_compositions = [
        "NaCl", "Fe", "SrTiO3", "MgO", "GaAs"
    ]

    library_path = "lemat_formula_indexed"
    if not os.path.exists(library_path):
        print(f"Error: Database folder '{library_path}' not found.")
        return []

    print(f"Using library database: {library_path}/")
    pcsp = ProtoCSP(library_path)
    print("-" * 80)

    results = []

    for comp in test_compositions:
        print(f"Testing: {comp:<30}", end="", flush=True)
        try:
            start_time = time.time()
            candidates = pcsp.generate(comp, top_k=3)
            elapsed = time.time() - start_time
            
            success = len(candidates) > 0
            status = "SUCCESS" if success else "FAILED"
            
            print(f"[{status}] {len(candidates)} candidates ({elapsed:.2f}s)")

            reason = "N/A"
            if not success:
                reason = analyze_failure(library_path, comp)
                print(f"  >>> REASON: {reason}")
            
            # --- UPDATED PRINTING LOGIC FOR DICTS ---
            for i, entry in enumerate(candidates):
                struct = entry['structure']
                e_val = entry.get('energy_per_atom')
                
                # Format energy string
                e_str = f"{e_val:.3f} eV/atom" if e_val is not None else "N/A (Generated)"
                src_id = entry.get('id') or entry.get('parent_id', 'Unknown')
                
                print(f"      [{i+1}] {struct.composition.reduced_formula}")
                print(f"          Source: {src_id}")
                print(f"          Energy: {e_str}")
                print(f"          Volume: {struct.volume:.1f} A^3")
                print(f"          SpaceG: {struct.get_space_group_info()[0]}")

            results.append({
                'composition': comp, 'success': success, 
                'candidates': len(candidates), 'time': elapsed, 'reason': reason
            })

        except Exception as e:
            print(f"[ERROR] {e}")
            results.append({'composition': comp, 'success': False, 'candidates': 0, 'time': 0, 'reason': str(e)})

    return results

def test_simple_alloys():
    """Test simple metal alloys, carbides, and nitrides."""
    print("\n" + "=" * 80)
    print("TESTING SIMPLE ALLOYS, CARBIDES, AND NITRIDES")
    print("=" * 80)

    test_compositions = [
        "FeNi", "CuZn", "TiC", "NbN", "WC"
    ]

    library_path = "lemat_formula_indexed"
    if not os.path.exists(library_path):
        print(f"Error: Database folder '{library_path}' not found.")
        return []

    print(f"Using library database: {library_path}/")
    pcsp = ProtoCSP(library_path)
    print("-" * 80)

    results = []

    for comp in test_compositions:
        print(f"Testing: {comp:<30}", end="", flush=True)
        try:
            start_time = time.time()
            candidates = pcsp.generate(comp, top_k=3)
            elapsed = time.time() - start_time

            success = len(candidates) > 0
            status = "SUCCESS" if success else "FAILED"

            print(f"[{status}] {len(candidates)} candidates ({elapsed:.2f}s)")

            reason = "N/A"
            if not success:
                reason = analyze_failure(library_path, comp)
                print(f"  >>> REASON: {reason}")

            # Print candidate details
            for i, entry in enumerate(candidates):
                struct = entry['structure']

                # Extract Metadata
                src_id = entry.get('id')
                parent_form = entry.get('parent_formula')
                method = entry.get('method', 'Unknown')
                e_val = entry.get('energy_per_atom')
                e_str = f"{e_val:.3f} eV/atom" if e_val is not None else "N/A"

                # Construct Source String
                source_str = src_id
                if parent_form:
                    source_str += f" (Base: {parent_form})"

                print(f"      [{i+1}] {struct.composition.reduced_formula}")
                print(f"          Source: {source_str}")
                if method != 'Unknown':
                    print(f"          Method: {method}")
                print(f"          Energy: {e_str}")
                print(f"          Volume: {struct.volume:.1f} A^3")

            results.append({
                'composition': comp, 'success': success,
                'candidates': len(candidates), 'time': elapsed, 'reason': reason
            })

        except Exception as e:
            print(f"[ERROR] {e}")
            results.append({'composition': comp, 'success': False, 'candidates': 0, 'time': 0, 'reason': str(e)})

    return results

def test_fractional_perovskites():
    """Test fractional perovskite compositions."""
    print("\n" + "=" * 80)
    print("TESTING FRACTIONAL PEROVSKITE COMPOSITIONS")
    print("=" * 80)

    test_compositions = [
        "La0.5Sr0.5MnO3", "LaMn0.75Fe0.25O3",
        "La0.5Ca0.5Mn0.8Fe0.2O3", "SrTi0.5Zr0.5O3", "Ba0.5Sr0.5TiO3"
    ]

    library_path = "lemat_formula_indexed"
    pcsp = ProtoCSP(library_path)
    print("-" * 80)

    results = []

    for comp in test_compositions:
        print(f"Testing: {comp:<30}", end="", flush=True)
        try:
            start_time = time.time()
            candidates = pcsp.generate(comp, top_k=3)
            elapsed = time.time() - start_time

            success = len(candidates) > 0
            status = "SUCCESS" if success else "FAILED"
            
            print(f"[{status}] {len(candidates)} candidates ({elapsed:.2f}s)")

            reason = "N/A"
            if not success:
                reason = analyze_failure(library_path, comp)
                print(f"  >>> REASON: {reason}")

            # --- UPDATED PRINTING LOGIC FOR DICTS ---
            for i, entry in enumerate(candidates):
                struct = entry['structure']
                
                # Extract Metadata
                src_id = entry.get('id')
                parent_form = entry.get('parent_formula')
                method = entry.get('method', 'Unknown')
                e_val = entry.get('energy_per_atom')
                e_str = f"{e_val:.3f} eV/atom" if e_val is not None else "N/A"
                
                # Construct Source String
                source_str = src_id
                if parent_form:
                    source_str += f" (Base: {parent_form})"
                
                print(f"      [{i+1}] {struct.composition.reduced_formula}")
                print(f"          Source: {source_str}")
                print(f"          Method: {method}")
                print(f"          Energy: {e_str}")
                print(f"          Volume: {struct.volume:.1f} A^3")

            results.append({
                'composition': comp, 'success': success, 
                'candidates': len(candidates), 'time': elapsed, 'reason': reason
            })

        except Exception as e:
            print(f"[ERROR] {e}")
            results.append({'composition': comp, 'success': False, 'candidates': 0, 'time': 0, 'reason': str(e)})

    return results
def test_steel_compositions():
    """Test steel compositions from CSV."""
    print("\n" + "=" * 80)
    print("TESTING STEEL COMPOSITIONS FROM CSV")
    print("=" * 80)

    csv_path = "test/matbench_steels_featurizedMM2020Comp.csv"

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} compositions from CSV")
        # Select first 5 compositions to save time
        test_comps = df.head(5)['composition'].tolist()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        test_comps = [
            "Fe0.8C0.1Mn0.05Cr0.03Ni0.02",
            "Fe0.75C0.15Si0.05Mn0.03Cr0.02"
        ]

    results = []

    for comp in test_comps:
        print(f"Testing: {comp:<30}", end="", flush=True)
        try:
            # We must use subprocess here to test CLI argument passing.
            # Updated to use the indexed folder instead of rebuilding.
            
            cmd = [
                sys.executable, "main.py", comp,
                "--index", "lemat_formula_indexed", # Use the folder database
                "--verbose",
                "--output-dir", "test_outputs",
                "--top-k", "3"
            ]

            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            elapsed = time.time() - start_time

            candidates = 0
            # Parse output for candidate count
            # Look for "GENERATION COMPLETE (N candidates in X.XXs)"
            for line in result.stdout.split('\n'):
                if "GENERATION COMPLETE" in line and "candidates" in line:
                    try:
                        # Extract number from "(N candidates in X.XXs)"
                        match = line.split('(')[1].split('candidates')[0].strip()
                        candidates = int(match)
                    except:
                        pass

            success = (result.returncode == 0) and (candidates > 0)
            status = "SUCCESS" if success else "FAILED"
            
            print(f"[{status}] {candidates} candidates ({elapsed:.2f}s)")

            reason = "N/A"
            if not success:
                if result.returncode != 0:
                    reason = f"CLI Process Crashed. Stderr: {result.stderr.strip()[:100]}..."
                elif candidates == 0:
                    # Try to find a DEBUG line in stdout
                    debug_lines = [l for l in result.stdout.split('\n') if "DEBUG" in l]
                    if debug_lines:
                        reason = f"Log: {debug_lines[-1].strip()}"
                    else:
                        reason = "No candidates found."
                
                print(f"  >>> REASON: {reason}")

            results.append({
                'composition': comp,
                'success': success,
                'candidates': candidates,
                'time': elapsed,
                'reason': reason
            })

        except Exception as e:
            print(f"[ERROR] {e}")
            results.append({
                'composition': comp,
                'success': False,
                'candidates': 0,
                'time': 0,
                'reason': f"Exception: {str(e)}"
            })

    return results


def main():
    print("=" * 80)
    print("CRYSTALGUESSER COMPREHENSIVE DIAGNOSTIC TEST SUITE")
    print("=" * 80)
    print("Testing increasingly complex compositions with failure analysis...")

    # Test 1: Simple Baseline
    simple_results = test_simple_structures()

    # Test 2: Simple Alloys, Carbides, Nitrides
    alloy_results = test_simple_alloys()

    # Test 3: Fractional perovskites
    perovskite_results = test_fractional_perovskites()

    # Test 4: Steel compositions
    steel_results = test_steel_compositions()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_results = simple_results + alloy_results + perovskite_results + steel_results

    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r['success'])
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")

    # Analysis and suggestions
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)

    failures = [r for r in all_results if not r['success']]
    if failures:
        print(f"Failures ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure['composition']}")
            print(f"    Reason: {failure['reason']}")
            print("-" * 40)
    else:
        print("All tests passed!")

    # Write detailed results to file
    output_file = "test_results_detailed.txt"
    with open(output_file, 'w') as file:
        file.write("CRYSTALGUESSER COMPREHENSIVE TEST RESULTS\n")
        file.write("=" * 80 + "\n\n")

        for r in all_results:
            status = "SUCCESS" if r['success'] else "FAILED"
            file.write(f"{r['composition']:<30} | {status} | {r['reason']}\n")

    print(f"\nDetailed results written to: {output_file}")


if __name__ == "__main__":
    main()