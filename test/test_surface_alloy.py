#!/usr/bin/env python3
"""
Standalone Test: High-Entropy Alloy Surface Generation
(SQS + Vacuum + MLIP)

This script generates a BCC Fe (001) surface slab with a 15 Å vacuum gap, 
saves it as a base CIF, and feeds it into ProtoCSP to be alloyed into a 
complex HEA (Fe-Ni-Co-Mo-Pt) using Special Quasirandom Structures (SQS). 
Finally, the resulting surfaces are relaxed using the MACE MLIP.
"""

import os
import sys
import subprocess
import time

# Set up paths so the script can find ProtoCSP regardless of where it's run from
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def main():
    print("=" * 80)
    print("TESTING HEA SURFACE ALLOY (SQS + Vacuum + MLIP)")
    print("=" * 80)

    try:
        from pymatgen.core import Structure, Lattice
        from pymatgen.core.surface import SlabGenerator
        
        print("[INFO] Generating base BCC Fe (001) surface...")
        # Create standard BCC Iron
        lattice = Lattice.cubic(2.866)
        fe_bcc = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        
        # Cut the (001) slab with a 15 Angstrom vacuum gap
        slabgen = SlabGenerator(fe_bcc, (0, 0, 1), min_slab_size=10, min_vacuum_size=15)
        slab = slabgen.get_slab()
        
        # Make it larger in the a/b plane to distribute multiple elements properly
        slab.make_supercell([2, 2, 1])
        
        # Save it to the test directory using an absolute path
        base_cif_path = os.path.join(current_dir, "fe_bcc_001_surface.cif")
        slab.to(filename=base_cif_path, fmt="cif")
        print(f"[INFO] Saved pristine surface with {slab.num_sites} atoms to: \n       {base_cif_path}")
        
    except ImportError:
        print("[ERROR] Pymatgen is required for this test.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to generate base surface: {e}")
        sys.exit(1)

    # Define the complex target alloy
    comp = "Fe0.6Ni0.15Co0.15Mo0.05Pt0.05"
    mode_str = "SQS + MLIP (Surface)"
    
    print(f"\nTesting: {comp:<30} [Mode: {mode_str}]\n")
    
    try:
        main_script_path = os.path.join(parent_dir, "protocsp", "main.py")
        
        # Construct the CLI command
        cmd = [
            sys.executable, main_script_path, comp,
            "--base-cif", base_cif_path,
            "--verbose",
            "--output-dir", os.path.join(current_dir, "test_surface_outputs"),
            "--top-k", "3",
            "--save-cif",
            "--sqs", 
            "--sqs-iterations", "5000000",
            "--mlip", 
            "--engine", "mace"
        ]

        start_time = time.time()
        
        # Run ProtoCSP from the parent directory so it can easily find the LeMat DB for reference energies
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            cwd=parent_dir
        )
        
        captured_output = []
        candidates = 0
        mlip_success = False
        
        # Stream output to terminal in real-time
        for line in process.stdout:
            print(line, end="")
            captured_output.append(line)
            
            # Catch candidate count
            if "GENERATION COMPLETE" in line and "candidates" in line:
                try: 
                    candidates = int(line.split('(')[1].split('candidates')[0].strip())
                except: 
                    pass
            
            # Catch MLIP success metric
            if "MLIP Evaluation finished" in line or "E_form:" in line:
                mlip_success = True

        process.wait()
        elapsed = time.time() - start_time

        # Verify execution
        has_candidates = candidates > 0
        sqs_success = any("applying sqs generation" in line.lower() for line in captured_output)

        success = (process.returncode == 0) and has_candidates and sqs_success and mlip_success
        status = "SUCCESS" if success else "FAILED"
        print(f"\n[{status}] {candidates} candidates evaluated with MLIP ({elapsed:.2f}s)\n" + "-"*80 + "\n")

        reason = "N/A"
        if not success:
            if process.returncode != 0: 
                reason = f"CLI Process Crashed. Return code: {process.returncode}"
            elif not has_candidates: 
                reason = "No candidates found."
            elif not sqs_success: 
                reason = "SQS Fallback Triggered. Failed to use requested SQS method."
            elif not mlip_success: 
                reason = "MLIP evaluation failed or was skipped."
            print(f"  >>> REASON: {reason}")

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")

if __name__ == "__main__":
    main()