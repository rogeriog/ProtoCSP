# coding: utf-8
#!/usr/bin/env python3
"""
Test CIF file generation with the updated main.py
"""

import os
import sys
import subprocess
import shutil

def test_cif_generation():
    print("=" * 80)
    print("TESTING CIF FILE GENERATION")
    print("=" * 80)
    
    # Clean up any existing test output
    output_dir = "./test_cif_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Test compositions
    test_cases = [
        ("TiO2", "simple oxide"),
        ("FeNi", "simple alloy"),
        ("LaMnO3", "perovskite"),
        ("La0.5Sr0.5MnO3", "perovskite"),
        ("Fe0.8C0.1Mn0.05Cr0.03Ni0.02", "steel")
    ]
    
    for comp, description in test_cases:
        print(f"\nTesting {description}: {comp}")
        print("-" * 80)
        
        # Run main.py
        # REMOVED capture_output=True so you can see the logs directly
        cmd = [
            sys.executable, "main.py",
            comp,
            "--top-k", "2",
            "--verbose",
            "--save-cif",
            "--output-dir", output_dir
        ]
        
        try:
            # check=True will raise CalledProcessError if returncode != 0
            subprocess.run(cmd, check=True) 
            print("✓ Command executed successfully")
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Command failed with return code {e.returncode}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("CIF GENERATION TEST COMPLETE")
    print("=" * 80)
    
    if os.path.exists(output_dir):
        all_cif_files = [f for f in os.listdir(output_dir) if f.endswith('.cif')]
        print(f"Total CIF files generated: {len(all_cif_files)}")
        print(f"Output directory: {output_dir}/")

if __name__ == "__main__":
    test_cif_generation()
