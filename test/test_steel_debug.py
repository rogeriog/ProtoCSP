#!/usr/bin/env python3
"""
Debug script for steel composition testing
"""

import sys
sys.path.append('.')

from lemat_unique_dataset.CrystalGuesser.core import CrystalGuesser

# Test steel composition
composition = "Fe0.8C0.1Mn0.05Cr0.03Ni0.02"

print(f"Testing steel composition: {composition}")
print("=" * 80)

guesser = CrystalGuesser('lemat_formula_indexed')
result = guesser.guess(composition, top_k=3)

print("=" * 80)
print(f"\nFinal result: {len(result)} candidates")

for i, entry in enumerate(result):
    struct = entry['structure']
    print(f"\n[{i+1}] {struct.composition.reduced_formula}")
    print(f"    Source: {entry.get('id', 'Unknown')}")
    print(f"    Method: {entry.get('method', 'Unknown')}")
    print(f"    Sites: {struct.num_sites}")
    print(f"    Volume: {struct.volume:.1f} A^3")

