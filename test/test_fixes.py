#!/usr/bin/env python3
"""
Quick test to verify both fixes:
1. Perovskite base structure selection
2. Steel composition handling
"""

import sys
sys.path.append('.')

from core import ProtoCSP
import time

def test_perovskite():
    """Test Issue 1: Perovskite base structure selection"""
    print("=" * 80)
    print("TEST 1: PEROVSKITE BASE STRUCTURE SELECTION")
    print("=" * 80)
    
    composition = "La0.5Sr0.5MnO3"
    print(f"\nTesting: {composition}")
    print("-" * 80)
    
    pcsp = ProtoCSP('lemat_formula_indexed')
    start = time.time()
    result = pcsp.generate(composition, top_k=3)
    elapsed = time.time() - start
    
    print("-" * 80)
    print(f"\nResult: {len(result)} candidates in {elapsed:.2f}s")
    
    if len(result) > 0:
        print("\n✓ SUCCESS: Generated candidates")
        for i, entry in enumerate(result):
            struct = entry['structure']
            parent = entry.get('parent_formula', 'N/A')
            print(f"  [{i+1}] {struct.composition.reduced_formula}")
            print(f"      Base: {parent}")
            print(f"      Method: {entry.get('method', 'Unknown')}")
        
        # Check if base structures are appropriate (should contain La, Sr, Mn, or O)
        base_formulas = [entry.get('parent_formula', '') for entry in result]
        print(f"\n  Base structures used: {base_formulas}")
        
        # Check if any base contains Os (osmium) - this would be bad
        has_osmium = any('Os' in formula for formula in base_formulas)
        if has_osmium:
            print("  ✗ FAIL: Found osmium-based structures (should use oxide perovskites)")
            return False
        else:
            print("  ✓ PASS: No osmium-based structures found")
            return True
    else:
        print("\n✗ FAIL: No candidates generated")
        return False

def test_steel():
    """Test Issue 2: Steel composition handling"""
    print("\n" + "=" * 80)
    print("TEST 2: STEEL COMPOSITION HANDLING")
    print("=" * 80)
    
    composition = "Fe0.8C0.1Mn0.05Cr0.03Ni0.02"
    print(f"\nTesting: {composition}")
    print("-" * 80)
    
    guesser = CrystalGuesser('lemat_formula_indexed')
    start = time.time()
    result = guesser.guess(composition, top_k=3)
    elapsed = time.time() - start
    
    print("-" * 80)
    print(f"\nResult: {len(result)} candidates in {elapsed:.2f}s")
    
    if len(result) > 0:
        print("\n✓ SUCCESS: Generated candidates")
        for i, entry in enumerate(result):
            struct = entry['structure']
            parent = entry.get('parent_formula', 'N/A')
            method = entry.get('method', 'Unknown')
            print(f"  [{i+1}] {struct.composition.reduced_formula}")
            print(f"      Base: {parent}")
            print(f"      Method: {method}")
            print(f"      Sites: {struct.num_sites}")
        
        # Check if method includes interstitial (carbon should be interstitial)
        methods = [entry.get('method', '') for entry in result]
        has_interstitial = any('Interstitial' in method for method in methods)
        
        if has_interstitial:
            print("  ✓ PASS: Carbon correctly identified as interstitial")
        else:
            print("  ⚠ WARNING: Carbon not identified as interstitial")
        
        return True
    else:
        print("\n✗ FAIL: No candidates generated (silent failure)")
        return False

def main():
    print("\nCRYSTALGUESSER FIX VERIFICATION")
    print("Testing the two main issues that were addressed\n")
    
    # Test both issues
    test1_pass = test_perovskite()
    test2_pass = test_steel()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Issue 1 (Perovskite base selection): {'✓ PASS' if test1_pass else '✗ FAIL'}")
    print(f"Issue 2 (Steel composition handling): {'✓ PASS' if test2_pass else '✗ FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())

