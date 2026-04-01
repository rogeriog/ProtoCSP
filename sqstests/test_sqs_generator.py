#!/usr/bin/env python3
"""
Test script for SQS generator functionality in ProtoCSP.

This script tests:
1. sqsgenerator availability and basic functionality
2. SQS generation with sample structures
3. Integration with ProtoCSP --sqs flag
4. Combined --sqs and --mlip workflow

Usage:
    python test_sqs_generator.py [--test-basic|--test-integration|--test-all]
"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path

# Add the current directory to Python path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pymatgen.core import Structure, Element, Composition
    from pymatgen.io.ase import AseAtomsAdaptor
except ImportError:
    print("Error: 'pymatgen' library not found. Please run 'pip install pymatgen'")
    sys.exit(1)

try:
    from sqs_generator import SQSGenerator, generate_sqs_structures, SQSGenerationError
except ImportError as e:
    print(f"Error importing sqs_generator: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_structure():
    """Create a simple test structure (K2NaAlF6 perovskite-like)."""
    try:
        # Try to load the base structure if it exists
        if os.path.exists('base_K2NaAlF6.cif'):
            structure = Structure.from_file('base_K2NaAlF6.cif')
            logger.info(f"Loaded base structure from base_K2NaAlF6.cif: {structure.composition.reduced_formula}")
            return structure
    except Exception as e:
        logger.warning(f"Could not load base_K2NaAlF6.cif: {e}")
    
    # Create a simple cubic structure for testing
    logger.info("Creating simple cubic test structure...")
    from pymatgen.core.lattice import Lattice
    
    # Simple cubic lattice
    lattice = Lattice.cubic(4.0)
    
    # Create a simple structure with K and Na
    species = ["K", "Na", "Al", "F", "F", "F"]
    coords = [
        [0.0, 0.0, 0.0],  # K
        [0.5, 0.5, 0.5],  # Na
        [0.25, 0.25, 0.25],  # Al
        [0.75, 0.75, 0.75],  # F
        [0.75, 0.25, 0.25],  # F
        [0.25, 0.75, 0.75],  # F
    ]
    
    structure = Structure(lattice, species, coords)
    logger.info(f"Created test structure: {structure.composition.reduced_formula}")
    return structure


def test_sqsgenerator_availability():
    """Test if sqsgenerator is available and working."""
    print("\n" + "="*60)
    print("TEST 1: sqsgenerator Availability Check")
    print("="*60)
    
    try:
        import sqsgenerator
        print(f"✓ sqsgenerator imported successfully")
        
        # Check version
        if hasattr(sqsgenerator, '__version__'):
            print(f"✓ sqsgenerator version: {sqsgenerator.__version__}")
        else:
            print("✓ sqsgenerator version: unknown")
        
        # Test basic functions
        print("✓ Available functions:", [func for func in dir(sqsgenerator) if not func.startswith('_')])
        
        return True
        
    except ImportError as e:
        print(f"✗ sqsgenerator not available: {e}")
        return False


def test_sqs_generator_class():
    """Test the SQSGenerator class functionality."""
    print("\n" + "="*60)
    print("TEST 2: SQSGenerator Class Functionality")
    print("="*60)
    
    try:
        # Initialize generator
        generator = SQSGenerator(method='sqsgenerator', fallback_method='atat')
        print("✓ SQSGenerator initialized successfully")
        
        # Get method info
        info = generator.get_method_info()
        print(f"✓ Method info: {info}")
        
        # Test with a simple structure
        test_struct = create_test_structure()
        
        # Test SQS generation
        print("\nTesting SQS generation...")
        try:
            sqs_structures = generator.generate_sqs(
                base_structure=test_struct,
                alloy_spec="K:Na",
                composition=0.5,  # 50% K, 50% Na
                supercell=(2, 2, 2),
                num_structures=2,
                iterations=1000  # Small number for quick testing
            )
            
            print(f"✓ Generated {len(sqs_structures)} SQS structures")
            
            for i, struct in enumerate(sqs_structures):
                print(f"  Structure {i+1}: {struct.composition.reduced_formula}, "
                      f"{struct.num_sites} sites, volume={struct.volume:.2f} Å³")
                struct.to(filename=f"test2_sqs_output_{i+1}.cif", fmt="cif")
            # Test quality analysis
            print("\nTesting SQS quality analysis...")
            quality = generator.analyze_sqs_quality(sqs_structures)
            print(f"✓ Quality analysis: {quality}")
            
            return True
            
        except SQSGenerationError as e:
            print(f"✗ SQS generation failed: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error in SQS generation: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ SQSGenerator class test failed: {e}")
        traceback.print_exc()
        return False


def test_convenience_function():
    """Test the convenience function."""
    print("\n" + "="*60)
    print("TEST 3: Convenience Function")
    print("="*60)
    
    try:
        test_struct = create_test_structure()
        
        sqs_structures = generate_sqs_structures(
            base_structure=test_struct,
            alloy_spec="K:Na",
            composition=0.3,  # 30% K, 70% Na
            supercell=(2, 2, 1),
            num_structures=1,
            method='sqsgenerator',
            iterations=500
        )
        
        print(f"✓ Convenience function generated {len(sqs_structures)} structures")
        
        if sqs_structures:
            struct = sqs_structures[0]
            print(f"  Structure: {struct.composition.reduced_formula}, "
                  f"{struct.num_sites} sites")
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience function test failed: {e}")
        traceback.print_exc()
        return False


def test_protocsp_integration():
    """Test integration with ProtoCSP --sqs flag."""
    print("\n" + "="*60)
    print("TEST 4: ProtoCSP Integration (--sqs flag)")
    print("="*60)
    
    try:
        # Import ProtoCSP
        from protocsp.core import ProtoCSP
        
        print("✓ ProtoCSP imported successfully")
        
        # Create a simple mock library for testing
        mock_library = {
            'K2NaAlF6': [{
                'id': 'test_K2NaAlF6',
                'reduced_formula': 'K2NaAlF6',
                'energy_per_atom': -5.0,
                'structure': create_test_structure(),
                'parent_space_group': 'Pm-3m'
            }]
        }
        
        # Initialize ProtoCSP
        generator = ProtoCSP(mock_library)
        print("✓ ProtoCSP initialized with mock library")
        
        # Test SQS generation
        print("\nTesting ProtoCSP SQS generation...")
        candidates = generator.generate(
            target_composition_str="KNaAlF6",  # Different composition to trigger substitution
            max_bases=1,
            top_k=2,
            min_atoms=10,
            sqs=True
        )
        
        print(f"✓ ProtoCSP generated {len(candidates)} candidates with --sqs flag")
        
        for i, candidate in enumerate(candidates):
            struct = candidate['structure']
            method = candidate.get('method', 'Unknown')
            print(f"  Candidate {i+1}: {struct.composition.reduced_formula}, "
                  f"method={method}, {struct.num_sites} sites")
        
        return True
        
    except Exception as e:
        print(f"✗ ProtoCSP integration test failed: {e}")
        traceback.print_exc()
        return False


def test_mlip_integration():
    """Test combined --sqs and --mlip workflow."""
    print("\n" + "="*60)
    print("TEST 5: Combined SQS + MLIP Workflow")
    print("="*60)
    
    try:
        # Check if MLIP utilities are available
        from protocsp.mlip_utils import initialize_calculator
        print("✓ MLIP utilities imported successfully")
        
        # Try to initialize a simple calculator
        print("Testing MLIP calculator initialization...")
        calc = initialize_calculator(engine='mace', model_name='small')
        
        if calc is None:
            print("✗ Could not initialize MLIP calculator (MACE may not be installed)")
            return False
        
        print("✓ MLIP calculator initialized successfully")
        
        # Test SQS generation first
        test_struct = create_test_structure()
        sqs_structures = generate_sqs_structures(
            base_structure=test_struct,
            alloy_spec="K:Na",
            composition=0.4,
            supercell=(2, 2, 1),
            num_structures=4,
            iterations=500
        )
        
        if not sqs_structures:
            print("✗ No SQS structures generated for MLIP testing")
            return False
        
        print(f"✓ Generated {len(sqs_structures)} SQS structures for MLIP testing")
        
        # Test energy evaluation
        from pymatgen.io.ase import AseAtomsAdaptor
        adapter = AseAtomsAdaptor()
        
        for i, struct in enumerate(sqs_structures):
            atoms = adapter.get_atoms(struct)
            atoms.calc = calc
            
            try:
                energy = atoms.get_potential_energy()
                energy_per_atom = energy / len(atoms)
                print(f"✓ SQS structure {i+1} energy: {energy_per_atom:.4f} eV/atom")
                struct.to(filename=f"test5_mlip_sqs_output_{i+1}.cif", fmt="cif")
                print(f"✓ Saved to test5_mlip_sqs_output_{i+1}.cif")
            except Exception as e:
                print(f"✗ Energy evaluation failed for structure {i+1}: {e}")
                return False
        
        print("✓ Combined SQS + MLIP workflow test successful")
        return True
        
    except ImportError as e:
        print(f"✗ MLIP utilities not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Combined workflow test failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test SQS generator functionality')
    parser.add_argument('--test-basic', action='store_true', help='Run basic SQS tests only')
    parser.add_argument('--test-integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--test-all', action='store_true', help='Run all tests (default)')
    
    args = parser.parse_args()
    
    # Default to all tests if no specific test requested
    if not any([args.test_basic, args.test_integration]):
        args.test_all = True
    
    print("SQS Generator Test Suite")
    print("="*60)
    
    results = {}
    
    # Basic tests
    if args.test_basic or args.test_all:
        results['availability'] = test_sqsgenerator_availability()
        results['class_function'] = test_sqs_generator_class()
        results['convenience_function'] = test_convenience_function()
    
    # Integration tests
    if args.test_integration or args.test_all:
        results['protocsp_integration'] = test_protocsp_integration()
        results['mlip_integration'] = test_mlip_integration()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SQS generator is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
