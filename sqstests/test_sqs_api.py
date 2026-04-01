#!/usr/bin/env python3
"""
Simple test to understand sqsgenerator API
"""

import sqsgenerator
from pymatgen.core import Structure, Element
import numpy as np

# Create a simple test structure
from pymatgen.core.lattice import Lattice

lattice = Lattice.cubic(4.0)
species = ["K", "K", "Na", "Na"]
coords = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.75],
]

structure = Structure(lattice, species, coords)
print(f"Original structure: {structure.composition.reduced_formula}")

# Convert to sqsgenerator format
sqs_structure = sqsgenerator.from_pymatgen(structure)
print(f"Converted to sqsgenerator structure: {type(sqs_structure)}")

# Try a simple configuration
config_dict = {
    'structure': {
        'lattice': structure.lattice.matrix.tolist(),
        'coords': [site.frac_coords.tolist() for site in structure.sites],
        'species': [str(site.specie) for site in structure.sites]
    },
    'composition': {'K': 2, 'Na': 2},
    'which': 'K',
    'iterations': 100,
    'max_output_configurations': 3
}

print("Testing configuration parsing...")
try:
    parsed_config = sqsgenerator.parse_config(config_dict)
    print(f"✓ Configuration parsed successfully: {type(parsed_config)}")
    
    print("Testing optimization...")
    result_pack = sqsgenerator.optimize(parsed_config)
    print(f"✓ Optimization completed: {type(result_pack)}")
    print(f"Result pack attributes: {[attr for attr in dir(result_pack) if not attr.startswith('_')]}")
    
    # Check different possible result attributes
    for attr_name in ['results', 'configurations', 'structures', 'best', 'num_results']:
        if hasattr(result_pack, attr_name):
            attr_value = getattr(result_pack, attr_name)
            print(f"  {attr_name}: {type(attr_value)} = {attr_value}")
    
    # Try to get the best result
    if hasattr(result_pack, 'best'):
        try:
            best_result = result_pack.best()
            print(f"Best result: {type(best_result)}")
            print(f"Best result attributes: {[attr for attr in dir(best_result) if not attr.startswith('_')]}")
            
            if hasattr(best_result, 'objective'):
                print(f"  Objective: {best_result.objective}")
            if hasattr(best_result, 'structure'):
                print(f"  Structure: {type(best_result.structure)}")
                # Convert back to pymatgen
                pymatgen_struct = sqsgenerator.to_pymatgen(best_result.structure)
                print(f"  Pymatgen structure: {pymatgen_struct.composition.reduced_formula}")
        except Exception as e:
            print(f"Error calling best(): {e}")
    
    # Check num_results
    if hasattr(result_pack, 'num_results'):
        print(f"Number of results: {result_pack.num_results}")
        
    # Try to iterate through results
    try:
        print("Trying to iterate through results...")
        for i, result in enumerate(result_pack):
            if i >= 3:  # Just show first 3
                break
            print(f"  Result {i}: {type(result)}")
            if hasattr(result, 'objective'):
                print(f"    Objective: {result.objective}")
    except Exception as e:
        print(f"Error iterating: {e}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
