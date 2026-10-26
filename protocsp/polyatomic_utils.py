"""
Polyatomic Ion Utilities for ProtoCSP

Handles substitution of dummy atoms (Fr, At, etc.) with molecular ions (NH4+, CN-, SCN-)
using the "Pseudo-Atom" approach for high-throughput generation.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

try:
    from pymatgen.core import Structure, Molecule, Element
    from scipy.spatial.transform import Rotation
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


# =============================================================================
# Built-in Molecule Definitions (XYZ format)
# =============================================================================

BUILTIN_MOLECULES = {
    "NH4": """5
Ammonium NH4+
N 0.000000 0.000000 0.000000
H 0.890000 0.890000 0.890000
H -0.890000 -0.890000 0.890000
H -0.890000 0.890000 -0.890000
H 0.890000 -0.890000 -0.890000
""",
    "CN": """2
Cyanide CN-
C 0.000000 0.000000 0.000000
N 0.000000 0.000000 1.160000
""",
    "SCN": """3
Thiocyanate SCN-
S 0.000000 0.000000 0.000000
C 0.000000 0.000000 1.550000
N 0.000000 0.000000 2.710000
"""
}


def get_molecule_xyz(molecule_name: str, custom_xyz_dir: Optional[str] = None) -> str:
    """
    Get XYZ content for a molecule.
    
    Priority:
    1. Custom XYZ file in custom_xyz_dir/{molecule_name}.xyz
    2. Built-in molecule definitions (NH4, CN, SCN)
    
    Args:
        molecule_name: Name of molecule (e.g., "NH4", "CN", "SCN")
        custom_xyz_dir: Optional directory with custom XYZ files
        
    Returns:
        XYZ file content as string
        
    Raises:
        ValueError: If molecule not found
    """
    # Check custom directory first
    if custom_xyz_dir:
        custom_path = os.path.join(custom_xyz_dir, f"{molecule_name}.xyz")
        if os.path.exists(custom_path):
            with open(custom_path, 'r') as f:
                return f.read()
    
    # Fall back to built-in
    if molecule_name.upper() in BUILTIN_MOLECULES:
        return BUILTIN_MOLECULES[molecule_name.upper()]
    
    raise ValueError(
        f"Molecule '{molecule_name}' not found. "
        f"Built-in options: {list(BUILTIN_MOLECULES.keys())}. "
        f"Provide custom XYZ file in: {custom_xyz_dir}"
    )


def load_molecule_from_xyz(xyz_content: str) -> 'Molecule':
    """
    Load a pymatgen Molecule from XYZ string content.
    
    Args:
        xyz_content: XYZ format string
        
    Returns:
        pymatgen Molecule object
    """
    if not HAVE_DEPS:
        raise ImportError("pymatgen is required for polyatomic operations")
    
    lines = xyz_content.strip().split('\n')
    # First line is number of atoms (optional in some formats)
    # Second line is comment
    # Remaining lines are atoms
    
    species = []
    coords = []
    
    for line in lines[2:]:  # Skip atom count and comment
        parts = line.strip().split()
        if len(parts) >= 4:
            elem = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            species.append(elem)
            coords.append([x, y, z])
    
    return Molecule(species, coords)


def rotate_molecule_random(mol: 'Molecule') -> 'Molecule':
    """
    Apply a random 3D rotation to a molecule.
    
    Args:
        mol: pymatgen Molecule
        
    Returns:
        New rotated molecule (original is modified)
    """
    if not HAVE_DEPS:
        raise ImportError("scipy is required for rotations")
    
    rot_matrix = Rotation.random().as_matrix()
    for site in mol:
        site.coords = np.dot(site.coords, rot_matrix)
    
    return mol


def expand_polyatomics_in_structure(
    structure: Union[Structure, 'Atoms'],
    polyatomic_mapping: Dict[str, str],
    custom_xyz_dir: Optional[str] = None,
    random_rotation: bool = True
) -> Structure:
    """
    Replace dummy atoms in a structure with polyatomic molecules.
    
    This is the core "Pseudo-Atom" function. It scans for dummy elements
    (like Fr, At) and replaces them with the specified molecules.
    
    Args:
        structure: pymatgen Structure or ASE Atoms
        polyatomic_mapping: Dict mapping dummy element to molecule name.
            Example: {"Fr": "NH4", "At": "SCN"}
        custom_xyz_dir: Optional directory with custom XYZ files
        random_rotation: If True, apply random 3D rotation to each molecule
        
    Returns:
        Structure with dummy atoms replaced by molecules
    """
    if not HAVE_DEPS:
        raise ImportError("pymatgen and scipy are required for polyatomic operations")
    
    # Convert ASE to pymatgen if needed
    if hasattr(structure, 'get_chemical_symbols'):  # ASE Atoms
        from ase.io import write
        import tempfile
        # Write to temp CIF and read back
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as tmp:
            tmp_path = tmp.name
        write(tmp_path, structure)
        struct = Structure.from_file(tmp_path)
        os.remove(tmp_path)
    else:
        struct = structure
    
    if not polyatomic_mapping:
        return struct.copy()
    
    new_struct = struct.copy()
    insertions = []
    deletions = []
    
    # Find all dummy atoms (iterate backwards)
    for i in reversed(range(len(new_struct))):
        site = new_struct[i]
        elem = str(site.specie)
        
        if elem in polyatomic_mapping:
            mol_name = polyatomic_mapping[elem]
            dummy_coords = site.coords
            
            deletions.append(i)
            
            try:
                xyz_content = get_molecule_xyz(mol_name, custom_xyz_dir)
                mol = load_molecule_from_xyz(xyz_content)
            except Exception as e:
                print(f"[Polyatomic] Warning: Could not load {mol_name}: {e}")
                continue
            
            if random_rotation:
                mol = rotate_molecule_random(mol)
            
            # Add molecule atoms at dummy position
            for mol_site in mol:
                insertions.append({
                    'specie': mol_site.specie,
                    'coords': mol_site.coords + dummy_coords
                })
    
    # Apply changes
    if deletions:
        new_struct.remove_sites(deletions)
    
    if insertions:
        for ins in insertions:
            new_struct.append(
                ins['specie'],
                ins['coords'],
                coords_are_cartesian=True
            )
        
        print(f"[Polyatomic] Replaced {len(deletions)} dummy atoms with {len(insertions)} atoms "
              f"({len(deletions)} molecules inserted)")
    
    return new_struct


def process_npz_with_polyatomics(
    npz_path: str,
    polyatomic_mapping: Dict[str, str],
    metadata_path: Optional[str] = None,
    custom_xyz_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_structures: Optional[int] = None
) -> List[Structure]:
    """
    Process NPZ tensor file and expand polyatomic ions.
    
    This is a convenience function for the fast pipeline output.
    
    Args:
        npz_path: Path to tensor_data.npz file
        polyatomic_mapping: Dict mapping dummy elements to molecules
        metadata_path: Optional path to tensor_metadata.json
        custom_xyz_dir: Optional directory with custom XYZ files
        output_dir: If provided, save expanded structures as CIF
        max_structures: Limit number of structures to process
        
    Returns:
        List of expanded pymatgen Structures
    """
    if not HAVE_DEPS:
        raise ImportError("pymatgen and scipy are required")
    
    # Import here to avoid circular imports
    from .tensor_generator import tensors_to_pymatgen
    
    data = np.load(npz_path)
    atomic_numbers = data['atomic_numbers']
    positions = data['positions']
    cell = data['cell']
    
    B = atomic_numbers.shape[0]
    if max_structures:
        B = min(B, max_structures)
    
    print(f"[Polyatomic] Processing {B} structures from {npz_path}")
    
    structures = []
    for i in range(B):
        # Convert tensor to structure
        structure = tensors_to_pymatgen(
            atomic_numbers[i],
            positions[i],
            cell[i],
            index=i
        )
        
        # Expand polyatomics
        expanded = expand_polyatomics_in_structure(
            structure,
            polyatomic_mapping,
            custom_xyz_dir,
            random_rotation=True
        )
        structures.append(expanded)
    
    # Save to CIF if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, struct in enumerate(structures):
            formula = struct.composition.reduced_formula
            filename = f"expanded_{i:04d}_{formula}.cif"
            filepath = os.path.join(output_dir, filename)
            struct.to(fmt="cif", filename=filepath)
        print(f"[Polyatomic] Saved {len(structures)} expanded structures to {output_dir}")
    
    return structures


def list_available_molecules() -> List[str]:
    """Return list of built-in molecule names."""
    return list(BUILTIN_MOLECULES.keys())


# =============================================================================
# Example usage and command-line interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Polyatomic ion expansion utilities')
    subparsers = parser.add_subparsers(dest='command')
    
    # List command
    subparsers.add_parser('list', help='List available built-in molecules')
    
    # Expand NPZ command
    expand_parser = subparsers.add_parser('expand', help='Expand NPZ tensors to CIF with polyatomic ions')
    expand_parser.add_argument('npz', help='Path to NPZ file')
    expand_parser.add_argument('--mapping', required=True, help='Mapping like "Fr:NH4,At:SCN"')
    expand_parser.add_argument('--metadata', help='Path to metadata JSON')
    expand_parser.add_argument('--output-dir', default='./expanded', help='Output directory')
    expand_parser.add_argument('--max', type=int, help='Max structures to process')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        print("Available built-in molecules:")
        for name in list_available_molecules():
            print(f"  - {name}")
    
    elif args.command == 'expand':
        # Parse mapping string
        mapping = {}
        for pair in args.mapping.split(','):
            dummy, mol = pair.split(':')
            mapping[dummy.strip()] = mol.strip()
        
        process_npz_with_polyatomics(
            args.npz,
            mapping,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            max_structures=args.max
        )
    
    else:
        parser.print_help()
