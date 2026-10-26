"""
ProtoCSP - Prototype-based Crystal Structure Prediction

Fast tensor pipeline and combinatorial structure generation.
"""

# Polyatomic ion utilities
from .polyatomic_utils import (
    expand_polyatomics_in_structure,
    process_npz_with_polyatomics,
    list_available_molecules,
    get_molecule_xyz,
    BUILTIN_MOLECULES
)

__all__ = [
    'expand_polyatomics_in_structure',
    'process_npz_with_polyatomics',
    'list_available_molecules',
    'get_molecule_xyz',
    'BUILTIN_MOLECULES'
]
