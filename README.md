# ProtoCSP

<p align="center">
  <img src="imgs/ProtoCSP_logo.png" alt="ProtoCSP Logo" width="400"/>
</p>

**Accelerated discovery of solid solutions, high-entropy alloys, and complex crystals through prototype transmutation and machine learning interatomic potential (MLIP) validation.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.XXXXX/xxxxx-green.svg)](https://doi.org/10.XXXXX/xxxxx)

---

## Overview

ProtoCSP (Prototype-based Crystal Structure Prediction) is an open-source Python framework for generating initial structural guesses for materials discovery. It leverages a large database of known crystal structures (from the LeMat database) and uses intelligent prototype matching, element transmutation, and doping strategies to propose candidate structures for any target chemical composition.

The tool is particularly effective for:
- **Solid solutions** and doped materials (e.g., La₁₋ₓSrₓMnO₃)
- **High-entropy alloys** (e.g., FeMnCoNiCr-based alloys)
- **Complex oxides** and perovskites
- **Fractional stoichiometry compositions** (e.g., Li₀.₅CoO₂)

Generated structures can optionally be evaluated and relaxed using state-of-the-art machine learning interatomic potentials (MLIPs) including MACE, eSEN, M3GNet, and GPUMD-NEP.

---

## Features

### Core Capabilities
- **Exact Stoichiometry Matching**: Finds prototypes with matching anonymized stoichiometry and substitutes elements based on electronegativity ordering
- **Element Transmutation**: Maps target elements to prototype sites by sorting both by electronegativity, ensuring electropositive elements occupy electropositive sites
- **Fractional Stoichiometry Handling**: For non-integer compositions (e.g., Li₀.₅CoO₂), finds integer parent structures and generates ordered vacancy configurations using supercell enumeration
- **Alloy Support**: Handles complex multi-element alloys (>4 elements) using bucket-filling algorithms and farthest-point sampling for diverse configurations
- **Physical Validity Checks**: Scales lattices using average atomic radii, validates against density thresholds, and checks for atomic overlaps
- **Fuzzy Matching**: Finds prototypes with overlapping element sets for compositions without exact matches

### MLIP Integration
ProtoCSP supports evaluation and relaxation with multiple MLIP engines:
- **MACE**: Medium-omat and other MACE-MP models
- **eSEN**: Equivariant Transformer from Open Catalyst Project
- **UMA**: Universal MLIP for Materials (FairChem)
- **M3GNet**: Many-body Graph Network
- **GPUMD-NEP**: GPU-accelerated Neural Equilibrium Potential
- **Calorine**: CPU-based NEP calculator

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-repo/ProtoCSP.git
cd ProtoCSP
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Optional MLIP Dependencies
For MLIP evaluation and relaxation, install additional packages:
```bash
# MACE (recommended)
pip install mace-torch

# FairChem (eSEN/UMA)
pip install fairchem

# M3GNet
pip install m3gnet

```

---

## Quick Start

### Basic Structure Generation

Generate structural guesses for a simple oxide:
```bash
python main.py SrTiO3 --top-k 10 --save-cif
```

Generate guesses for a doped perovskite:
```bash
python main.py La0.5Sr0.5MnO3 --top-k 5 --verbose
```

Generate alloy structure candidates:
```bash
python main.py Fe0.8C0.1Mn0.05Cr0.03Ni0.02 --top-k 5 --verbose
```

### With MLIP Evaluation

Evaluate candidates with MACE and compute formation energies:
```bash
python main.py SrTiO3 --mlip --engine mace --top-k 5
```

Relax candidates with a specific MLIP:
```bash
python main.py La0.5Sr0.5MnO3 --mlip --engine mace --fmax 0.01 --save-cif
```

---

## Command-Line Options

### Main Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `composition` | Target chemical composition (required) | - |
| `--csv-dir` | Directory containing LeMat CSV files | `../lemat_unique_csv_500_parts` |
| `--output-dir` | Output directory for generated structures | `./generated_structures` |
| `--top-k` | Number of candidates to generate | 5 |
| `--max-bases` | Maximum different parent structures to use | 3 |

### Generation Options
| Argument | Description |
|----------|-------------|
| `--min-atoms` | Minimum atoms in supercell (default: 20) |
| `--randomize-scaling` | Randomize supercell dimensions for diversity |
| `--symmetrize` | Use rigorous enumeration for high-symmetry ordered structures |
| `--base-cif` | Use a manual CIF file as structural base |
| `--save-cif` | Save generated structures as CIF files |
| `--verbose` | Print detailed progress information |

### MLIP Options
| Argument | Description |
|----------|-------------|
| `--mlip` | Activate MLIP energy evaluation |
| `--no-relax` | Skip relaxation, only compute single-point energies |
| `--engine` | MLIP engine: mace, esen, uma, m3gnet, gpumd, nep, calorine |
| `--fmax` | Force tolerance for relaxation (eV/Å) |
| `--checkpoint-model` | Path to checkpoint file (required for eSEN/UMA/NEP) |

---

## How It Works

### 1. Library Building
ProtoCSP loads crystal structures from the LeMat database (stored as CSV files) and indexes them by their **anonymized stoichiometry** (e.g., "AB", "AB₂C₄", "A₂B₃"). This enables fast lookup of potential parent structures.

### 2. Prototype Selection
For a target composition, ProtoCSP finds matching prototypes using multiple strategies:
1. **Exact Match**: Direct lookup of structures with identical stoichiometry
2. **Transmutation**: 1-to-1 element substitution based on electronegativity
3. **Generalized Grouping**: Bucket-filling algorithm for complex multi-element compositions
4. **Recursive Reduction**: Strip minority elements to find simpler parent structures

### 3. Supercell Generation
For non-integer stoichiometries or when the target size differs from the prototype:
- Smart cubic scaling to reach target atom count
- Optional randomization for structural diversity
- Symmetry-preserving enumeration for ordered configurations

### 4. Element Mapping
Elements are mapped to crystallographic sites using:
- **Electronegativity ordering**: More electropositive elements occupy electropositive sites
- **Group similarity**: Elements in the same periodic group preferred for substitution
- **Radius ratio analysis**: Determines substitutional vs. interstitial insertion

### 5. Validation
Generated structures are validated through:
- Lattice scaling based on average atomic radii
- Density range checking (0.1 < ρ < 60 g/cm³)
- Minimum distance verification (d > 0.5 Å)
- Structural fingerprinting for deduplication

---

## Output

### Console Output
For each candidate, ProtoCSP reports:
- Reduced formula
- Source prototype ID
- Parent space group
- Final space group (after generation)
- Volume (Å³)
- Energy per atom (if MLIP enabled)
- Formation energy per atom (if MLIP enabled)

### Files Generated
- **CIF files**: Optional CIF output for visualization (VESTA, Mercury) or further processing
- **JSON summary**: MLIP evaluation results with energies and rankings (when `--mlip` enabled)

---

## Directory Structure

```
ProtoCSP/
├── main.py                 # Main entry point
├── core.py                 # ProtoCSP core class and algorithms
├── mlip_utils.py           # MLIP calculator utilities
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── imgs/
│   └── ProtoCSP_logo.png  # Project logo
└── protocsp/
    ├── __init__.py
    └── (additional modules)
```

---

## Performance Notes

- For large datasets, use `--limit` to restrict the number of structures loaded for testing
- The LeMat CSV directory should be at `../lemat_unique_csv_500_parts` relative to the ProtoCSP folder
- Generated structures are **initial guesses** and should be relaxed with appropriate MLIPs for accurate energy comparisons

---

## Dependencies

### Core
- **pymatgen**: Crystal structure manipulation and analysis
- **pandas**: Data loading and processing
- **numpy**: Numerical operations
- **tqdm**: Progress bar display
- **ASE**: Atomic simulation environment

### Optional (MLIP)
- **mace-torch**: MACE MLIP
- **fairchem**: eSEN/UMA calculators
- **m3gnet**: M3GNet calculator
- **torch**: PyTorch (for CUDA-based MLIPs)

---

## Citation

If you use ProtoCSP in your research, please cite:

```bibtex
@article{ProtoCSP2024,
  title = {ProtoCSP: Prototype-based Crystal Structure Prediction for Materials Discovery},
  author = {Gouvea, R. and contributors},
  journal = {GitHub Repository},
  year = {2024},
  url = {https://github.com/your-repo/ProtoCSP}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the authors at: rogeriog.em@gmail.com

---

## Acknowledgments

ProtoCSP builds upon several open-source projects:
- [pymatgen](https://github.com/materialsproject/pymatgen) for crystal structure analysis
- [ASE](https://wiki.fysik.dtu.dk/ase/) for atomic simulation
- [LeMat](https://lematerial.org/) database for structure data
- [MACE](https://github.com/ACEsuit/mace), [FairChem](https://github.com/FAIR-Chem/fairchem), and other MLIP developers