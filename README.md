# ProtoCSP

<p align="center">
  <img src="imgs/ProtoCSP_logo.png" alt="ProtoCSP Logo" width="200"/>
</p>

**An automated toolkit for the accelerated discovery of solid solutions, high-entropy alloys, and complex crystals through prototype transmutation, combinatorial superstructure generation, and machine learning interatomic potential (MLIP) validation.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/rogeriog/ProtoCSP)

---

## Overview

ProtoCSP (Prototype-based Crystal Structure Prediction) is an open-source Python framework designed to seamlessly integrate with [VibroML](https://github.com/rogeriog/VibroML) for advanced materials discovery. It navigates the complex chemical spaces of novel solid solutions and alloying by generating initial structural guesses and combinatorial ordered superstructures.

When investigating materials that suffer from geometric frustration or dynamical instability, ProtoCSP and VibroML operate iteratively. While VibroML maps competing lower-symmetry polymorphs, ProtoCSP systematically models targeted alloying across these phases to evaluate thermodynamic stability, allowing researchers to chemically engineer away lattice instabilities.

The tool operates in two primary modes:
1. **Combinatorial Superstructure Generation** for exploring solid solutions and fractional stoichiometries.
2. **Prototype Retrieval and Substitution** for "cold start" discovery of entirely novel compounds leveraging the LeMat-Bulk dataset.

---

## Features

### Core Capabilities
* **Combinatorial Superstructure Generation:** For targeted fractional compositions (e.g., Cs₂(K₀.₈Na₀.₂)In(I₀.₆₇Br₀.₃₃)₆), ProtoCSP generates exact integer-ratio supercells and enumerates all symmetrically distinct atomic orderings.
* **Exact Stoichiometry Matching:** Queries an indexed library of anonymized stoichiometries (e.g., ABC₃) to retrieve parent structures that match the target composition exactly.
* **Intelligent Chemical Substitution:** Maps target elements ($T$) to prototype species ($P$) by ranking both according to their Pauling electronegativity ($\chi$). The mapping $f: T \to P$ ensures electropositive target elements occupy electropositive sites, minimizing the risk of placing cations in anion positions.
* **Physical Validity Checks:** Isotropicaly scales unit cell lattice vectors based on the sum of covalent radii to prevent atomic overlap, followed by rigorous density checks to filter out unphysical configurations.
* **Complex Alloy Support:** For compositions with $>4$ elements where exact matches are rare, ProtoCSP employs a fuzzy matching algorithm identifying prototypes with overlapping element sets, followed by random substitution and decorrelation of site occupancy.

### MLIP Integration
Generated configurations can be immediately evaluated and relaxed to pinpoint the lowest-energy atomic ordering. ProtoCSP supports multiple equivariant and graph-based MLIP engines:
* **MACE (Message Passing Atomic Cluster Expansion):** Medium-omat and other MACE-MP models (Highly recommended for high-frequency mode accuracy).
* **eSEN (equivariant Smooth Energy Network):** Equivariant Transformer from the Open Catalyst Project (eSEN-30M-OMA).
* **UMA (Universal Models for Atoms):** Meta FAIR's universal machine-learning models (UMA-m-1p1).
* **M3GNet (Materials 3-body Graph Network):** Invariant graph neural network architecture incorporating 3-body interactions.

---

## Installation

### Prerequisites
* Python 3.8 or higher
* pip package manager

### Step 1: Clone the Repository
```bash
git clone [https://github.com/rogeriog/ProtoCSP.git](https://github.com/rogeriog/ProtoCSP.git)
cd ProtoCSP
````

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional MLIP Dependencies

For MLIP evaluation and structural relaxation, install the corresponding packages:

```bash
# MACE (recommended)
pip install mace-torch

# FairChem (eSEN/UMA)
pip install fairchem

# M3GNet
pip install m3gnet
```

-----

## Loading the LeMat Database

ProtoCSP requires the LeMat-Bulk dataset for prototype retrieval and structure generation. The database is hosted on HuggingFace and can be downloaded as parquet files for efficient loading.

### Prerequisites

* **Disk Space:** ~3.2GB for the parquet files
* **Dependencies:** `huggingface_hub` (included in requirements.txt)

### Download Instructions

1. Navigate to the `lemat_database_scripts/` directory:
```bash
cd lemat_database_scripts/
```

2. Run the download script:
```bash
python lemat_parquet_download.py
```

This will download the LeMat-BulkUnique dataset (unique_pbe configuration) and save the parquet files to `./lemat_parquet_files/unique_pbe/`.

### Configuration

ProtoCSP is configured to use the parquet files by default:

* The `--index` parameter in `main.py` defaults to `lemat_parquet_files/unique_pbe/`
* No additional configuration is needed if you use the default download location
* To use a custom path, specify it with the `--index` flag:
```bash
python main.py SrTiO3 --index /path/to/your/parquet/files
```

### Verification

To verify the database is loaded correctly, run a simple test:

```bash
cd ..
python main.py SrTiO3 --top-k 1 --verbose
```

If successful, you should see output indicating that ProtoCSP is using parquet lazy-scan mode and retrieving structures from the database.

-----

## Quick Start

### Basic Structure Generation

Generate structural guesses for a simple oxide using cold-start prototype retrieval:

```bash
python main.py SrTiO3 --top-k 10 --save-cif
```

Generate combinatorial guesses for a doped perovskite:

```bash
python main.py La0.5Sr0.5MnO3 --top-k 5 --verbose
```

Generate complex alloy structure candidates using fuzzy matching:

```bash
python main.py Fe0.8C0.1Mn0.05Cr0.03Ni0.02 --top-k 5 --verbose
```

### With MLIP Evaluation

Evaluate candidates with MACE and compute formation energies:

```bash
python main.py SrTiO3 --mlip --engine mace --top-k 5
```

Relax candidates with an MLIP to find the lowest-energy ordering:

```bash
python main.py La0.5Sr0.5MnO3 --mlip --engine mace --fmax 0.01 --save-cif
```

-----

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
| `--engine` | MLIP engine: `mace`, `esen`, `uma`, `m3gnet` |
| `--fmax` | Force tolerance for relaxation (eV/Å) |
| `--checkpoint-model` | Path to checkpoint file (required for eSEN/UMA) |

-----

## How It Works

ProtoCSP effectively translates theoretical discovery into computable structures through two main pathways:

### I. Combinatorial Superstructure Generation for Alloying

When exploring solid solutions to relieve geometric frustration, ProtoCSP takes a base crystal structure (e.g., a specific polymorph mapped by VibroML) and a target fractional composition. It automatically scales exact integer-ratio supercells, enumerates all symmetrically distinct atomic orderings using Pymatgen/ASE algorithms, and interfaces directly with an MLIP engine to perform rapid relaxations. This isolates the lowest-energy atomic ordering, which can then be fed back into VibroML for rigorous thermal stability tracking.

### II. Prototype Retrieval and Substitution (Cold Start)

To explore new chemical spaces without known starting structures, ProtoCSP queries the LeMat-Bulk dataset:

1.  **Retrieval:** Searches an indexed library of anonymized stoichiometries.
2.  **Chemical Substitution:** Maps elements based on Pauling electronegativity rankings to ensure chemical plausibility (cations to cation sites).
3.  **Scaling & Validation:** Scales lattice vectors isotropically based on covalent radii to prevent overlap and performs rigorous density verifications.

-----

## Output

### Console Output

For each candidate, ProtoCSP reports:

  * Reduced formula
  * Source prototype ID
  * Parent space group
  * Final space group (after generation)
  * Volume (Å³)
  * Energy per atom & Formation energy per atom (if `--mlip` is enabled)

### Files Generated

  * **CIF files:** Optional output for visualization (VESTA) or external workflows.
  * **JSON summary:** MLIP evaluation results with energies, physical metrics, and rankings.

-----

## Authors & Citation

**Rogério Almeida Gouvêa** and **Gian-Marco Rignanese** *Institute of Condensed Matter and Nanosciences, Université Catholique de Louvain, Louvain-la-Neuve, Belgium.*

If you use ProtoCSP in your research, please cite our corresponding paper:

```bibtex
@article{Gouvea2024VibroML,
  title = {VibroML: an automated toolkit for high-throughput vibrational analysis and dynamic instability remediation of crystalline materials using machine-learned potentials},
  author = {Gouv{\^e}a, Rog{\'e}rio Almeida and Rignanese, Gian-Marco},
  year = {2024},
  url = {[https://github.com/rogeriog/ProtoCSP](https://github.com/rogeriog/ProtoCSP)}
}
```

-----

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the authors:

  * Rogério Almeida Gouvêa: rogeriog.em@gmail.com
  * Gian-Marco Rignanese: gian-marco.rignanese@uclouvain.be

-----

## Acknowledgments

ProtoCSP builds upon several open-source projects:

  * [VibroML](https://github.com/rogeriog/VibroML) for dynamical stability analysis.
  * [pymatgen](https://github.com/materialsproject/pymatgen) for crystal structure analysis.
  * [ASE](https://wiki.fysik.dtu.dk/ase/) for atomic simulation.
  * [LeMat-Bulk](https://lematerial.org/) database for structure data.
  * [MACE](https://github.com/ACEsuit/mace), [FairChem](https://github.com/FAIR-Chem/fairchem), and other MLIP developers.
