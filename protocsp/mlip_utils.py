import os
import sys
import shutil
import tempfile
import subprocess
import logging
import re
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
from ase.io import read

# --- Optional Dependency Checks ---
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel(logging.ERROR)
    HAVE_TENSORFLOW = True
except ImportError:
    HAVE_TENSORFLOW = False

try:
    from mace.calculators import mace_mp
    HAVE_MACE = True
except ImportError:
    HAVE_MACE = False
    mace_mp = None

HAVE_ESEN = False
HAVE_UMA = False
FAIRCHEM_API_VERSION = None

try:
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
    HAVE_ESEN = True
    FAIRCHEM_API_VERSION = "old"
except ImportError:
    try:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.pretrained_mlip import load_predict_unit
        HAVE_UMA = True
        FAIRCHEM_API_VERSION = "new"
    except ImportError:
        pass

try:
    from calorine.calculators import CPUNEP
    HAVE_CALORINE = True
except ImportError:
    HAVE_CALORINE = False
    CPUNEP = None

try:
    from m3gnet.models import M3GNet, M3GNetCalculator, Potential
    HAVE_M3GNET = True
except ImportError:
    HAVE_M3GNET = False

HAVE_GPUMD = False
GPUMD_BINARY_PATH = None
gpumd_search_paths = [
    "/auto/globalscratch/users/r/g/rgouvea/VibroML/GPUMD/src/gpumd",
    os.path.expanduser("~/VibroML/GPUMD/src/gpumd"),
    os.path.expanduser("~/GPUMD/src/gpumd"),
    "/opt/gpumd/bin/gpumd",
    "/usr/local/bin/gpumd",
]
for path in gpumd_search_paths:
    if os.path.exists(path) and os.access(path, os.X_OK):
        GPUMD_BINARY_PATH = path
        HAVE_GPUMD = True
        break

# --- Utility Functions ---
def get_mace_device():
    if HAVE_MACE:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return None

# --- Custom GPUMD Calculator ---
class GPUMDCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, gpumd_binary, potential_path):
        Calculator.__init__(self)
        self.gpumd_binary = gpumd_binary
        self.potential_path = potential_path

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        work_dir = tempfile.mkdtemp(prefix="gpumd_calc_")
        try:
            write_gpumd_model_xyz(self.atoms, os.path.join(work_dir, "model.xyz"))
            shutil.copy(self.potential_path, os.path.join(work_dir, "nep.txt"))
            with open(os.path.join(work_dir, "run.in"), 'w') as f:
                f.write("potential nep.txt\nensemble nve\ntime_step 0\ndump_force 1\ndump_thermo 1\nrun 1\n")
            
            result = subprocess.run([self.gpumd_binary], cwd=work_dir, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"GPUMD execution failed: {result.stderr}")

            energy, forces, stress = parse_gpumd_output_files(self.atoms, work_dir)
            self.results['energy'] = energy
            self.results['forces'] = forces
            if stress is not None:
                self.results['stress'] = stress
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

def write_gpumd_model_xyz(atoms, filepath):
    with open(filepath, 'w') as f:
        f.write(f"{len(atoms)}\n")
        lattice_str = " ".join([f"{x:.10f}" for x in atoms.get_cell().flatten()])
        f.write(f'pbc="T T T" Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:mass:R:1\n')
        for symbol, pos, mass in zip(atoms.get_chemical_symbols(), atoms.get_positions(), atoms.get_masses()):
            f.write(f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f} {mass:.10f}\n")

def parse_gpumd_output_files(atoms, work_dir):
    forces = []
    with open(os.path.join(work_dir, "force.out"), 'r') as f:
        for line in f:
            if line.strip(): forces.append([float(x) for x in line.split()[:3]])
    forces = np.array(forces)[-len(atoms):]

    energy, stress = 0.0, None
    thermo_file = os.path.join(work_dir, "thermo.out")
    if os.path.exists(thermo_file):
        with open(thermo_file, 'r') as f:
            lines = f.readlines()
        for line in reversed(lines):
            parts = line.split()
            if len(parts) >= 3:
                try: energy = float(parts[2])
                except ValueError: continue
                if len(parts) >= 9:
                    try: stress = -np.array([float(x) for x in parts[3:9]]) / 160.21766208
                    except ValueError: pass
                break
    return energy, forces, stress

# --- Relaxation Utilities ---
class EnergyVolumeStopper:
    def __init__(self, optimizer, energy_increase_threshold=0.5, energy_decrease_threshold=-5.0, volume_threshold=2.5, max_steps=1000, min_iterations=5):
        self.optimizer = optimizer
        self.energy_increase_threshold = energy_increase_threshold
        self.energy_decrease_threshold = energy_decrease_threshold
        self.volume_threshold = volume_threshold
        self.max_steps = max_steps
        self.min_iterations = min_iterations
        self.initial_energy_per_atom = None
        self.initial_volume = None
        self.step_count = 0
          
    def __call__(self):  
        self.step_count += 1  
        atoms = self.optimizer.atoms.atoms if isinstance(self.optimizer.atoms, UnitCellFilter) else self.optimizer.atoms
          
        try:
            current_energy = atoms.get_potential_energy()
            if current_energy is None: return
            current_energy_per_atom = current_energy / len(atoms)
            current_volume = atoms.get_volume()
            if current_volume is None: return

            if self.initial_energy_per_atom is None and self.step_count > self.min_iterations:
                self.initial_energy_per_atom = current_energy_per_atom
                self.initial_volume = current_volume
                return

            if self.initial_energy_per_atom is None or self.step_count <= self.min_iterations: return

            energy_change = current_energy_per_atom - self.initial_energy_per_atom
            
            if energy_change > self.energy_increase_threshold:
                raise StopIteration
            if energy_change < self.energy_decrease_threshold:
                raise StopIteration
            if current_volume / self.initial_volume > self.volume_threshold:
                raise StopIteration
            if self.step_count >= self.max_steps:  
                raise StopIteration  
                  
        except StopIteration:  
            raise  

def relax_with_gpumd_native(atoms, nep_path, gpumd_binary, fmax, max_steps, output_dir):
    gpumd_dir = os.path.join(output_dir, 'gpumd_native_relax')
    os.makedirs(gpumd_dir, exist_ok=True)
    try:
        write_gpumd_model_xyz(atoms, os.path.join(gpumd_dir, 'model.xyz'))
        with open(os.path.join(gpumd_dir, 'run.in'), 'w') as f:
            f.write(f"potential {nep_path}\nminimize fire {fmax} {max_steps} 1 0\nensemble nve\ntime_step 0\ndump_thermo 1\ndump_xyz -1 0 1 relaxed.xyz\nrun 1\n")
        
        subprocess.run([gpumd_binary], cwd=gpumd_dir, capture_output=True, text=True, timeout=600)
        relaxed_xyz_path = os.path.join(gpumd_dir, 'relaxed.xyz')
        if not os.path.exists(relaxed_xyz_path): return None
        return read(relaxed_xyz_path)
    except Exception:
        return None

def relax_structure(atoms, calculator, engine, fmax, output_dir):
    """Relaxes the structure using the given calculator and engine."""
    os.makedirs(output_dir, exist_ok=True)
    atoms.set_calculator(calculator)
    relaxed_atoms = None

    if engine == "gpumd":
        nep_path = getattr(calculator, 'potential_path', None)
        if not nep_path: nep_path = "/globalscratch/ucl/modl/rgouvea/VibroML/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt"
        relaxed_atoms = relax_with_gpumd_native(atoms, nep_path, GPUMD_BINARY_PATH, fmax, 1000, output_dir)
    else:
        ucf = UnitCellFilter(atoms)
        opt = BFGS(ucf, logfile=os.path.join(output_dir, "relax.log"))

        def terminal_log():
            # This prints a summary every step and flushes the buffer
            step = opt.get_number_of_steps()
            energy = atoms.get_potential_energy() / len(atoms)
            print(f"      [Step {step}] E/atom: {energy:.6f} eV", flush=True)
        opt.attach(terminal_log, interval=1)
        
        stopper = EnergyVolumeStopper(opt, max_steps=1000, min_iterations=5)
        opt.attach(stopper)
        
        try:
            opt.run(fmax=fmax)
            relaxed_atoms = ucf.atoms.copy()
        except StopIteration:
            print("    [Relaxation] Stopped early due to Energy/Volume criteria or max steps.")
            relaxed_atoms = ucf.atoms.copy()
        except Exception as e:
            print(f"    [Relaxation Error] Optimization failed completely: {e}")
            relaxed_atoms = None

    # If we got a structure out, verify it and return it
    if relaxed_atoms is not None:
        try:
            # FIX 1: Re-attach the calculator before requesting forces!
            relaxed_atoms.calc = calculator
            
            # FIX 2: Do not throw away the structure if it didn't perfectly converge
            forces = relaxed_atoms.get_forces()
            max_force = np.sqrt((forces**2).sum(axis=1)).max()
            
            if max_force <= 2 * fmax:
                print(f"    [Relaxation] Converged beautifully (Max force: {max_force:.4f} eV/Å)")
            else:
                print(f"    [Relaxation] Partial convergence (Max force: {max_force:.4f} eV/Å). Keeping structure anyway.")
                
            return relaxed_atoms
            
        except Exception as e:
            print(f"    [WARNING] Could not verify final forces, but returning structure anyway: {e}")
            return relaxed_atoms
            
    return None 

# --- Main Calculator Initializer ---
def initialize_calculator(engine, model_name="medium-omat-0", checkpoint_path=None, nep_model_path=None, checkpoint_model_path=None):
    if checkpoint_path is not None:
        if engine in ("gpumd", "nep", "calorine") and nep_model_path is None:
            nep_model_path = checkpoint_path
        elif engine in ("esen", "uma") and checkpoint_model_path is None:
            checkpoint_model_path = checkpoint_path

    calculator = None
    if engine == "m3gnet":
        if not HAVE_M3GNET: sys.exit("[ERROR] M3GNet not found.")
        potential = Potential(M3GNet.load())
        calculator = M3GNetCalculator(potential=potential, stress_weight=0.01)
    elif engine == "mace":
        if not HAVE_MACE: sys.exit("[ERROR] MACE not found.")
        calculator = mace_mp(model=model_name, dispersion=False, default_dtype="float64", device=get_mace_device(), stress=True)
    elif engine == "esen":
        if not HAVE_ESEN: sys.exit("[ERROR] eSEN not found.")
        import torch
        calculator = OCPCalculator(checkpoint_path=checkpoint_model_path, cpu=not torch.cuda.is_available())
    elif engine == "uma":
        if not HAVE_UMA: sys.exit("[ERROR] UMA not found.")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        calculator = FAIRChemCalculator(predict_unit=load_predict_unit(checkpoint_model_path, device=device), task_name="omat")
    elif engine in ("nep", "calorine"):
        if not HAVE_CALORINE or CPUNEP is None: sys.exit("[ERROR] calorine not found.")
        calculator = CPUNEP(nep_model_path)
    elif engine == "gpumd":
        if not HAVE_GPUMD: sys.exit("[ERROR] GPUMD binary not found.")
        calculator = GPUMDCalculator(GPUMD_BINARY_PATH, nep_model_path)
    else:
        print(f"[ERROR] Engine '{engine}' not supported.")
        return None

    return calculator