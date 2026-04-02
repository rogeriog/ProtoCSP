#!/bin/bash
#SBATCH --job-name=protocsp_test
#SBATCH --time=1:00:00
#SBATCH --output=log_test_protocsp.txt
#SBATCH --partition=debug-gpu 
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --account=htforft

# 1. Clean Environment to prevent NumPy path pollution
module purge

source ~/.bashrc

module load EasyBuild/2022a CUDA/11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0

conda activate /gpfs/scratch/acad/htforft/rgouvea/vibroml_runs/vibroml_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
# Keep this to prevent deadlock
export MP_START_METHOD=spawn

echo "start"
date

# python3 test/test_comprehensive.py
python3 test/test_surface_alloy.py
echo "done"
date

