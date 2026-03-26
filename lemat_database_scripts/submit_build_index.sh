#!/bin/bash
#SBATCH --job-name=build_index
#SBATCH --output=logs/build_index_%A_%a.out
#SBATCH --error=logs/build_index_%A_%a.err
#SBATCH --array=0-9
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=shared
#SBATCH --account=htforft

# Parallel Index Building Script
# This script launches 10 parallel tasks to build partial indices
# Each task processes 50 CSV files (500 total / 10 tasks)

echo "=========================================="
echo "Parallel Index Building - Task ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Change to working directory
cd /gpfs/scratch/acad/htforft/rgouvea/lemat_unique_dataset

# Load conda environment
source /gpfs/home/acad/ucl-modl/rgouvea/miniconda3/etc/profile.d/conda.sh

# Create logs directory if it doesn't exist
mkdir -p logs

# Create partial_indices directory if it doesn't exist
mkdir -p partial_indices

# Run the parallel index building script
echo "Running build_index_parallel.py for task ${SLURM_ARRAY_TASK_ID}..."
echo ""

python3 build_index_parallel.py ${SLURM_ARRAY_TASK_ID}

exit_code=$?

echo ""
echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} Complete"
echo "=========================================="
echo "Exit code: ${exit_code}"
echo "End time: $(date)"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "✓ Task ${SLURM_ARRAY_TASK_ID} completed successfully"
    echo ""
    echo "After all tasks complete (check with 'squeue -u \$USER'),"
    echo "run the merge script:"
    echo "  python3 merge_indices.py"
else
    echo "✗ Task ${SLURM_ARRAY_TASK_ID} failed with exit code ${exit_code}"
fi

exit $exit_code

