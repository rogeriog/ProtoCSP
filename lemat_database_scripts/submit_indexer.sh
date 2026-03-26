#!/bin/bash
#
# SLURM Job Array submission script for parallel structure processing.
# This script processes all 500 LeMat CSV files using job arrays.
# Each task processes a chunk of the 500 part files.

# Also, it runs reordered db to generate json files

# ---------------- SLURM DIRECTIVES ----------------

#SBATCH --job-name=lemat_indexer    # Job name
#SBATCH --output=%A_%a_lemat_indexer.out  # Output file (%A=Job ID, %a=Array ID)
#SBATCH --array=0                        # Defines the array size: Task IDs 0-4 (5 jobs total)
#SBATCH --partition=debug                 # Use the 'shared' partition (adjust as needed)
#SBATCH --nodes=1                          # Request one node per task
#SBATCH --ntasks-per-node=1                # Serial job per task
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G                          # Request 32GB memory per task (increased for structure processing)
#SBATCH --time=2:00:00                     # Max wall time per task (3 hours - adjust based on testing)
#SBATCH --account=htforft                  # Replace with your project/account name

# ---------------- SETUP ----------------

echo "=========================================="
echo "Job Array Task ${SLURM_ARRAY_TASK_ID} started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"
echo "=========================================="

# Load necessary modules
module purge
module load EasyBuild/2024a OpenMPI/5.0.3-GCC-13.3.0 FlexiBLAS/3.4.4-GCC-13.3.0

# Activate your conda environment 
source /gpfs/home/acad/ucl-modl/rgouvea/miniconda3/etc/profile.d/conda.sh


# ---------------- EXECUTION ----------------
# Execute the Python script, passing the array task ID as the argument.
# The Python script calculates its specific part file range internally.
# python indexer.py --csv-dir ../lemat_unique_csv_500_parts --output lemat_indexed_library --workers 32
python reorganize_db.py --input-prefix lemat_indexed_library --output-dir lemat_formula_indexed --workers 32
EXIT_CODE=$?

# ---------------- CLEANUP ----------------

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job Array Task ${SLURM_ARRAY_TASK_ID} finished SUCCESSFULLY on $(date)"
else
    echo "Job Array Task ${SLURM_ARRAY_TASK_ID} finished with ERROR (exit code: $EXIT_CODE) on $(date)"
fi
echo "=========================================="

exit $EXIT_CODE

