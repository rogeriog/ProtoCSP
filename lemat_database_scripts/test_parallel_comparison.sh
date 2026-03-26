#!/bin/bash
#
# Test script for parallel dataset comparison
# Tests the parallel workflow with a small subset

echo "=========================================="
echo "Testing Parallel Dataset Comparison"
echo "=========================================="
echo ""

# Test 1: Check script files
echo "Test 1: Checking script files..."
for script in compare_datasets.py merge_comparison_results.py submit_compare_datasets.sh; do
    if [ -f "$script" ]; then
        echo "  ✓ $script exists"
    else
        echo "  ✗ $script not found"
        exit 1
    fi
    
    if [ -x "$script" ]; then
        echo "  ✓ $script is executable"
    else
        echo "  ✗ $script is not executable"
        exit 1
    fi
done
echo ""

# Test 2: Check LeMat metadata
echo "Test 2: Checking LeMat metadata..."
if [ -f "lemat_metadata.csv" ]; then
    echo "  ✓ lemat_metadata.csv exists"
    num_rows=$(wc -l < lemat_metadata.csv)
    echo "  ✓ Contains $num_rows rows (including header)"
else
    echo "  ✗ lemat_metadata.csv not found"
    exit 1
fi
echo ""

# Test 3: Create test datasets file
echo "Test 3: Creating test datasets file..."
cat > test_datasets_parallel.txt << 'EOF'
From the base path: /gpfs/scratch/acad/htforft/rgouvea
./matbench_tests/data/matbench_perovskites/matbench_perovskites_featurizedMM2020Struct_mattervial.csv
./matbench_tests/data/matbench_steels/matbench_steels_featurizedMM2020Comp_mattervial.csv
./matbench_tests/data/matbench_glass/matbench_glass_featurizedMM2020_mattervial.csv
EOF

echo "  ✓ Created test_datasets_parallel.txt with 3 datasets"
echo ""

# Test 4: Test single dataset processing (index 0)
echo "Test 4: Testing single dataset processing (index 0)..."
echo ""

rm -rf test_parallel_results
mkdir -p test_parallel_results

python3 compare_datasets.py \
    --lemat-metadata ./lemat_metadata.csv \
    --datasets-file ./test_datasets_parallel.txt \
    --base-path /gpfs/scratch/acad/htforft/rgouvea \
    --output-dir ./test_parallel_results \
    --dataset-index 0 \
    --save-matches \
    --verbose

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "  ✓ Dataset 0 processed successfully"
else
    echo "  ✗ Dataset 0 failed with exit code $exit_code"
    exit 1
fi
echo ""

# Test 5: Verify partial results
echo "Test 5: Verifying partial results..."

if [ -f "test_parallel_results/partial_summary_000.csv" ]; then
    echo "  ✓ partial_summary_000.csv created"
    cat test_parallel_results/partial_summary_000.csv
else
    echo "  ✗ partial_summary_000.csv not found"
    exit 1
fi
echo ""

# Test 6: Process remaining datasets
echo "Test 6: Processing remaining datasets (indices 1 and 2)..."
echo ""

for idx in 1 2; do
    echo "Processing dataset $idx..."
    python3 compare_datasets.py \
        --lemat-metadata ./lemat_metadata.csv \
        --datasets-file ./test_datasets_parallel.txt \
        --base-path /gpfs/scratch/acad/htforft/rgouvea \
        --output-dir ./test_parallel_results \
        --dataset-index $idx \
        --save-matches \
        > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Dataset $idx processed successfully"
    else
        echo "  ✗ Dataset $idx failed"
        exit 1
    fi
done
echo ""

# Test 7: Verify all partial files
echo "Test 7: Verifying all partial files..."

num_partial=$(ls -1 test_parallel_results/partial_summary_*.csv 2>/dev/null | wc -l)
echo "  Found $num_partial partial summary files"

if [ $num_partial -eq 3 ]; then
    echo "  ✓ All 3 partial files created"
else
    echo "  ✗ Expected 3 partial files, found $num_partial"
    exit 1
fi

for f in test_parallel_results/partial_summary_*.csv; do
    echo "    - $(basename $f)"
done
echo ""

# Test 8: Test merge script
echo "Test 8: Testing merge script..."
echo ""

python3 merge_comparison_results.py \
    --input-dir ./test_parallel_results \
    --output-dir ./test_parallel_results \
    --lemat-metadata ./lemat_metadata.csv \
    --verbose

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "  ✓ Merge completed successfully"
else
    echo "  ✗ Merge failed with exit code $exit_code"
    exit 1
fi
echo ""

# Test 9: Verify final output files
echo "Test 9: Verifying final output files..."

if [ -f "test_parallel_results/comparison_summary.csv" ]; then
    echo "  ✓ comparison_summary.csv created"
    num_rows=$(wc -l < test_parallel_results/comparison_summary.csv)
    echo "    Contains $num_rows rows (including header)"
else
    echo "  ✗ comparison_summary.csv not found"
    exit 1
fi

if [ -f "test_parallel_results/comparison_report.md" ]; then
    echo "  ✓ comparison_report.md created"
    num_lines=$(wc -l < test_parallel_results/comparison_report.md)
    echo "    Contains $num_lines lines"
else
    echo "  ✗ comparison_report.md not found"
    exit 1
fi

# Check that partial files were cleaned up
num_partial_after=$(ls -1 test_parallel_results/partial_summary_*.csv 2>/dev/null | wc -l)
if [ $num_partial_after -eq 0 ]; then
    echo "  ✓ Partial files cleaned up"
else
    echo "  ⚠ Warning: $num_partial_after partial files still present"
fi
echo ""

# Test 10: Display final results
echo "Test 10: Displaying final results..."
echo ""

echo "Final summary CSV:"
cat test_parallel_results/comparison_summary.csv
echo ""

echo "First 40 lines of report:"
head -40 test_parallel_results/comparison_report.md
echo ""

# Cleanup
echo "Cleanup: Removing test files..."
rm -f test_datasets_parallel.txt
rm -rf test_parallel_results
echo "  ✓ Test files removed"
echo ""

# Summary
echo "=========================================="
echo "All Tests Passed! ✓"
echo "=========================================="
echo ""
echo "The parallel dataset comparison workflow is working correctly!"
echo ""
echo "To compare all 13 datasets in parallel:"
echo "  1. Submit SLURM job array:"
echo "     sbatch submit_compare_datasets.sh"
echo ""
echo "  2. Wait for all tasks to complete:"
echo "     squeue -u \$USER"
echo ""
echo "  3. Merge results:"
echo "     python3 merge_comparison_results.py"
echo ""
echo "Expected wall time: ~1-2 minutes (parallel) vs ~5-10 minutes (sequential)"
echo ""

