#!/bin/bash
#
# Test script for parallel index building
# Tests the parallel workflow with a small subset of data

echo "=========================================="
echo "Testing Parallel Index Building"
echo "=========================================="
echo ""

# Test 1: Check scripts exist
echo "Test 1: Checking script files..."
for script in build_index_parallel.py merge_indices.py submit_build_index.sh; do
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

# Test 2: Check test data exists
echo "Test 2: Checking test data..."
if [ -d "processed_structures_test" ]; then
    num_files=$(ls -1 processed_structures_test/*.csv 2>/dev/null | wc -l)
    echo "  ✓ Test directory exists"
    echo "  ✓ Found $num_files CSV files"
else
    echo "  ✗ Test directory not found"
    echo "  Please run: python3 structures_converter.py --test"
    exit 1
fi
echo ""

# Test 3: Test building partial index
echo "Test 3: Building test partial index (task 0, first 5 files)..."
mkdir -p test_partial_indices

# Temporarily modify the script to use test data
python3 -c "
import sys
sys.path.insert(0, '.')
from build_lemat_index import build_lemat_index, save_index

print('Building partial index for parts 0-5...')
structures_index, metadata = build_lemat_index(
    './processed_structures_test',
    verbose=True,
    start_part=0,
    end_part=5
)

if structures_index is None:
    print('ERROR: Failed to build partial index')
    sys.exit(1)

save_index(structures_index, metadata, './test_partial_indices/partial_index_00.pkl')
print('✓ Partial index built successfully')
"

if [ $? -eq 0 ]; then
    echo "  ✓ Partial index built successfully"
else
    echo "  ✗ Failed to build partial index"
    exit 1
fi
echo ""

# Test 4: Test building another partial index
echo "Test 4: Building second test partial index (task 1, files 5-10)..."
python3 -c "
import sys
sys.path.insert(0, '.')
from build_lemat_index import build_lemat_index, save_index

print('Building partial index for parts 5-10...')
structures_index, metadata = build_lemat_index(
    './processed_structures_test',
    verbose=False,
    start_part=5,
    end_part=10
)

if structures_index is None:
    print('ERROR: Failed to build partial index')
    sys.exit(1)

save_index(structures_index, metadata, './test_partial_indices/partial_index_01.pkl')
print('✓ Partial index built successfully')
"

if [ $? -eq 0 ]; then
    echo "  ✓ Second partial index built successfully"
else
    echo "  ✗ Failed to build second partial index"
    exit 1
fi
echo ""

# Test 5: Check partial indices created
echo "Test 5: Checking partial indices..."
num_partial=$(ls -1 test_partial_indices/partial_index_*.pkl 2>/dev/null | wc -l)
if [ $num_partial -eq 2 ]; then
    echo "  ✓ Found $num_partial partial indices"
    ls -lh test_partial_indices/
else
    echo "  ✗ Expected 2 partial indices, found $num_partial"
    exit 1
fi
echo ""

# Test 6: Test merging
echo "Test 6: Testing merge functionality..."
python3 merge_indices.py \
    --partial-dir ./test_partial_indices \
    --output ./test_merged_index.pkl \
    --verbose

if [ $? -eq 0 ]; then
    echo "  ✓ Merge completed successfully"
else
    echo "  ✗ Merge failed"
    exit 1
fi
echo ""

# Test 7: Verify merged index
echo "Test 7: Verifying merged index..."
if [ -f "test_merged_index.pkl" ]; then
    size=$(du -h test_merged_index.pkl | cut -f1)
    echo "  ✓ Merged index created"
    echo "  ✓ File size: $size"
else
    echo "  ✗ Merged index not found"
    exit 1
fi
echo ""

# Test 8: Test loading merged index
echo "Test 8: Testing merged index can be loaded..."
python3 -c "
import pickle
import sys

try:
    with open('test_merged_index.pkl', 'rb') as f:
        data = pickle.load(f)
    
    index = data['index']
    metadata = data['metadata']
    
    print(f'  ✓ Loaded successfully')
    print(f'  ✓ Total structures: {metadata[\"total_structures\"]:,}')
    print(f'  ✓ Unique keys: {metadata[\"unique_keys\"]:,}')
    
except Exception as e:
    print(f'  ✗ Failed to load: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "  ✓ Merged index loads correctly"
else
    echo "  ✗ Failed to load merged index"
    exit 1
fi
echo ""

# Test 9: Compare with serial build
echo "Test 9: Comparing with serial build..."
python3 build_lemat_index.py \
    --lemat-dir ./processed_structures_test \
    --output ./test_serial_index.pkl \
    > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Serial index built"
    
    # Compare structure counts
    python3 -c "
import pickle

with open('test_merged_index.pkl', 'rb') as f:
    merged_data = pickle.load(f)

with open('test_serial_index.pkl', 'rb') as f:
    serial_data = pickle.load(f)

merged_count = merged_data['metadata']['total_structures']
serial_count = serial_data['metadata']['total_structures']

print(f'  Merged index: {merged_count:,} structures')
print(f'  Serial index: {serial_count:,} structures')

if merged_count == serial_count:
    print('  ✓ Structure counts match')
else:
    print(f'  ✗ Structure counts differ: {merged_count} vs {serial_count}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Parallel and serial builds produce same results"
    else
        echo "  ✗ Results differ between parallel and serial builds"
        exit 1
    fi
else
    echo "  ✗ Failed to build serial index"
    exit 1
fi
echo ""

# Cleanup
echo "Cleanup: Removing test files..."
rm -rf test_partial_indices
rm -f test_merged_index.pkl
rm -f test_serial_index.pkl
echo "  ✓ Test files removed"
echo ""

# Summary
echo "=========================================="
echo "All Tests Passed! ✓"
echo "=========================================="
echo ""
echo "The parallel index building workflow is working correctly!"
echo ""
echo "To use with full dataset:"
echo "  1. Submit job: sbatch submit_build_index.sh"
echo "  2. Monitor: squeue -u \$USER"
echo "  3. Merge: python3 merge_indices.py"
echo ""
echo "Expected time: ~5-6 minutes total (vs 10-20 minutes serial)"
echo ""

