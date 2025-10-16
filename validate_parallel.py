#!/usr/bin/env python3
"""
Quick validation script to check if parallel evaluation module is properly set up.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("Validating parallel evaluation implementation...")
print()

# Test 1: Import parallel evaluation module
print("Test 1: Importing parallel_evaluation module...")
try:
    from src.parallel_evaluation import ParallelEvaluator
    print("✓ Successfully imported ParallelEvaluator")
except Exception as e:
    print(f"✗ Failed to import ParallelEvaluator: {e}")
    sys.exit(1)

# Test 2: Check if required dependencies are available
print("\nTest 2: Checking dependencies...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available (will run on CPU)")
    
    import torch.multiprocessing as mp
    print("✓ torch.multiprocessing available")
    
    import psutil
    print(f"✓ psutil available (CPU count: {psutil.cpu_count()})")
    
except Exception as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

# Test 3: Check if nsga_ii module has parallel support
print("\nTest 3: Checking nsga_ii module...")
try:
    from src.nsga_ii import run_quantum_sequence_optimization
    print("✓ Successfully imported run_quantum_sequence_optimization")
    
    # Check if function signature includes num_workers parameter
    import inspect
    sig = inspect.signature(run_quantum_sequence_optimization)
    if 'num_workers' in sig.parameters:
        print("✓ run_quantum_sequence_optimization supports num_workers parameter")
    else:
        print("✗ run_quantum_sequence_optimization missing num_workers parameter")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Failed to import or validate nsga_ii: {e}")
    sys.exit(1)

# Test 4: Check if quo.py supports num_workers
print("\nTest 4: Checking quo.py...")
try:
    # Read the file and check if it has num_workers argument
    with open('quo.py', 'r') as f:
        content = f.read()
        if '--num_workers' in content:
            print("✓ quo.py supports --num_workers argument")
        else:
            print("✗ quo.py missing --num_workers argument")
            sys.exit(1)
except Exception as e:
    print(f"✗ Failed to check quo.py: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
print()
print("All basic checks passed! ✓")
print()
print("The parallel evaluation system is properly integrated.")
print("You can now run the full test with:")
print("  python test_parallel_optimization.py --test parallel")
print()


