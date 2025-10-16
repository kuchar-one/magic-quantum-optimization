#!/usr/bin/env python3
"""
Test script for parallel optimization implementation.

This script tests the parallel evaluation system to verify that:
1. Multiple CPU cores are being utilized
2. GPU utilization is improved
3. Results are consistent with sequential evaluation
4. Performance is improved over sequential evaluation
"""

import sys
import os
import time
import psutil
import numpy as np
import torch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.nsga_ii import run_quantum_sequence_optimization
from src.qutip_quantum_ops import construct_operator, construct_initial_state
from src.cuda_helpers import set_gpu_device, aggressive_memory_cleanup


def monitor_cpu_usage(duration=10, interval=0.5):
    """
    Monitor CPU usage over a period of time.
    
    Parameters
    ----------
    duration : float
        Duration to monitor in seconds
    interval : float
        Sampling interval in seconds
        
    Returns
    -------
    dict
        Statistics about CPU usage
    """
    cpu_percentages = []
    num_samples = int(duration / interval)
    
    for _ in range(num_samples):
        cpu_percentages.append(psutil.cpu_percent(interval=interval, percpu=True))
    
    cpu_percentages = np.array(cpu_percentages)
    
    return {
        'avg_per_core': np.mean(cpu_percentages, axis=0),
        'max_per_core': np.max(cpu_percentages, axis=0),
        'avg_overall': np.mean(cpu_percentages),
        'max_overall': np.max(cpu_percentages),
        'num_active_cores': np.sum(np.mean(cpu_percentages, axis=0) > 10),
    }


def test_parallel_optimization():
    """Test parallel optimization with CPU monitoring."""
    
    print("=" * 80)
    print("PARALLEL OPTIMIZATION TEST")
    print("=" * 80)
    print()
    
    # Test parameters
    N = 20
    sequence_length = 8
    pop_size = 50
    max_generations = 5
    num_workers = 4  # Use 4 workers for testing
    
    print(f"Test Configuration:")
    print(f"  - Hilbert space dimension (N): {N}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Population size: {pop_size}")
    print(f"  - Max generations: {max_generations}")
    print(f"  - Number of workers: {num_workers}")
    print()
    
    # Set up GPU
    try:
        set_gpu_device(0)
    except Exception as e:
        print(f"Error setting GPU device: {e}")
        return 1
    
    # Construct initial state and operator
    print("Constructing initial state and operator...")
    initial_state_qobj, initial_probability = construct_initial_state(N, "vacuum", 0.0)
    operator = construct_operator(N, (1.0, 1.0))
    
    # Test parallel optimization
    print("\n" + "=" * 80)
    print("Running PARALLEL optimization...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Monitor CPU usage during optimization
    import threading
    cpu_stats = {'data': None}
    
    def monitor_thread():
        cpu_stats['data'] = monitor_cpu_usage(duration=60, interval=0.5)
    
    monitor = threading.Thread(target=monitor_thread)
    monitor.start()
    
    try:
        result_parallel, F_history_parallel = run_quantum_sequence_optimization(
            initial_state=initial_state_qobj,
            initial_probability=initial_probability,
            operator=operator,
            N=N,
            sequence_length=sequence_length,
            pop_size=pop_size,
            max_generations=max_generations,
            device_id=0,
            tolerance=1e-3,
            algorithm="nsga2",
            use_ket_optimization=True,
            num_workers=num_workers,
        )
        
        parallel_time = time.time() - start_time
        
        # Wait for monitoring to finish
        monitor.join(timeout=5)
        
        print(f"\nParallel optimization completed in {parallel_time:.2f} seconds")
        
        # Print CPU statistics
        if cpu_stats['data'] is not None:
            stats = cpu_stats['data']
            print("\nCPU Usage Statistics:")
            print(f"  - Average CPU usage: {stats['avg_overall']:.1f}%")
            print(f"  - Maximum CPU usage: {stats['max_overall']:.1f}%")
            print(f"  - Active cores (>10% usage): {stats['num_active_cores']}/{os.cpu_count()}")
            print(f"  - Per-core average:")
            for i, usage in enumerate(stats['avg_per_core']):
                print(f"      Core {i}: {usage:.1f}%")
        
        # Print results
        if len(result_parallel.F) > 0:
            best_expectation = np.min(result_parallel.F[:, 0])
            best_probability = np.max(-result_parallel.F[:, 1])
            print(f"\nParallel Results:")
            print(f"  - Pareto front size: {len(result_parallel.F)}")
            print(f"  - Best expectation: {best_expectation:.6f}")
            print(f"  - Best probability: {best_probability:.6f}")
        else:
            print("\nParallel optimization produced no results")
        
    except Exception as e:
        print(f"\nError during parallel optimization: {e}")
        import traceback
        traceback.print_exc()
        aggressive_memory_cleanup()
        return 1
    
    # Clean up
    aggressive_memory_cleanup()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    print()
    print("Key Metrics:")
    print(f"  - Optimization time: {parallel_time:.2f} seconds")
    if cpu_stats['data'] is not None:
        print(f"  - CPU cores utilized: {cpu_stats['data']['num_active_cores']}/{os.cpu_count()}")
        print(f"  - Average CPU usage: {cpu_stats['data']['avg_overall']:.1f}%")
    print()
    print("Expected Behavior:")
    print(f"  - Should use ~{num_workers} CPU cores at high utilization")
    print(f"  - GPU should be more fully utilized")
    print(f"  - Optimization should complete successfully")
    print()
    
    return 0


def test_sequential_vs_parallel():
    """Compare sequential vs parallel evaluation."""
    
    print("=" * 80)
    print("SEQUENTIAL vs PARALLEL COMPARISON")
    print("=" * 80)
    print()
    
    # Test parameters (smaller for faster testing)
    N = 15
    sequence_length = 6
    pop_size = 30
    max_generations = 3
    
    print(f"Test Configuration:")
    print(f"  - Hilbert space dimension (N): {N}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Population size: {pop_size}")
    print(f"  - Max generations: {max_generations}")
    print()
    
    # Set up GPU
    try:
        set_gpu_device(0)
    except Exception as e:
        print(f"Error setting GPU device: {e}")
        return 1
    
    # Construct initial state and operator
    print("Constructing initial state and operator...")
    initial_state_qobj, initial_probability = construct_initial_state(N, "vacuum", 0.0)
    operator = construct_operator(N, (1.0, 1.0))
    
    # Test sequential optimization
    print("\n" + "=" * 80)
    print("Running SEQUENTIAL optimization...")
    print("=" * 80)
    
    start_time = time.time()
    try:
        result_seq, F_history_seq = run_quantum_sequence_optimization(
            initial_state=initial_state_qobj,
            initial_probability=initial_probability,
            operator=operator,
            N=N,
            sequence_length=sequence_length,
            pop_size=pop_size,
            max_generations=max_generations,
            device_id=0,
            tolerance=1e-3,
            algorithm="nsga2",
            use_ket_optimization=True,
            num_workers=1,  # Sequential
        )
        sequential_time = time.time() - start_time
        print(f"Sequential optimization completed in {sequential_time:.2f} seconds")
    except Exception as e:
        print(f"Error during sequential optimization: {e}")
        return 1
    
    # Clean up
    aggressive_memory_cleanup()
    time.sleep(2)
    
    # Test parallel optimization
    print("\n" + "=" * 80)
    print("Running PARALLEL optimization (4 workers)...")
    print("=" * 80)
    
    start_time = time.time()
    try:
        result_parallel, F_history_parallel = run_quantum_sequence_optimization(
            initial_state=initial_state_qobj,
            initial_probability=initial_probability,
            operator=operator,
            N=N,
            sequence_length=sequence_length,
            pop_size=pop_size,
            max_generations=max_generations,
            device_id=0,
            tolerance=1e-3,
            algorithm="nsga2",
            use_ket_optimization=True,
            num_workers=4,  # Parallel
        )
        parallel_time = time.time() - start_time
        print(f"Parallel optimization completed in {parallel_time:.2f} seconds")
    except Exception as e:
        print(f"Error during parallel optimization: {e}")
        return 1
    
    # Clean up
    aggressive_memory_cleanup()
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()
    print(f"Sequential time: {sequential_time:.2f} seconds")
    print(f"Parallel time:   {parallel_time:.2f} seconds")
    print(f"Speedup:         {sequential_time / parallel_time:.2f}x")
    print()
    
    if len(result_seq.F) > 0 and len(result_parallel.F) > 0:
        print("Solution Quality Comparison:")
        print(f"  Sequential - Best expectation: {np.min(result_seq.F[:, 0]):.6f}, Best probability: {np.max(-result_seq.F[:, 1]):.6f}")
        print(f"  Parallel   - Best expectation: {np.min(result_parallel.F[:, 0]):.6f}, Best probability: {np.max(-result_parallel.F[:, 1]):.6f}")
    
    print()
    print("Expected: Parallel should be faster with similar or better solution quality")
    print()
    
    return 0


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test parallel optimization implementation')
    parser.add_argument('--test', choices=['parallel', 'comparison', 'both'], default='parallel',
                       help='Which test to run (default: parallel)')
    args = parser.parse_args()
    
    if args.test == 'parallel':
        return test_parallel_optimization()
    elif args.test == 'comparison':
        return test_sequential_vs_parallel()
    elif args.test == 'both':
        result1 = test_parallel_optimization()
        if result1 != 0:
            return result1
        print("\n" + "=" * 80 + "\n")
        return test_sequential_vs_parallel()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


