#!/usr/bin/env python3
"""
Algorithm Benchmark Script for Magic Quantum Sequence Optimization

This script compares different optimization algorithms for quantum sequence optimization
in the magic project. It runs multiple algorithms with the same parameters and compares
their performance in terms of convergence speed, final solution quality, and robustness.

Author: AI Assistant
Date: 2024
"""

import argparse
import time
import json
import os
import sys
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.nsga_ii import run_quantum_sequence_optimization
from src.qutip_quantum_ops import construct_operator, construct_initial_state
from src.cuda_helpers import set_gpu_device, aggressive_memory_cleanup
import qutip as qt


class AlgorithmBenchmark:
    """Benchmark class for comparing optimization algorithms."""
    
    def __init__(self, N: int = 20, sequence_length: int = 10, pop_size: int = 100, 
                 max_generations: int = 100, target_superposition: Tuple[complex, ...] = (1.0, 1.0),
                 initial_state: str = "vacuum", gpu_id: int = 0, tolerance: float = 1e-3):
        """
        Initialize the benchmark with optimization parameters.
        
        Parameters
        ----------
        N : int
            Hilbert space dimension
        sequence_length : int
            Length of operation sequences
        pop_size : int
            Population size for optimization
        max_generations : int
            Maximum number of generations
        target_superposition : Tuple[complex, ...]
            Target superposition state coefficients
        initial_state : str
            Initial state type
        gpu_id : int
            GPU device ID
        tolerance : float
            Convergence tolerance
        """
        self.N = N
        self.sequence_length = sequence_length
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.target_superposition = target_superposition
        self.initial_state = initial_state
        self.gpu_id = gpu_id
        self.tolerance = tolerance
        
        # Available algorithms
        self.algorithms = [
            "nsga2", "nsga3", "moead", "age", "age2", 
            "rvea", "smsemoa", "ctaea", "unsga3", "rnsga2", "rnsga3"
        ]
        
        # Results storage
        self.results = {}
        
    def run_single_algorithm(self, algorithm: str, run_id: int = 0) -> Dict[str, Any]:
        """
        Run a single algorithm and return performance metrics.
        
        Parameters
        ----------
        algorithm : str
            Algorithm name to run
        run_id : int
            Run identifier for multiple runs
            
        Returns
        -------
        Dict[str, Any]
            Performance metrics and results
        """
        print(f"Running {algorithm} (run {run_id + 1})...")
        
        try:
            # Set up GPU
            set_gpu_device(self.gpu_id)
            
            # Construct initial state and operator
            initial_state_qobj, initial_probability = construct_initial_state(
                self.N, self.initial_state, 0.0
            )
            operator = construct_operator(self.N, self.target_superposition)
            
            # Record start time
            start_time = time.time()
            
            # Run optimization
            result, F_history = run_quantum_sequence_optimization(
                initial_state=initial_state_qobj,
                initial_probability=initial_probability,
                operator=operator,
                N=self.N,
                sequence_length=self.sequence_length,
                pop_size=self.pop_size,
                max_generations=self.max_generations,
                device_id=self.gpu_id,
                tolerance=self.tolerance,
                algorithm=algorithm,
                use_ket_optimization=True,
            )
            
            # Record end time
            end_time = time.time()
            
            # Calculate metrics
            F = result.F
            if len(F) == 0:
                return {
                    'algorithm': algorithm,
                    'run_id': run_id,
                    'success': False,
                    'error': 'No solutions found',
                    'runtime': end_time - start_time,
                    'final_generation': 0,
                    'pareto_size': 0,
                    'best_expectation': float('inf'),
                    'best_probability': 0.0,
                    'convergence_generation': 0,
                    'hypervolume': 0.0
                }
            
            # Extract metrics
            best_expectation = np.min(F[:, 0])
            best_probability = np.max(F[:, 1])
            pareto_size = len(F)
            
            # Calculate convergence generation (when best solution was found)
            convergence_generation = 0
            if len(F_history) > 0:
                best_expectation_history = [np.min(F_gen[:, 0]) for F_gen in F_history]
                convergence_generation = np.argmin(best_expectation_history)
            
            # Calculate hypervolume (approximate)
            hypervolume = self._calculate_hypervolume(F)
            
            # Clean up GPU memory
            aggressive_memory_cleanup()
            
            return {
                'algorithm': algorithm,
                'run_id': run_id,
                'success': True,
                'error': None,
                'runtime': end_time - start_time,
                'final_generation': len(F_history),
                'pareto_size': pareto_size,
                'best_expectation': float(best_expectation),
                'best_probability': float(best_probability),
                'convergence_generation': int(convergence_generation),
                'hypervolume': float(hypervolume),
                'F_history': [F_gen.tolist() for F_gen in F_history] if F_history else []
            }
            
        except Exception as e:
            print(f"Error running {algorithm}: {e}")
            aggressive_memory_cleanup()
            return {
                'algorithm': algorithm,
                'run_id': run_id,
                'success': False,
                'error': str(e),
                'runtime': 0.0,
                'final_generation': 0,
                'pareto_size': 0,
                'best_expectation': float('inf'),
                'best_probability': 0.0,
                'convergence_generation': 0,
                'hypervolume': 0.0
            }
    
    def _calculate_hypervolume(self, F: np.ndarray) -> float:
        """
        Calculate hypervolume metric for Pareto front.
        
        Parameters
        ----------
        F : np.ndarray
            Pareto front solutions
            
        Returns
        -------
        float
            Hypervolume value
        """
        if len(F) == 0:
            return 0.0
        
        # Reference point (worse than all solutions)
        ref_point = np.array([np.max(F[:, 0]) + 1, np.min(F[:, 1]) - 1])
        
        # Sort solutions by first objective
        sorted_F = F[np.argsort(F[:, 0])]
        
        # Calculate hypervolume using 2D method
        hypervolume = 0.0
        prev_x = ref_point[0]
        
        for point in sorted_F:
            hypervolume += (prev_x - point[0]) * (point[1] - ref_point[1])
            prev_x = point[0]
        
        return hypervolume
    
    def run_benchmark(self, num_runs: int = 3, algorithms: List[str] = None) -> Dict[str, Any]:
        """
        Run benchmark comparison across multiple algorithms and runs.
        
        Parameters
        ----------
        num_runs : int
            Number of runs per algorithm
        algorithms : List[str], optional
            List of algorithms to test. If None, uses all available algorithms.
            
        Returns
        -------
        Dict[str, Any]
            Complete benchmark results
        """
        if algorithms is None:
            algorithms = self.algorithms
        
        print(f"Starting benchmark with {len(algorithms)} algorithms, {num_runs} runs each")
        print(f"Parameters: N={self.N}, sequence_length={self.sequence_length}, pop_size={self.pop_size}")
        print(f"Target superposition: {self.target_superposition}")
        print("=" * 80)
        
        all_results = []
        
        for algorithm in algorithms:
            print(f"\nTesting algorithm: {algorithm}")
            print("-" * 40)
            
            algorithm_results = []
            for run_id in range(num_runs):
                result = self.run_single_algorithm(algorithm, run_id)
                algorithm_results.append(result)
                all_results.append(result)
                
                if result['success']:
                    print(f"  Run {run_id + 1}: {result['runtime']:.2f}s, "
                          f"expectation={result['best_expectation']:.6f}, "
                          f"probability={result['best_probability']:.6f}")
                else:
                    print(f"  Run {run_id + 1}: FAILED - {result['error']}")
            
            # Calculate statistics for this algorithm
            successful_runs = [r for r in algorithm_results if r['success']]
            if successful_runs:
                avg_runtime = np.mean([r['runtime'] for r in successful_runs])
                avg_expectation = np.mean([r['best_expectation'] for r in successful_runs])
                avg_probability = np.mean([r['best_probability'] for r in successful_runs])
                avg_hypervolume = np.mean([r['hypervolume'] for r in successful_runs])
                
                print(f"  Average: {avg_runtime:.2f}s, expectation={avg_expectation:.6f}, "
                      f"probability={avg_probability:.6f}, hypervolume={avg_hypervolume:.6f}")
        
        # Store results
        self.results = {
            'parameters': {
                'N': self.N,
                'sequence_length': self.sequence_length,
                'pop_size': self.pop_size,
                'max_generations': self.max_generations,
                'target_superposition': [str(c) for c in self.target_superposition],
                'initial_state': self.initial_state,
                'gpu_id': self.gpu_id,
                'tolerance': self.tolerance,
                'num_runs': num_runs,
                'algorithms_tested': algorithms
            },
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }
        
        return self.results
    
    def save_results(self, filename: str = None) -> str:
        """
        Save benchmark results to JSON file.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, generates timestamped filename.
            
        Returns
        -------
        str
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"magic_algorithm_benchmark_{timestamp}.json"
        
        # Ensure output directory exists
        os.makedirs("output/benchmarks", exist_ok=True)
        filepath = os.path.join("output/benchmarks", filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Returns
        -------
        str
            Formatted report string
        """
        if not self.results:
            return "No results available. Run benchmark first."
        
        # Group results by algorithm
        algorithm_stats = {}
        for result in self.results['results']:
            alg = result['algorithm']
            if alg not in algorithm_stats:
                algorithm_stats[alg] = []
            algorithm_stats[alg].append(result)
        
        # Calculate statistics
        report = []
        report.append("MAGIC QUANTUM SEQUENCE OPTIMIZATION - ALGORITHM BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.results['timestamp']}")
        report.append("")
        
        # Parameters
        params = self.results['parameters']
        report.append("BENCHMARK PARAMETERS:")
        report.append(f"  Hilbert Space Dimension (N): {params['N']}")
        report.append(f"  Sequence Length: {params['sequence_length']}")
        report.append(f"  Population Size: {params['pop_size']}")
        report.append(f"  Max Generations: {params['max_generations']}")
        report.append(f"  Target Superposition: {params['target_superposition']}")
        report.append(f"  Initial State: {params['initial_state']}")
        report.append(f"  Number of Runs per Algorithm: {params['num_runs']}")
        report.append("")
        
        # Algorithm comparison
        report.append("ALGORITHM COMPARISON:")
        report.append("-" * 80)
        report.append(f"{'Algorithm':<12} {'Success':<8} {'Avg Time':<10} {'Best Exp':<12} {'Best Prob':<12} {'Hypervol':<12}")
        report.append("-" * 80)
        
        for alg in sorted(algorithm_stats.keys()):
            results = algorithm_stats[alg]
            successful = [r for r in results if r['success']]
            success_rate = len(successful) / len(results) * 100
            
            if successful:
                avg_time = np.mean([r['runtime'] for r in successful])
                best_exp = np.min([r['best_expectation'] for r in successful])
                best_prob = np.max([r['best_probability'] for r in successful])
                avg_hypervol = np.mean([r['hypervolume'] for r in successful])
                
                report.append(f"{alg:<12} {success_rate:>6.1f}% {avg_time:>9.2f}s "
                            f"{best_exp:>11.6f} {best_prob:>11.6f} {avg_hypervol:>11.6f}")
            else:
                report.append(f"{alg:<12} {'0.0%':>8} {'N/A':>10} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
        
        report.append("")
        
        # Best performing algorithms
        successful_algs = {alg: results for alg, results in algorithm_stats.items() 
                          if any(r['success'] for r in results)}
        
        if successful_algs:
            # Best expectation value
            best_exp_alg = min(successful_algs.keys(), 
                             key=lambda alg: min(r['best_expectation'] for r in successful_algs[alg] if r['success']))
            best_exp_val = min(r['best_expectation'] for r in successful_algs[best_exp_alg] if r['success'])
            
            # Best probability
            best_prob_alg = max(successful_algs.keys(),
                              key=lambda alg: max(r['best_probability'] for r in successful_algs[alg] if r['success']))
            best_prob_val = max(r['best_probability'] for r in successful_algs[best_prob_alg] if r['success'])
            
            # Fastest convergence
            fastest_alg = min(successful_algs.keys(),
                            key=lambda alg: np.mean([r['runtime'] for r in successful_algs[alg] if r['success']]))
            fastest_time = np.mean([r['runtime'] for r in successful_algs[fastest_alg] if r['success']])
            
            report.append("BEST PERFORMING ALGORITHMS:")
            report.append(f"  Best Expectation Value: {best_exp_alg} ({best_exp_val:.6f})")
            report.append(f"  Best Probability: {best_prob_alg} ({best_prob_val:.6f})")
            report.append(f"  Fastest Convergence: {fastest_alg} ({fastest_time:.2f}s)")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if successful_algs:
            report.append("  - For best solution quality: Use algorithms with lowest expectation values")
            report.append("  - For fastest results: Use algorithms with shortest runtime")
            report.append("  - For robust optimization: Use algorithms with high success rates")
        else:
            report.append("  - No algorithms completed successfully. Check parameters and error logs.")
        
        return "\n".join(report)
    
    def plot_results(self, save_path: str = None) -> str:
        """
        Generate plots comparing algorithm performance.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save plots. If None, saves to output/benchmarks/
            
        Returns
        -------
        str
            Path to saved plot file
        """
        if not self.results:
            print("No results available. Run benchmark first.")
            return None
        
        # Group results by algorithm
        algorithm_stats = {}
        for result in self.results['results']:
            alg = result['algorithm']
            if alg not in algorithm_stats:
                algorithm_stats[alg] = []
            algorithm_stats[alg].append(result)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Magic Quantum Sequence Optimization - Algorithm Comparison', fontsize=16)
        
        # 1. Runtime comparison
        ax1 = axes[0, 0]
        algorithms = []
        avg_runtimes = []
        std_runtimes = []
        
        for alg in sorted(algorithm_stats.keys()):
            results = [r for r in algorithm_stats[alg] if r['success']]
            if results:
                algorithms.append(alg)
                avg_runtimes.append(np.mean([r['runtime'] for r in results]))
                std_runtimes.append(np.std([r['runtime'] for r in results]))
        
        if algorithms:
            ax1.bar(algorithms, avg_runtimes, yerr=std_runtimes, capsize=5, alpha=0.7)
            ax1.set_title('Average Runtime Comparison')
            ax1.set_ylabel('Runtime (seconds)')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Best expectation value comparison
        ax2 = axes[0, 1]
        best_expectations = []
        
        for alg in algorithms:
            results = [r for r in algorithm_stats[alg] if r['success']]
            best_expectations.append(min([r['best_expectation'] for r in results]))
        
        if best_expectations:
            ax2.bar(algorithms, best_expectations, alpha=0.7, color='orange')
            ax2.set_title('Best Expectation Value (Lower is Better)')
            ax2.set_ylabel('Expectation Value')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Best probability comparison
        ax3 = axes[1, 0]
        best_probabilities = []
        
        for alg in algorithms:
            results = [r for r in algorithm_stats[alg] if r['success']]
            best_probabilities.append(max([r['best_probability'] for r in results]))
        
        if best_probabilities:
            ax3.bar(algorithms, best_probabilities, alpha=0.7, color='green')
            ax3.set_title('Best Probability (Higher is Better)')
            ax3.set_ylabel('Probability')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Success rate comparison
        ax4 = axes[1, 1]
        success_rates = []
        
        for alg in sorted(algorithm_stats.keys()):
            results = algorithm_stats[alg]
            successful = [r for r in results if r['success']]
            success_rates.append(len(successful) / len(results) * 100)
        
        ax4.bar(sorted(algorithm_stats.keys()), success_rates, alpha=0.7, color='red')
        ax4.set_title('Success Rate')
        ax4.set_ylabel('Success Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"output/benchmarks/magic_algorithm_comparison_{timestamp}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {save_path}")
        return save_path


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark optimization algorithms for Magic project')
    parser.add_argument('--N', type=int, default=20, help='Hilbert space dimension')
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size')
    parser.add_argument('--max_generations', type=int, default=100, help='Max generations')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs per algorithm')
    parser.add_argument('--algorithms', nargs='+', default=None, 
                       help='Algorithms to test (default: all)')
    parser.add_argument('--target_superposition', nargs='+', default=['1.0', '1.0'],
                       help='Target superposition coefficients')
    parser.add_argument('--initial_state', default='vacuum', help='Initial state type')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='Convergence tolerance')
    parser.add_argument('--output', default=None, help='Output filename for results')
    
    args = parser.parse_args()
    
    # Parse target superposition
    try:
        # Simple parsing for real numbers
        target_superposition = tuple(float(x) for x in args.target_superposition)
    except ValueError as e:
        print(f"Error parsing target superposition: {e}")
        return 1
    
    # Create benchmark instance
    benchmark = AlgorithmBenchmark(
        N=args.N,
        sequence_length=args.sequence_length,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        target_superposition=target_superposition,
        initial_state=args.initial_state,
        gpu_id=args.gpu_id,
        tolerance=args.tolerance
    )
    
    # Run benchmark
    results = benchmark.run_benchmark(
        num_runs=args.num_runs,
        algorithms=args.algorithms
    )
    
    # Save results
    json_file = benchmark.save_results(args.output)
    
    # Generate and print report
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Generate plots
    plot_file = benchmark.plot_results()
    
    # Save report to file
    report_file = json_file.replace('.json', '_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
