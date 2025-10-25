import numpy as np
import torch
import qutip as qt
import random
import logging
import sys
import signal
import os
from typing import Optional
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.util.ref_dirs import get_reference_directions
from src.cuda_helpers import set_gpu_device, GPUDeviceWrapper, check_memory_and_cleanup, aggressive_memory_cleanup
from src.cuda_quantum_ops import GPUQuantumOps
from src.qutip_quantum_ops import gkp_operator_new, p0_projector, breeding
from src.logging import setup_logger
from src.sampling import (
    HundredsDiscreteMutation,
    HundredsDiscreteCrossover,
    HundredsDiscreteSampling,
)
from src.parallel_evaluation import ParallelEvaluator

logger = setup_logger()

# Global interrupt handling
_interrupt_requested = False
_interrupt_count = 0

def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully - first press saves results, second press exits immediately."""
    global _interrupt_requested, _interrupt_count
    _interrupt_count += 1
    
    if _interrupt_count == 1:
        _interrupt_requested = True
        print("\n🛑 Interrupt requested! Finishing current generation and saving results...")
        print("   (Press Ctrl+C again to exit immediately)")
    else:
        print("\n🚨 Force exit requested!")
        sys.exit(1)

def get_optimization_algorithm(algorithm_name: str, pop_size: int, initial_population: Optional[np.ndarray] = None, custom_crossover=None, custom_mutation=None):
    """
    Get the appropriate optimization algorithm based on the name.
    
    Parameters
    ----------
    algorithm_name : str
        Name of the algorithm to use
    pop_size : int
        Population size
    initial_population : Optional[np.ndarray]
        Initial population for the algorithm
    custom_crossover : dict, optional
        Custom crossover parameters {'eta': float, 'prob': float}
    custom_mutation : dict, optional
        Custom mutation parameters {'eta': float, 'prob': float}
        
    Returns
    -------
    algorithm
        Configured PyMOO algorithm
    """
    algorithm_name = algorithm_name.lower()
    
    # Extract custom parameters with defaults
    crossover_eta = custom_crossover.get('eta', 1.0) if custom_crossover else 1.0
    crossover_prob = custom_crossover.get('prob', 0.8) if custom_crossover else 0.8
    mutation_eta = custom_mutation.get('eta', 3.0) if custom_mutation else 3.0
    mutation_prob = custom_mutation.get('prob', 0.95) if custom_mutation else 0.95
    
    if algorithm_name == "nsga2":
        return NSGA2(
            pop_size=pop_size,
            sampling=HundredsDiscreteSampling(initial_population=initial_population),
            crossover=HundredsDiscreteCrossover(eta=crossover_eta, prob=crossover_prob),
            mutation=HundredsDiscreteMutation(eta=mutation_eta, prob=mutation_prob),
            eliminate_duplicates=True,
        )
    
    elif algorithm_name == "nsga3":
        # NSGA3 needs reference directions - use population size to determine partitions
        n_partitions = max(12, pop_size - 1)  # Ensure at least 13 directions
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=n_partitions)
        
        print(f"  - NSGA3 configured with {len(ref_dirs)} reference directions (n_partitions={n_partitions})")
        print(f"  - Max possible Pareto solutions: {len(ref_dirs)}")
        
        return NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=HundredsDiscreteSampling(initial_population=initial_population),
            crossover=HundredsDiscreteCrossover(eta=crossover_eta, prob=crossover_prob),
            mutation=HundredsDiscreteMutation(eta=mutation_eta, prob=mutation_prob),
            eliminate_duplicates=True,
        )
    
    elif algorithm_name == "moead":
        # MOEA/D needs reference directions - use population size to determine partitions
        n_partitions = max(12, pop_size - 1)  # Ensure at least 13 directions
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=n_partitions)
        
        # Adjust neighbors based on number of reference directions
        n_neighbors = min(15, len(ref_dirs) // 2)
        
        print(f"  - MOEA/D configured with {len(ref_dirs)} reference directions (n_partitions={n_partitions})")
        print(f"  - Max possible Pareto solutions: {len(ref_dirs)}")
        
        return MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=n_neighbors,
            prob_neighbor_mating=0.7,
            sampling=HundredsDiscreteSampling(initial_population=initial_population),
            crossover=HundredsDiscreteCrossover(eta=crossover_eta, prob=crossover_prob),
            mutation=HundredsDiscreteMutation(eta=mutation_eta, prob=mutation_prob),
        )
    
    elif algorithm_name == "age":
        return AGEMOEA(
            pop_size=pop_size,
            sampling=HundredsDiscreteSampling(initial_population=initial_population),
            crossover=HundredsDiscreteCrossover(eta=crossover_eta, prob=crossover_prob),
            mutation=HundredsDiscreteMutation(eta=mutation_eta, prob=mutation_prob),
            eliminate_duplicates=True,
        )
    
    elif algorithm_name == "age2":
        return AGEMOEA2(
            pop_size=pop_size,
            sampling=HundredsDiscreteSampling(initial_population=initial_population),
            crossover=HundredsDiscreteCrossover(eta=crossover_eta, prob=crossover_prob),
            mutation=HundredsDiscreteMutation(eta=mutation_eta, prob=mutation_prob),
            eliminate_duplicates=True,
        )
    
    elif algorithm_name == "rvea":
        # RVEA requires different termination criteria, skip for now
        raise ValueError("RVEA requires special termination criteria - not supported in current setup")
    
    elif algorithm_name == "smsemoa":
        return SMSEMOA(
            pop_size=pop_size,
            sampling=HundredsDiscreteSampling(initial_population=initial_population),
            crossover=HundredsDiscreteCrossover(eta=crossover_eta, prob=crossover_prob),
            mutation=HundredsDiscreteMutation(eta=mutation_eta, prob=mutation_prob),
            eliminate_duplicates=True,
        )
    
    elif algorithm_name == "ctaea":
        # CTAEA has parameter conflicts, skip for now
        raise ValueError("CTAEA has parameter conflicts - not supported in current setup")
    
    elif algorithm_name == "unsga3":
        # UNSGA3 needs reference directions - use population size to determine partitions
        n_partitions = max(12, pop_size - 1)  # Ensure at least 13 directions
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=n_partitions)
        
        print(f"  - UNSGA3 configured with {len(ref_dirs)} reference directions (n_partitions={n_partitions})")
        print(f"  - Max possible Pareto solutions: {len(ref_dirs)}")
        
        return UNSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=HundredsDiscreteSampling(initial_population=initial_population),
            crossover=HundredsDiscreteCrossover(eta=crossover_eta, prob=crossover_prob),
            mutation=HundredsDiscreteMutation(eta=mutation_eta, prob=mutation_prob),
            eliminate_duplicates=True,
        )
    
    elif algorithm_name == "rnsga2":
        # RNSGA2 requires reference points, skip for now
        raise ValueError("RNSGA2 requires reference points - not supported in current setup")
    
    elif algorithm_name == "rnsga3":
        # RNSGA3 requires reference points and pop_per_ref_point, skip for now
        raise ValueError("RNSGA3 requires reference points and pop_per_ref_point - not supported in current setup")
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available algorithms: "
                        "nsga2, nsga3, moead, age, age2, rvea, smsemoa, ctaea, unsga3, rnsga2, rnsga3")


class QuantumOperationSequence(Problem):
    def __init__(
        self,
        initial_state,
        initial_probability,
        operator,
        N=30,
        sequence_length=10,
        device_id=0,
        use_ket_optimization=True,
        num_workers=None,
    ):
        """
        Initialize the quantum operation sequence optimization problem

        Args:
            initial_state: Initial quantum state
            operator: Positive operator to minimize
            N: Dimension of Hilbert space
            sequence_length: Number of operations in sequence
            device_id: ID of GPU device to use
            use_ket_optimization: If True, use ket-based optimization (pure states)
            num_workers: Number of parallel workers (None = auto-detect as half of CPU cores)
        """
        super().__init__(
            n_var=sequence_length * 3,
            n_obj=2,
            n_constr=0,
            xl=np.array([-0.1, -2.0, -2.0] * sequence_length),
            xu=np.array([5.1, 2.0, 2.0] * sequence_length),
        )
        logger.info(f"Quantum operation sequence optimization problem initialization")
        self.N = N
        self.sequence_length = sequence_length
        self.device = f"cuda:{device_id}"
        self.use_ket_optimization = use_ket_optimization
        self.initial_probability = initial_probability
        self.num_workers = num_workers

        # Initialize quantum operations
        base_quantum_ops = GPUQuantumOps(self.N)
        extended_N = np.round(self.N * 1.5).astype(int)
        helper_quantum_ops = GPUQuantumOps(extended_N)
        self.quantum_ops = GPUDeviceWrapper(base_quantum_ops, device_id)
        self.helper_ops = GPUDeviceWrapper(helper_quantum_ops, device_id)

        if operator is None:
            operator = gkp_operator_new(N)
            logger.info("GKP operator utilized as default.")

        groundstate_eigval_qutip = operator.groundstate()[0]
        logger.info(f"Operator groundstate eigenvalue: {groundstate_eigval_qutip}")

        print(f"Operator type: {type(operator)}")
        print(f"Operator has .full() method: {hasattr(operator, 'full')}")
        self.op = torch.tensor(operator.full(), dtype=torch.complex64).to(self.device)
        eigenvalues, _ = torch.linalg.eig(self.op)
        self.groundstate_eigenvalue = torch.min(eigenvalues.real)
        logger.info(
            f"Operator torch groundstate eigenvalue: {self.groundstate_eigenvalue}"
        )

        try:
            projector_qobj = p0_projector(N)
            self.projector = torch.tensor(projector_qobj.full(), dtype=torch.complex64).to(
                self.device
            )
        except Exception as e:
            print(f"Error with projector: {e}")
            import traceback
            traceback.print_exc()
            raise
        self.qutip_projector = p0_projector(N)
        self._current_eval_idx = None
        self.previous_op_changed = False

        try:
            # Initialize gamma for operation probability calculation
            self.gamma = 0.1
            
            if use_ket_optimization:
                # Ket-based optimization: work with pure states
                self.initial_state = torch.tensor(
                    initial_state.full(), dtype=torch.complex64
                ).squeeze().to(self.device)
                self.initial_state = self.initial_state / torch.norm(self.initial_state)
                logger.info("Using ket-based optimization (pure states)")
            else:
                # Density matrix optimization: convert to density matrix
                self.initial_state = torch.tensor(
                    qt.ket2dm(initial_state).full(), dtype=torch.complex64
                ).to(self.device)
                self.initial_state = self.initial_state / torch.norm(self.initial_state)
                # Initialize Kraus operators for mixed state operations
                self.damping_operator = self._damping_operator()
                self.subtraction_kraus = torch.sqrt(torch.exp(torch.tensor(2*self.gamma)) - 1.0) * (self.damping_operator @ self.quantum_ops.d)
                self.addition_kraus = torch.sqrt(torch.exp(torch.tensor(2*self.gamma)) - 1.0) * (self.damping_operator @ self.quantum_ops.d.T.conj())
                logger.info("Using density matrix optimization (mixed states)")
        except Exception as e:
            print(f"Error with initial state: {e}")
            import traceback
            traceback.print_exc()
            raise

        logger.info("Operators initialized")
        logger.info("Testing GPU operations...")
        
        # Initialize parallel evaluator
        self.parallel_evaluator = ParallelEvaluator(
            num_workers=self.num_workers,
            device_id=device_id,
            use_ket_optimization=use_ket_optimization,
        )
        self.parallel_evaluator.start()
        
        # Prepare problem data for parallel evaluation
        self.problem_data = {
            'initial_state': self.initial_state,
            'operator': self.op,
            'projector': self.projector,
            'sequence_length': self.sequence_length,
            'N': self.N,
            'gamma': self.gamma,
        }
        
        
    def _damping_operator(self):
        """
        Returns the diagonal 'damping' operator E = e^{-γ a†a}.
        """
        a = self.quantum_ops.d
        adag = a.T.conj()
        n_op = adag @ a
        return torch.linalg.matrix_exp(-self.gamma * n_op)

    def sequence_probability(self, operations, total_probability=None):
        """
        Calculate the total probability of a sequence of quantum operations.

        This function iterates over a list of operations, where each operation is
        represented as a tuple containing the operation type and its probability.
        It calculates the cumulative probability of the sequence by multiplying
        the probabilities of individual operations, applying a decay factor of 0.99
        for non-zero operations, and handling special cases such as breeding
        operations (op type 3) which have recursive probability contributions.

        Parameters
        ----------
        operations : list of tuple
            A list of tuples where each tuple consists of an operation type (int)
            and its probability (float).
        total_probability : float, optional
            Initial probability to start the calculation with. Default is 1.

        Returns
        -------
        float
            The total probability of the given operation sequence.
        """
        if total_probability == None:
            total_probability = 1
        elif len(operations) == 0:
            total_probability *= self.initial_probability
            return total_probability

        for index, (op, probability) in enumerate(operations):
            total_probability *= probability
            if op != 0:
                total_probability *= 0.99
            if op == 3:
                total_probability *= (
                    self.sequence_probability(
                        operations[index + 1 :], total_probability
                    )
                    ** 2
                )
                return total_probability
        total_probability *= self.initial_probability
        return total_probability

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the given set of operation sequences using parallel processing.

        Parameters
        ----------
        x : array_like
            2D array of operation sequences, where each row is a sequence of operations.
            Each operation is represented as a triplet (op_type, param1, param2).
        out : dict
            Dictionary to store the results.
        """
        # Use parallel evaluation
        try:
            f = self.parallel_evaluator.evaluate_batch(x, self.problem_data)
        except Exception as e:
            logger.error(f"Error in parallel evaluation: {e}")
            # Fallback to sequential evaluation
            f = self._evaluate_sequential(x)
        
        out["F"] = f
        
        # Periodic memory cleanup
        if hasattr(self, '_eval_count'):
            self._eval_count += 1
        else:
            self._eval_count = 1
        
        if self._eval_count % 10 == 0:
            check_memory_and_cleanup(memory_threshold_mb=1200, label=f"EvalBatch{self._eval_count}")
            
class OptimizationCallback(Callback):
    def __init__(self) -> None:
        """
        Initialize the optimization callback.

        This callback will store the optimization results in the 'data' attribute,
        specifically the objective function values in the 'F' key.
        """
        super().__init__()
        self.data["F"] = []

    def notify(self, algorithm):
        """
        Notify callback method for the optimization algorithm.

        This method is called by the optimization algorithm and stores the current
        objective function values in the 'data' attribute.

        Parameters
        ----------
        algorithm : pymoo.algorithms.Algorithm
            The optimization algorithm.

        Returns
        -------
        None
        """
        global _interrupt_requested
        
        F = algorithm.opt.get("F").copy()
        self.data["F"].append(F)
        
        # Limit callback data size to prevent memory leaks
        if len(self.data["F"]) > 50:  # Keep only last 50 generations
            self.data["F"] = self.data["F"][-50:]
        
        gen = algorithm.n_gen
        pop_size = len(F)
        min_exp = np.min(F[:, 0])
        max_prob = np.max(-F[:, 1])
        print(
            f"Generation {gen:3d}: {pop_size:3d} solutions, min expectation: {min_exp:.6f}, max probability: {max_prob:.6f}"
        )
        
        # Check memory usage and cleanup if needed every 10 generations (reduced frequency)
        if gen % 10 == 0:
            check_memory_and_cleanup(memory_threshold_mb=1200, label=f"Gen{gen}")
        
        # Check for interrupt request
        if _interrupt_requested:
            print(f"\n✅ Gracefully stopping optimization at generation {gen}")
            print(f"   Final results: {pop_size} solutions, best expectation: {min_exp:.6f}, best probability: {max_prob:.6f}")
            # Stop the algorithm by setting termination criteria
            algorithm.termination.force_termination = True


def run_quantum_sequence_optimization(
    initial_state,
    initial_probability,
    operator=None,
    N=30,
    sequence_length=10,
    pop_size=100,
    max_generations=100,
    device_id=0,
    tolerance=1e-3,
    initial_population=None,
    algorithm="nsga2",
    enable_signal_handler=True,
    use_ket_optimization=True,
    callback=None,
    verbose=False,
    custom_crossover=None,
    custom_mutation=None,
    num_workers=None,
):
    """
    Run the quantum operation sequence optimization

    Args:
        initial_state: Initial quantum state
        N: Hilbert space dimension
        sequence_length: Number of operations in sequence
        pop_size: Population size
        max_generations: Number of generations
        initial_population: Initial population for the optimization
        algorithm: Optimization algorithm to use
        enable_signal_handler: Whether to enable graceful interrupt handling
        num_workers: Number of parallel workers (None = auto-detect as half of CPU cores)

    Returns:
        Optimization results
    """
    global _interrupt_requested, _interrupt_count
    
    # Reset interrupt flags
    _interrupt_requested = False
    _interrupt_count = 0
    
    # Set up signal handler for graceful interruption (only in main thread)
    if enable_signal_handler:
        try:
            signal.signal(signal.SIGINT, _signal_handler)
        except ValueError:
            # Signal handler can only be set in main thread
            print("⚠️  Signal handler disabled (not in main thread)")
            enable_signal_handler = False
    
    # Determine number of workers if not specified
    if num_workers is None:
        num_workers = max(1, os.cpu_count() // 2)
    
    print(f"Starting quantum sequence optimization:")
    print(f"  - Algorithm: {algorithm.upper()}")
    print(f"  - Hilbert space dimension: {N}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Population size: {pop_size}")
    print(f"  - Max generations: {max_generations}")
    print(f"  - Parallel workers: {num_workers} (using {num_workers}/{os.cpu_count()} CPU cores)")
    if enable_signal_handler:
        print(f"  - Press Ctrl+C once to gracefully stop and save results")
    print()

    problem = QuantumOperationSequence(
        initial_state, initial_probability, operator, N, sequence_length, device_id, use_ket_optimization, num_workers
    )

    # Create trivial solution (do nothing: all identity operations)
    # This gives highest probability since it preserves the initial state
    n_var = sequence_length * 3  # 3 parameters per operation (op_type, param1, param2)
    trivial_solution = np.zeros(n_var)  # All identity operations (op_type=0, param1=0, param2=0)
    
    # If no initial population provided, create one with trivial solution
    if initial_population is None:
        initial_population = np.zeros((pop_size, n_var))
        initial_population[0] = trivial_solution  # First individual is trivial solution
        # Fill rest with random values within bounds
        for i in range(1, pop_size):
            # Operation types: bounded by [0, 5]
            initial_population[i, 0::3] = np.random.uniform(0, 5, sequence_length)
            # Parameters 1 and 2: bounded by [-2, 2]
            initial_population[i, 1::3] = np.random.uniform(-2, 2, sequence_length)
            initial_population[i, 2::3] = np.random.uniform(-2, 2, sequence_length)
    else:
        # If initial population provided, ensure trivial solution is included
        if len(initial_population) < pop_size:
            # Extend population to include trivial solution
            extended_pop = np.zeros((pop_size, n_var))
            extended_pop[:len(initial_population)] = initial_population
            extended_pop[len(initial_population)] = trivial_solution
            initial_population = extended_pop
        else:
            # Replace first individual with trivial solution
            initial_population[0] = trivial_solution

    algorithm_obj = get_optimization_algorithm(algorithm, pop_size, initial_population, custom_crossover, custom_mutation)
    
    print(f"  - Initial population includes trivial solution (all identity operations)")
    print(f"  - Population size: {pop_size}")

    # Use provided callback or create default one
    if callback is None:
        callback = OptimizationCallback()

    # Instantiate our custom termination object.
    termination = DefaultMultiObjectiveTermination(
        xtol=tolerance,
        cvtol=tolerance,
        ftol=tolerance,
        period=50,
        n_max_gen=max_generations,
        n_max_evals=max_generations * pop_size,
    )

    res = minimize(
        problem, algorithm_obj, termination, seed=42, verbose=verbose, callback=callback
    )
    
    # Clean up parallel evaluator
    if hasattr(problem, 'parallel_evaluator'):
        problem.parallel_evaluator.stop()

    # Check if optimization was interrupted
    if _interrupt_requested:
        print(f"\n🛑 Optimization interrupted gracefully!")
        print(f"  - Completed generations: {res.algorithm.n_gen}")
        print(f"  - Pareto front size: {len(res.F)}")
        print(f"  - Best expectation: {np.min(res.F[:, 0]):.6f}")
        print(f"  - Best probability: {np.max(-res.F[:, 1]):.6f}")
        print(f"  - Results will be saved and animations created...")
    else:
        print(f"\n✅ Optimization completed!")
        print(f"  - Final generation: {res.algorithm.n_gen}")
        print(f"  - Pareto front size: {len(res.F)}")
        print(f"  - Best expectation: {np.min(res.F[:, 0]):.6f}")
        print(f"  - Best probability: {np.max(-res.F[:, 1]):.6f}")

    return res, callback.data["F"]
