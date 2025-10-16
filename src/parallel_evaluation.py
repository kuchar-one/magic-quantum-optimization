"""
Parallel evaluation module for quantum operation sequence optimization.

This module provides parallel evaluation capabilities to utilize multiple CPU cores
and fully leverage GPU resources. It uses torch.multiprocessing to enable proper
CUDA support across multiple processes.
"""

import torch
import torch.multiprocessing as mp
import numpy as np
from typing import List, Tuple, Dict, Any
from functools import partial
import os
import sys


class ParallelEvaluator:
    """
    Parallel evaluator for quantum operation sequences.
    
    This class manages a pool of worker processes that evaluate individuals
    in parallel, maximizing CPU and GPU utilization.
    """
    
    def __init__(
        self,
        num_workers: int = None,
        device_id: int = 0,
        use_ket_optimization: bool = True,
    ):
        """
        Initialize the parallel evaluator.
        
        Parameters
        ----------
        num_workers : int, optional
            Number of worker processes to use. If None, uses half of available CPU cores.
        device_id : int
            GPU device ID to use
        use_ket_optimization : bool
            Whether to use ket-based optimization
        """
        self.device_id = device_id
        self.use_ket_optimization = use_ket_optimization
        
        # Determine number of workers
        if num_workers is None:
            # Use half of available CPU cores
            num_workers = max(1, os.cpu_count() // 2)
        
        self.num_workers = num_workers
        self.pool = None
        
        # Set multiprocessing start method for CUDA compatibility
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Start method already set
            pass
        
        print(f"ParallelEvaluator initialized with {self.num_workers} workers on GPU {device_id}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def start(self):
        """Start the worker pool."""
        if self.pool is None:
            self.pool = mp.Pool(
                processes=self.num_workers,
                initializer=_worker_init,
                initargs=(self.device_id, self.use_ket_optimization)
            )
    
    def stop(self):
        """Stop the worker pool."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
    
    def evaluate_batch(
        self,
        individuals: np.ndarray,
        problem_data: Dict[str, Any],
    ) -> np.ndarray:
        """
        Evaluate a batch of individuals in parallel.
        
        Parameters
        ----------
        individuals : np.ndarray
            Array of individuals to evaluate (shape: [n_individuals, n_vars])
        problem_data : dict
            Problem data containing initial state, operator, etc.
            
        Returns
        -------
        np.ndarray
            Objective values for each individual (shape: [n_individuals, n_obj])
        """
        if self.pool is None:
            self.start()
        
        # Split individuals into chunks for workers
        n_individuals = len(individuals)
        chunk_size = max(1, n_individuals // self.num_workers)
        chunks = [
            individuals[i:i + chunk_size]
            for i in range(0, n_individuals, chunk_size)
        ]
        
        # Create evaluation function with problem data
        eval_func = partial(_evaluate_worker, problem_data=problem_data)
        
        # Evaluate chunks in parallel
        results = self.pool.map(eval_func, chunks)
        
        # Combine results
        return np.vstack(results)


# Global worker state
_worker_state = None


def _worker_init(device_id: int, use_ket_optimization: bool):
    """
    Initialize a worker process.
    
    This function is called once per worker process to set up the GPU context.
    
    Parameters
    ----------
    device_id : int
        GPU device ID to use
    use_ket_optimization : bool
        Whether to use ket-based optimization
    """
    global _worker_state
    
    # Set CUDA device for this worker
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        # Enable memory efficient attention if available
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize worker state
    _worker_state = {
        'device_id': device_id,
        'use_ket_optimization': use_ket_optimization,
        'device': f'cuda:{device_id}',
        'initialized': True,
    }
    
    # Suppress worker initialization messages
    sys.stdout = open(os.devnull, 'w')


def _evaluate_worker(
    chunk: np.ndarray,
    problem_data: Dict[str, Any],
) -> np.ndarray:
    """
    Worker function to evaluate a chunk of individuals.
    
    Parameters
    ----------
    chunk : np.ndarray
        Chunk of individuals to evaluate
    problem_data : dict
        Problem data containing initial state, operator, etc.
        
    Returns
    -------
    np.ndarray
        Objective values for the chunk
    """
    global _worker_state
    
    if _worker_state is None:
        raise RuntimeError("Worker not initialized")
    
    device = _worker_state['device']
    use_ket_optimization = _worker_state['use_ket_optimization']
    
    # Extract problem data
    initial_state = problem_data['initial_state']
    operator = problem_data['operator']
    projector = problem_data['projector']
    sequence_length = problem_data['sequence_length']
    N = problem_data['N']
    gamma = problem_data['gamma']
    
    # Initialize quantum operations for this worker
    from src.cuda_quantum_ops import GPUQuantumOps
    from src.cuda_helpers import GPUDeviceWrapper
    
    base_quantum_ops = GPUQuantumOps(N)
    extended_N = int(round(N * 1.5))
    helper_quantum_ops = GPUQuantumOps(extended_N)
    quantum_ops = GPUDeviceWrapper(base_quantum_ops, _worker_state['device_id'])
    helper_ops = GPUDeviceWrapper(helper_quantum_ops, _worker_state['device_id'])
    
    # Convert tensors to appropriate device
    initial_state = initial_state.to(device)
    operator = operator.to(device)
    projector = projector.to(device)
    
    # Evaluate each individual in the chunk
    results = []
    for i in range(len(chunk)):
        individual_result = _evaluate_single_individual(
            chunk[i],
            initial_state,
            operator,
            projector,
            quantum_ops,
            helper_ops,
            sequence_length,
            N,
            gamma,
            use_ket_optimization,
            device,
        )
        results.append(individual_result)
        
        # Periodic memory cleanup
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    return np.array(results)


def _evaluate_single_individual(
    individual: np.ndarray,
    initial_state: torch.Tensor,
    operator: torch.Tensor,
    projector: torch.Tensor,
    quantum_ops,
    helper_ops,
    sequence_length: int,
    N: int,
    gamma: float,
    use_ket_optimization: bool,
    device: str,
) -> Tuple[float, float]:
    """
    Evaluate a single individual.
    
    Parameters
    ----------
    individual : np.ndarray
        Individual to evaluate
    initial_state : torch.Tensor
        Initial quantum state
    operator : torch.Tensor
        Operator for expectation value calculation
    projector : torch.Tensor
        Projector for breeding operations
    quantum_ops : GPUQuantumOps
        Quantum operations handler
    helper_ops : GPUQuantumOps
        Helper quantum operations handler
    sequence_length : int
        Length of operation sequence
    N : int
        Hilbert space dimension
    gamma : float
        Damping parameter
    use_ket_optimization : bool
        Whether to use ket-based optimization
    device : str
        Device string (e.g., 'cuda:0')
        
    Returns
    -------
    tuple
        (operator_expectation, -probability)
    """
    operations = individual.reshape(-1, 3)
    current_state = initial_state.clone()
    ops = []
    total_probability = 1.0
    
    for op_idx in range(len(operations)):
        try:
            op_type = int(np.round(operations[op_idx, 0]))
            param1, param2 = operations[op_idx, 1:]
            
            # Apply operation
            new_state, op_type = _apply_operation_to_state(
                current_state,
                op_type,
                param1,
                param2,
                quantum_ops,
                helper_ops,
                projector,
                N,
                gamma,
                use_ket_optimization,
                device,
            )
            
            # Check if operation was valid
            if torch.isfinite(torch.norm(new_state)):
                current_state = new_state
                
                # Calculate operation probability
                op_probability = _calculate_operation_probability(
                    op_type, param1, param2, gamma
                )
                
                if op_probability == 0:
                    total_probability = 0
                    break
                
                ops.append((op_type, op_probability))
            else:
                total_probability = 0
                break
                
        except Exception as e:
            total_probability = 0
            break
    
    # Calculate sequence probability
    if total_probability != 0:
        total_probability = _calculate_sequence_probability(ops, 1.0)
    
    # Calculate operator expectation
    if total_probability != 0:
        try:
            if use_ket_optimization and len(current_state.shape) == 1:
                operator_expectation = torch.real(
                    torch.vdot(current_state, torch.mv(operator, current_state))
                ).cpu().numpy()
            else:
                operator_expectation = torch.real(
                    torch.trace(torch.matmul(operator, current_state))
                ).cpu().numpy()
            
            # Check for invalid values
            if not np.isfinite(operator_expectation):
                operator_expectation = np.inf
        except Exception:
            operator_expectation = np.inf
    else:
        operator_expectation = np.inf
    
    return (float(operator_expectation), -float(total_probability))


def _apply_operation_to_state(
    state: torch.Tensor,
    op_type: int,
    param1: float,
    param2: float,
    quantum_ops,
    helper_ops,
    projector: torch.Tensor,
    N: int,
    gamma: float,
    use_ket_optimization: bool,
    device: str,
) -> Tuple[torch.Tensor, int]:
    """
    Apply a quantum operation to a state.
    
    Returns
    -------
    tuple
        (new_state, op_type)
    """
    state = state.clone()
    
    if op_type == 0:  # Unitary (identity)
        return state, op_type
    elif op_type == 1:  # Displacement
        complex_param = param1 + 1j * param2
        state = _apply_displacement(state, complex_param, helper_ops, use_ket_optimization, device)
        return state / torch.norm(state), op_type
    elif op_type == 2:  # Squeezing
        complex_param = param1 + 1j * param2
        state = _apply_squeezing(state, complex_param, helper_ops, use_ket_optimization, device)
        return state / torch.norm(state), op_type
    elif op_type == 3:  # Breeding
        rounds = 1
        state = _apply_breeding(state, rounds, quantum_ops, projector, use_ket_optimization, device)
        return state / torch.norm(state), op_type
    elif op_type == 4:  # Annihilation
        if use_ket_optimization:
            state = torch.mv(quantum_ops.d, state)
        else:
            state = quantum_ops.subtraction_kraus @ state @ quantum_ops.subtraction_kraus.T.conj()
        return state / torch.norm(state), op_type
    elif op_type == 5:  # Creation
        if use_ket_optimization:
            state = torch.mv(quantum_ops.d.T.conj(), state)
        else:
            state = quantum_ops.addition_kraus @ state @ quantum_ops.addition_kraus.T.conj()
        return state / torch.norm(state), op_type
    
    return state, op_type


def _apply_displacement(
    state: torch.Tensor,
    complex_param: complex,
    helper_ops,
    use_ket_optimization: bool,
    device: str,
) -> torch.Tensor:
    """Apply displacement operation."""
    try:
        displacement_op = helper_ops.displace(complex_param)
        
        if use_ket_optimization and len(state.shape) == 1:
            state_dim = state.shape[0]
            displacement_dim = displacement_op.shape[0]
            if state_dim < displacement_dim:
                extended_ket = torch.zeros(
                    displacement_dim,
                    device=state.device,
                    dtype=state.dtype,
                )
                extended_ket[:state_dim] = state
                gpu_result = torch.mv(displacement_op, extended_ket)
                gpu_result = gpu_result[:state_dim]
            else:
                gpu_result = torch.mv(displacement_op, state)
        else:
            state_dim = state.shape[0]
            displacement_dim = displacement_op.shape[0]
            if state_dim < displacement_dim:
                extended_state = torch.zeros(
                    (displacement_dim, displacement_dim),
                    device=state.device,
                    dtype=state.dtype,
                )
                extended_state[:state_dim, :state_dim] = state
                gpu_result = torch.matmul(
                    displacement_op, torch.matmul(extended_state, displacement_op.T.conj())
                )
                gpu_result = gpu_result[:state_dim, :state_dim]
            else:
                gpu_result = torch.matmul(
                    displacement_op, torch.matmul(state, displacement_op.T.conj())
                )
        
        return gpu_result
    except Exception:
        return state


def _apply_squeezing(
    state: torch.Tensor,
    complex_param: complex,
    helper_ops,
    use_ket_optimization: bool,
    device: str,
) -> torch.Tensor:
    """Apply squeezing operation."""
    try:
        squeezing_op = helper_ops.squeeze(complex_param / 2)
        
        if use_ket_optimization and len(state.shape) == 1:
            state_dim = state.shape[0]
            squeezing_dim = squeezing_op.shape[0]
            if state_dim < squeezing_dim:
                extended_ket = torch.zeros(
                    squeezing_dim,
                    device=state.device,
                    dtype=state.dtype,
                )
                extended_ket[:state_dim] = state
                gpu_result = torch.mv(squeezing_op, extended_ket)
                gpu_result = gpu_result[:state_dim]
            else:
                gpu_result = torch.mv(squeezing_op, state)
        else:
            state_dim = state.shape[0]
            squeezing_dim = squeezing_op.shape[0]
            if state_dim < squeezing_dim:
                extended_state = torch.zeros(
                    (squeezing_dim, squeezing_dim),
                    device=state.device,
                    dtype=state.dtype,
                )
                extended_state[:state_dim, :state_dim] = state
                gpu_result = torch.matmul(
                    squeezing_op, torch.matmul(extended_state, squeezing_op.T.conj())
                )
                gpu_result = gpu_result[:state_dim, :state_dim]
            else:
                gpu_result = torch.matmul(
                    squeezing_op, torch.matmul(state, squeezing_op.T.conj())
                )
        
        return gpu_result
    except Exception:
        return state


def _apply_breeding(
    state: torch.Tensor,
    rounds: int,
    quantum_ops,
    projector: torch.Tensor,
    use_ket_optimization: bool,
    device: str,
) -> torch.Tensor:
    """Apply breeding operation."""
    try:
        if use_ket_optimization and len(state.shape) == 1:
            # Simplified breeding for kets
            return state
        else:
            result = quantum_ops.breeding_gpu(rounds, state, projector)
            return result
    except Exception:
        return state


def _calculate_operation_probability(
    op_type: int,
    param1: float,
    param2: float,
    gamma: float,
) -> float:
    """Calculate operation probability."""
    op_type = int(np.round(op_type))
    
    if op_type == 0:  # Unitary
        return 1.0
    elif op_type == 1:  # Displacement
        magnitude = 1 - np.sqrt(param1**2 + param2**2) / 100
        return magnitude
    elif op_type == 2:  # Squeezing
        magnitude = np.sqrt(param1**2 + param2**2)
        return 1 - (10 * np.log10(np.exp(2 * magnitude))) / 100
    elif op_type == 3:  # Breeding
        return 0.95
    elif op_type == 4:  # Annihilation
        return np.exp(2*gamma-1)
    elif op_type == 5:  # Creation
        return 0.5*np.exp(2*gamma-1)
    return 0.0


def _calculate_sequence_probability(
    operations: List[Tuple[int, float]],
    initial_probability: float = 1.0,
) -> float:
    """Calculate total probability of operation sequence."""
    total_probability = initial_probability
    
    for op, probability in operations:
        total_probability *= probability
        if op != 0:
            total_probability *= 0.99
        # Note: Breeding (op==3) recursive probability calculation removed
        # to avoid infinite recursion. Breeding probability is handled in
        # calculate_operation_probability as a fixed value (0.95)
    
    return total_probability

