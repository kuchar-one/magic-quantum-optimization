"""
Parallel evaluation module for quantum operation sequence optimization.

This module provides parallel evaluation capabilities to utilize multiple CPU cores
and fully leverage GPU resources. It uses torch.multiprocessing to enable proper
CUDA support across multiple processes.
"""
from __future__ import annotations

import os
import sys
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

SAME_OP_PENALTY = 0.5  # Penalty factor for repeating the same operation type


class ParallelEvaluator:
    """
    Parallel evaluator for quantum operation sequences.

    This class manages a pool of worker processes that evaluate individuals
    in parallel, maximizing CPU and GPU utilization.
    """

    def __init__(
        self,
        num_workers: int | None = None,
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
            Base GPU device ID to use (workers will be offset by pid to spread across GPUs).
            Use -1 to force CPU-only workers.
        use_ket_optimization : bool
            Whether to use ket-based optimization
        """
        self.base_device_id = int(device_id)
        self.use_ket_optimization = bool(use_ket_optimization)

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) // 2)
        self.num_workers = int(num_workers)

        self.pool: mp.pool.Pool | None = None

        # Ensure 'spawn' start method for CUDA-multiprocessing correctness
        try:
            current = mp.get_start_method(allow_none=True)
            if current != "spawn":
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # start method already set and can't be changed -- that's fine
            pass

        print(
            f"ParallelEvaluator initialized with {self.num_workers} workers, base_device_id={self.base_device_id}, use_ket_optimization={self.use_ket_optimization}"
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """Start the worker pool."""
        if self.pool is None:
            initializer = _worker_init
            initargs = (self.base_device_id, self.use_ket_optimization)
            # Create pool
            self.pool = mp.Pool(processes=self.num_workers, initializer=initializer, initargs=initargs)

    def stop(self):
        """Stop the worker pool."""
        if self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
            finally:
                self.pool = None

    def evaluate_batch(self, individuals: np.ndarray, problem_data: Dict[str, Any]) -> np.ndarray:
        """
        Evaluate a batch of individuals in parallel.

        Parameters
        ----------
        individuals : np.ndarray
            Array of individuals to evaluate (shape: [n_individuals, n_vars])
        problem_data : dict
            Problem data containing initial state, operator, projector, sequence_length, N, ...

        Returns
        -------
        np.ndarray
            Objective values for each individual (shape: [n_individuals, n_obj])
            each row is (expectation, -probability)
        """
        if self.pool is None:
            self.start()

        # Ensure individuals is a 2D numpy array
        individuals = np.asarray(individuals)
        if individuals.ndim == 1:
            individuals = individuals.reshape(1, -1)

        n_individuals = individuals.shape[0]
        # create roughly equal chunks (every individual evaluated once)
        # chunk_size computed so we have at most num_workers chunks and each non-empty
        chunk_size = max(1, (n_individuals + self.num_workers - 1) // self.num_workers)
        chunks = [individuals[i : i + chunk_size] for i in range(0, n_individuals, chunk_size)]

        eval_func = partial(_evaluate_worker, problem_data=problem_data)
        results = self.pool.map(eval_func, chunks)

        # results is list of arrays (n_chunk, 2); combine to (n_individuals, 2)
        combined = np.vstack(results) if len(results) > 0 else np.zeros((0, 2), dtype=float)
        return combined


# Global worker state inside each worker process
_worker_state: Dict[str, Any] | None = None


def _worker_init(base_device_id: int, use_ket_optimization: bool):
    """
    Initialize a worker process.

    This function runs inside each worker process.
    We choose a device deterministically based on base_device_id and the process PID
    (so multiple spawned workers attempt to spread across available GPUs).

    Parameters
    ----------
    base_device_id : int
        base GPU id (or -1 for CPU-only)
    use_ket_optimization : bool
        whether to use ket-based optimization
    """
    global _worker_state

    # default device
    assigned_device: torch.device
    assigned_index: int | None = None

    try:
        if base_device_id is not None and base_device_id >= 0 and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Spread across GPUs using pid so different workers likely go to different GPUs
                pid_offset = os.getpid() % gpu_count
                assigned_index = (base_device_id + pid_offset) % gpu_count
                try:
                    torch.cuda.set_device(assigned_index)
                except Exception:
                    # set_device may fail in rare envs; fallback to device object without set_device
                    pass
                assigned_device = torch.device(f"cuda:{assigned_index}")
            else:
                assigned_device = torch.device("cpu")
                assigned_index = None
        else:
            assigned_device = torch.device("cpu")
            assigned_index = None
    except Exception:
        assigned_device = torch.device("cpu")
        assigned_index = None

    # Optional memory config for CUDA - do before heavy allocations
    if assigned_device.type == "cuda":
        # Enable expandable segments if environment allows (this is non-fatal)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

    _worker_state = {
        "device": assigned_device,
        "device_index": assigned_index,
        "use_ket_optimization": bool(use_ket_optimization),
        "initialized": True,
    }

    # light trace for debugging (prints to stderr)
    print(f"[worker init] pid={os.getpid()} device={assigned_device} ket={use_ket_optimization}", file=sys.stderr)


def _evaluate_worker(chunk: np.ndarray, problem_data: Dict[str, Any]) -> np.ndarray:
    """
    Worker function to evaluate a chunk of individuals.

    Returns a numpy array of shape (len(chunk), 2): (expectation_value, -probability)
    """
    global _worker_state
    if _worker_state is None:
        raise RuntimeError("Worker not initialized. Ensure _worker_init ran in worker process.")

    device: torch.device = _worker_state["device"]
    use_ket_optimization: bool = _worker_state["use_ket_optimization"]

    # Convert problem_data fields safely to torch tensors on worker device
    # Accept either numpy arrays or torch tensors for the inputs
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return torch.as_tensor(x).to(device)

    # required keys in problem_data
    required = ["initial_state", "operator", "projector", "sequence_length", "N"]
    for k in required:
        if k not in problem_data:
            raise KeyError(f"problem_data missing required key: {k}")

    initial_state = _to_tensor(problem_data["initial_state"])
    operator = _to_tensor(problem_data["operator"])
    projector = _to_tensor(problem_data["projector"])
    sequence_length: int = int(problem_data["sequence_length"])
    N: int = int(problem_data["N"])

    # Import GPU ops inside worker (so imports happen in worker process)
    # Expect src.cuda_quantum_ops.GPUQuantumOps and src.cuda_helpers.GPUDeviceWrapper to exist in project
    try:
        from src.cuda_quantum_ops import GPUQuantumOps  # type: ignore
        from src.cuda_helpers import GPUDeviceWrapper  # type: ignore
    except Exception as e:
        # Fail early and clearly if imports are missing
        raise ImportError(f"Failed to import GPU quantum ops helpers inside worker: {e}")

    # Create ops and wrap them to the device (GPUDeviceWrapper expected)
    base_quantum_ops = GPUQuantumOps(N)
    # use device_index (int) when available, otherwise fall back to 0 (or special-case CPU)
    device_index = _worker_state.get("device_index")
    if device_index is None:
        # no CUDA available on this worker -> don't use GPUDeviceWrapper, or pass 0 if you know it's ok
        # If your GPUQuantumOps class requires CUDA, you should not run on CPU; here we still attempt 0.
        device_index = 0
    quantum_ops = GPUDeviceWrapper(base_quantum_ops, device_index)


    # Prepare chunk as array of individuals
    chunk = np.asarray(chunk)
    results: List[Tuple[float, float]] = []

    for i in range(chunk.shape[0]):
        individual = chunk[i]
        try:
            res = _evaluate_single_individual(
                individual=individual,
                initial_state=initial_state,
                operator=operator,
                projector=projector,
                quantum_ops=quantum_ops,
                sequence_length=sequence_length,
                N=N,
                use_ket_optimization=use_ket_optimization,
                device=device,
            )
            results.append(res)
        except Exception as e:
            # On failure return (inf, 0) equivalent to bad individual
            print(f"[worker {os.getpid()}] exception evaluating individual {i}: {e}", file=sys.stderr)
            results.append((float("inf"), -0.0))

        # periodic cleanup if on CUDA
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return np.asarray(results, dtype=float)


def _evaluate_single_individual(
    individual: np.ndarray,
    initial_state: torch.Tensor,
    operator: torch.Tensor,
    projector: torch.Tensor,
    quantum_ops,
    sequence_length: int,
    N: int,
    use_ket_optimization: bool,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate a single individual and return (expectation, -probability)
    """
    # Ensure numpy row -> 1D array
    individual = np.asarray(individual).astype(float).reshape(-1)
    # Expect each op encoded as 3 numbers (type, param1, param2)
    if individual.size % 3 != 0:
        raise ValueError("Individual length must be multiple of 3 (op_type, param1, param2)")

    operations = individual.reshape(-1, 3)
    # Clone initial state so we don't modify shared object
    current_state = initial_state.clone()
    ops: List[Tuple[int, float, float, float]] = []
    total_probability = 1.0

    for op_idx in range(operations.shape[0]):
        try:
            op_type = int(np.round(operations[op_idx, 0]))
            param1 = float(operations[op_idx, 1])
            param2 = float(operations[op_idx, 2])

            new_state, applied_op_type = _apply_operation_to_state(
                state=current_state,
                op_type=op_type,
                param1=param1,
                param2=param2,
                quantum_ops=quantum_ops,
                projector=projector,
                N=N,
                use_ket_optimization=use_ket_optimization,
                device=device,
            )

            # Validate new_state
            try:
                if isinstance(new_state, torch.Tensor):
                    # If vector (ket) -> vector norm; if density matrix -> Frobenius norm.
                    if new_state.numel() == 0:
                        raise ValueError("Empty state returned")
                    norm_val = torch.linalg.vector_norm(new_state).to("cpu").item()
                else:
                    raise ValueError("Operation returned non-tensor state")
            except Exception:
                print(f"Non-finite or invalid state after op idx {op_idx}, op_type {op_type}", file=sys.stderr)
                total_probability = 0.0
                break

            if not np.isfinite(norm_val) or norm_val == 0.0:
                print(f"Invalid norm ({norm_val}) after op idx {op_idx}, op_type {op_type}", file=sys.stderr)
                total_probability = 0.0
                break

            # Accept the new normalized state
            current_state = new_state / norm_val

            # compute op probability and apply same-op penalty
            op_probability = _calculate_operation_probability(op_type, param1, param2)
            penalty = _apply_same_op_penalty(ops, op_type, param1, param2)
            op_probability = float(op_probability) * float(penalty)
            # clamp to [0,1]
            op_probability = max(0.0, min(1.0, op_probability))

            if op_probability == 0.0:
                total_probability = 0.0
                ops.append((op_type, op_probability, param1, param2))
                break

            ops.append((op_type, op_probability, param1, param2))

        except Exception as e:
            print(f"Exception during operation application at index {op_idx}: {e}", file=sys.stderr)
            total_probability = 0.0
            break

    # Sequence probability
    if total_probability != 0.0:
        total_probability = _calculate_sequence_probability_recursive(ops, initial_prob=1.0)

    # Operator expectation
    if total_probability != 0.0:
        try:
            if use_ket_optimization and current_state.dim() == 1:
                # expectation = real(v^H * (Operator * v))
                tmp = torch.mv(operator, current_state)
                operator_expectation = torch.real(torch.vdot(current_state, tmp)).to("cpu").item()
            else:
                # treat current_state as density matrix
                mat = torch.matmul(operator, current_state)
                operator_expectation = torch.real(torch.trace(mat)).to("cpu").item()

            if not np.isfinite(operator_expectation):
                print("Non-finite operator expectation encountered.", file=sys.stderr)
                operator_expectation = float("inf")
        except Exception as e:
            print(f"Exception during operator expectation calculation: {e}", file=sys.stderr)
            operator_expectation = float("inf")
    else:
        operator_expectation = float("inf")

    # Return tuple (expectation, -probability) to match earlier convention
    return float(operator_expectation), -float(total_probability)


def _apply_operation_to_state(
    state: torch.Tensor,
    op_type: int,
    param1: float,
    param2: float,
    quantum_ops,
    projector: torch.Tensor,
    N: int,
    use_ket_optimization: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """
    Apply a quantum operation to a state.
    Returns (new_state, op_type)
    """
    # Work on a clone to avoid side-effects
    state = state.clone()
    try:
        if op_type == 0:  # Unitary/identity
            return state, op_type
        elif op_type == 1:  # Displacement
            complex_param = complex(param1, param2)
            new_state = _apply_displacement(state, complex_param, use_ket_optimization, quantum_ops, device)
            return new_state, op_type
        elif op_type == 2:  # Squeezing
            complex_param = complex(param1, param2)
            new_state = _apply_squeezing(state, complex_param, use_ket_optimization, quantum_ops, device)
            return new_state, op_type
        elif op_type == 3:  # Breeding
            rounds = 1
            new_state = _apply_breeding(state, rounds, quantum_ops, projector, use_ket_optimization, device)
            return new_state, op_type
        elif op_type == 4:  # Annihilation (subtraction)
            if use_ket_optimization and hasattr(quantum_ops, "d"):
                # `d` assumed to be an operator matrix appropriate for mv
                new_state = torch.mv(quantum_ops.d, state)
            else:
                # density-matrix Kraus application
                if hasattr(quantum_ops, "subtraction_kraus"):
                    K = quantum_ops.subtraction_kraus
                    new_state = K @ state @ K.T.conj()
                else:
                    new_state = state
            return new_state, op_type
        elif op_type == 5:  # Creation (addition)
            if use_ket_optimization and hasattr(quantum_ops, "d"):
                new_state = torch.mv(quantum_ops.d.T.conj(), state)
            else:
                if hasattr(quantum_ops, "addition_kraus"):
                    K = quantum_ops.addition_kraus
                    new_state = K @ state @ K.T.conj()
                else:
                    new_state = state
            return new_state, op_type
    except Exception as e:
        print(f"Exception in _apply_operation_to_state(op_type={op_type}): {e}", file=sys.stderr)
        return state, op_type

    return state, op_type


def _apply_same_op_penalty(ops: List[Tuple[int, float, float, float]], current_op_type: int, current_param1: float, current_param2: float) -> float:
    """
    Applies same-op penalty if the same operation appears earlier in the sequence,
    ignoring intermediate operations with negligible effect (small displacement/squeezing).

    ops : list of tuples (op_type, op_probability, param1, param2) in chronological order
    current_op_type : int (1=Displacement, 2=Squeezing)
    returns multiplicative penalty factor (1.0 or SAME_OP_PENALTY)
    """
    # Only consider displacements and squeezings for penalty
    if current_op_type not in (1, 2):
        return 1.0

    THRESHOLD = 0.1  # magnitude below which an op is considered negligible
    # Scan backwards (most recent first)
    for prev_op in reversed(ops):
        prev_type, prev_prob, prev_p1, prev_p2 = prev_op
        if prev_type == current_op_type:
            return SAME_OP_PENALTY
        elif prev_type in (1, 2):
            mag = float(np.hypot(prev_p1, prev_p2))
            if mag > THRESHOLD:
                # significant intermediate same-class op -> stop scanning
                break
        else:
            # other operation interrupts the chain
            break
    return 1.0


def _apply_displacement(
    state: torch.Tensor,
    complex_param: complex,
    use_ket_optimization: bool,
    quantum_ops,
    device: torch.device,
) -> torch.Tensor:
    """Apply displacement operation using quantum_ops.displace(complex_param)."""
    try:
        displacement_op = quantum_ops.displace(complex_param)
        # Ensure op and state are on same device
        if isinstance(displacement_op, torch.Tensor):
            displacement_op = displacement_op.to(device)

        if use_ket_optimization and state.dim() == 1:
            # vector case
            result = torch.mv(displacement_op, state)
        else:
            # density matrix: new_rho = D rho D^†
            result = displacement_op @ state @ displacement_op.T.conj()
        return result
    except Exception as e:
        print(f"_apply_displacement exception: {e}", file=sys.stderr)
        return state


def _apply_squeezing(
    state: torch.Tensor,
    complex_param: complex,
    use_ket_optimization: bool,
    quantum_ops,
    device: torch.device,
) -> torch.Tensor:
    """Apply squeezing operation using quantum_ops.squeeze(complex_param/2)."""
    try:
        squeezing_op = quantum_ops.squeeze(complex_param / 2)
        if isinstance(squeezing_op, torch.Tensor):
            squeezing_op = squeezing_op.to(device)

        if use_ket_optimization and state.dim() == 1:
            result = torch.mv(squeezing_op, state)
        else:
            result = squeezing_op @ state @ squeezing_op.T.conj()
        return result
    except Exception as e:
        print(f"_apply_squeezing exception: {e}", file=sys.stderr)
        return state


def _apply_breeding(
    state: torch.Tensor,
    rounds: int,
    quantum_ops,
    projector: torch.Tensor,
    use_ket_optimization: bool,
    device: torch.device,
) -> torch.Tensor:
    """Apply breeding operation. For kets this may be simplified."""
    try:
        if use_ket_optimization and state.dim() == 1:
            # simplified no-op or custom ket-breeding code
            return state
        else:
            if hasattr(quantum_ops, "breeding_gpu"):
                return quantum_ops.breeding_gpu(rounds, state, projector)
            else:
                return state
    except Exception as e:
        print(f"_apply_breeding exception: {e}", file=sys.stderr)
        return state


def _calculate_operation_probability(op_type: int, param1: float, param2: float) -> float:
    """Calculate operation probability in a bounded, safe way."""
    op_type = int(np.round(op_type))
    try:
        if op_type == 0:
            return 1.0
        elif op_type == 1:  # displacement
            magnitude = np.hypot(param1, param2)
            # safe formula: decreasing probability with magnitude, clamp [0,1]
            val = 1.0 - (magnitude / (magnitude + 100.0))
            return float(max(0.0, min(1.0, val)))
        elif op_type == 2:  # squeezing
            magnitude = np.hypot(param1, param2)
            # mapping magnitude to [0,1] in a stable way
            val = 1.0 - (np.tanh(magnitude) * 0.5)
            return float(max(0.0, min(1.0, val)))
        elif op_type == 3:  # breeding
            return 0.95
        elif op_type == 4:
            return 0.5
        elif op_type == 5:
            return 0.4
    except Exception:
        pass
    return 0.0


def _calculate_sequence_probability_recursive(operations: List[Tuple[int, float, float, float]], initial_prob: float = 1.0) -> float:
    """
    Compute total sequence probability.

    operations: list of tuples (op_type, prob, p1, p2) in chronological order (oldest first)
    If a breeding op (type 3) is found, older ops are squared (preparation twice) as in original logic.
    """
    total = float(initial_prob)
    n = len(operations)
    for idx, (op, prob, p1, p2) in enumerate(operations):
        total *= float(prob)
        if op != 0:
            total *= 0.99
        if op == 3:
            # breeding: earlier ops (older than this op) must be prepared twice -> multiply by tail^2
            # tail corresponds to operations after idx (i.e., older? original code was confusing).
            # Here we take the remaining operations after the breeding as the 'earlier' ones.
            # compute tail product of probs for remaining ops
            tail = 1.0
            for (op2, prob2, _, _) in operations[idx + 1 :]:
                tail *= float(prob2)
                if op2 != 0:
                    tail *= 0.99
            total *= tail ** 2
            return total
    return total
