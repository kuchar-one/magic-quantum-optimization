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
    
    def _evaluate_sequential(self, x):
        """
        Sequential fallback evaluation (original implementation).
        
        Parameters
        ----------
        x : array_like
            2D array of operation sequences
            
        Returns
        -------
        np.ndarray
            Objective values
        """
        f = np.zeros((x.shape[0], 2))

        for i in range(x.shape[0]):
            operations = x[i].reshape(-1, 3)
            current_state = self.initial_state.clone()
            ops = []
            modified_sequence = False
            for op_idx in range(len(operations)):
                try:
                    op_type = int(np.round(operations[op_idx, 0]))
                    param1, param2 = operations[op_idx, 1:]

                    valid_op = self.validate_operation(current_state, op_type, ops)
                    if not valid_op["valid"]:
                        operations[op_idx, 0] = valid_op["new_op"]
                        op_type = valid_op["new_op"]
                        modified_sequence = True

                    new_state, _ = self.apply_operation(
                        current_state, op_type, param1, param2
                    )
                    
                    # Only check hermiticity for density matrices
                    if not self.use_ket_optimization and not self.is_hermitian(new_state):
                        logger.info(
                            f"State is not hermitian after operation {op_type}."
                        )
                        total_probability = 0
                        break

                    if torch.isfinite(torch.norm(new_state)):
                        current_state = new_state
                        op_probability = self.calculate_operation_probability(
                            op_type, param1, param2
                        )

                        if op_probability == 0:
                            total_probability = 0

                        ops.append((op_type, op_probability))
                    else:
                        total_probability = 0
                        break

                except Exception as e:
                    logger.error(f"Error in operation sequence: {e}")
                    import traceback
                    traceback.print_exc()
                    total_probability = 0
                    break

            if modified_sequence:
                x[i] = operations.flatten()

            reversed_ops = list(reversed(ops))
            total_probability = self.sequence_probability(reversed_ops)

            try:
                if total_probability != 0:
                    operator_expectation = self.calculate_operator_expectation(
                        current_state
                    )
                    if operator_expectation < self.groundstate_eigenvalue.item():
                        logger.error(f"We got bullshit value {operator_expectation} for state {current_state}, norm {torch.norm(current_state).item()}.")
                        if self.use_ket_optimization and len(current_state.shape) == 1:
                            # For kets, use vdot
                            operator_expectation_test = torch.real(torch.vdot(current_state, torch.mv(self.op, current_state))).item()
                        else:
                            # For density matrices, use trace
                            operator_expectation_test = torch.trace(torch.matmul(self.op, current_state)).item()
                        logger.error(f"Test value {operator_expectation_test}")
                        if not self.use_ket_optimization:
                            state_hermitianity = self.is_hermitian(current_state)
                            operator_hermitianity = self.is_hermitian(self.op)
                            logger.error(f"State hermitianity {state_hermitianity}, operator hermitianity {operator_hermitianity}") 
                        operator_expectation = np.inf
                else:
                    operator_expectation = np.inf
            except Exception as e:
                logger.error(f"Error calculating operator expectation: {e}")
                operator_expectation = np.inf  

            f[i] = [operator_expectation, -total_probability]
            
            # Clean up intermediate tensors (minimal performance impact)
            del current_state, operations, ops, reversed_ops
            
            # Force cleanup after every individual to prevent accumulation
            torch.cuda.empty_cache()
            
            # Check memory usage every 25 individuals (reduced frequency)
            if i % 25 == 0:
                check_memory_and_cleanup(memory_threshold_mb=1200, label=f"Individual{i}")

        return f

    def validate_operation(self, state, op_type, prev_ops):
        """
        Validate a quantum operation in the context of a sequence of operations.

        Parameters
        ----------
        state : torch.Tensor
            The current state of the quantum system.
        op_type : int
            The type of the operation to be validated.
        prev_ops : list
            A list of the previous operations in the sequence.

        Returns
        -------
        result : dict
            A dictionary containing the result of the validation.
            The dictionary contains the following keys:
            - valid : bool
                Whether the operation is valid in the context of the sequence.
            - new_op : int
                The new operation type to be used in the sequence.
        """
        result = {"valid": True, "new_op": op_type}

        if op_type == 4:  # Annihilation
            if self.use_ket_optimization and len(state.shape) == 1:
                # For kets, check if in ground state
                if torch.abs(state[0] - 1) < 1e-2:
                    result["valid"] = False
                    result["new_op"] = 0
                    return result
            else:
                # For density matrices, check diagonal element
                if torch.abs(state[0, 0] - 1) < 1e-2:
                    result["valid"] = False
                    result["new_op"] = 0
                    return result

            has_non_unitary = False
            for prev_op in prev_ops:
                if prev_op != 0 and prev_op != 4:
                    has_non_unitary = True
                    break

            if not has_non_unitary:
                result["valid"] = False
                result["new_op"] = 0
                return result

        elif op_type == 5:  # Creation
            if self.use_ket_optimization and len(state.shape) == 1:
                # For kets, check if in highest state
                if torch.abs(state[-1] - 1) < 1e-2:
                    result["valid"] = False
                    result["new_op"] = 0
                    return result
            else:
                # For density matrices, check diagonal element
                if torch.abs(state[-1, -1] - 1) < 1e-2:
                    result["valid"] = False
                    result["new_op"] = 0
                    return result

        return result

    def calculate_operation_probability(self, op_type, param1, param2):
        """
        Calculate the probability of an operation type given the parameters.

        Parameters
        ----------
        op_type : int
            The type of the operation.
        param1 : float
            The first parameter of the operation.
        param2 : float
            The second parameter of the operation.

        Returns
        -------
        probability : float
            The probability of the operation given the parameters.
        """

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
            rounds = 0.95
            return rounds
        elif op_type == 4:  # Annihilation
            return np.exp(2*self.gamma-1)
        elif op_type == 5:  # Creation
            return 0.5*np.exp(2*self.gamma-1)
        return 0.0

    def apply_operation(self, state, op_type, param1, param2):
        """
        Apply a quantum operation to the given state.

        Parameters
        ----------
        state : torch.Tensor
            The quantum state to which the operation should be applied.
        op_type : int
            The type of the operation to be applied.
        param1 : float
            The first parameter of the operation.
        param2 : float
            The second parameter of the operation.

        Returns
        -------
        new_state : torch.Tensor
            The state after the operation has been applied.
        new_op_type : int
            The type of the operation after it has been applied.
        """
        op_type = int(np.round(op_type))
        state = state.clone()

        if op_type == 0:  # Unitary
            return state, op_type
        elif op_type == 1:  # Displacement
            complex_param = param1 + 1j * param2
            state = self.apply_displacement(state, complex_param)
            return state / torch.norm(state), op_type
        elif op_type == 2:  # Squeezing
            complex_param = param1 + 1j * param2
            state = self.apply_squeezing(state, complex_param)
            return state / torch.norm(state), op_type
        elif op_type == 3:  # Breeding
            rounds = 1
            state = self.apply_breeding(state, rounds)
            return state / torch.norm(state), op_type
        elif op_type == 4:  # Annihilation
            if self.use_ket_optimization:
                state = self.apply_annihilation_ket(state)
            else:
                state = self.apply_photon_subtraction(state)
            return state / torch.norm(state), op_type
        elif op_type == 5:  # Creation
            if self.use_ket_optimization:
                state = self.apply_creation_ket(state)
            else:
                state = self.apply_photon_addition(state)
            return state / torch.norm(state), op_type

        return state, op_type

    def is_hermitian(self, matrix, tol=1e-3):
        """
        Check if a matrix is hermitian.

        Parameters
        ----------
        matrix : torch.Tensor
            The matrix to be checked.
        tol : float, default: 1e-3
            The tolerance of the check.

        Returns
        -------
        hermitian : bool
            Whether the matrix is hermitian.
        """
        conjugate_transpose = matrix.t().conj()
        difference = matrix - conjugate_transpose
        return torch.norm(difference) < tol

    def apply_displacement(self, state, complex_param):
        """
        Apply a displacement operation to a given state.

        Parameters
        ----------
        state : torch.Tensor
            The quantum state to which the displacement operation should be applied.
        complex_param : complex
            The complex parameter of the displacement operation.

        Returns
        -------
        new_state : torch.Tensor
            The state after the displacement operation has been applied.

        Raises
        ------
        ValueError
            If the displacement operation changes the norm of the state by more than 0.05.
        """
        try:
            displacement_op = self.helper_ops.displace(complex_param)
            
            if self.use_ket_optimization and len(state.shape) == 1:
                # Pure state (ket) operation
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
                # Density matrix operation
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
            
            norm = torch.norm(gpu_result).item()
            if np.abs(norm - 1) > 0.05:
                raise ValueError("Displacement broke norm.")
            return gpu_result
        except Exception as e:
            if e != "Displacement broke norm.":
                logger.info(f"Error applying displacement: {e}")
            return state

    def apply_squeezing(self, state, complex_param):
        """
        Apply a squeezing operation to a given state.

        Parameters
        ----------
        state : torch.Tensor
            The quantum state to which the squeezing operation should be applied.
        complex_param : complex
            The complex parameter of the squeezing operation.

        Returns
        -------
        new_state : torch.Tensor
            The state after the squeezing operation has been applied.

        Raises
        ------
        ValueError
            If the squeezing operation changes the norm of the state by more than 0.05.
        """

        try:
            squeezing_op = self.helper_ops.squeeze(complex_param / 2)
            
            if self.use_ket_optimization and len(state.shape) == 1:
                # Pure state (ket) operation
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
                # Density matrix operation
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
            
            norm = torch.norm(gpu_result).item()
            if np.abs(norm - 1) > 0.05:
                raise ValueError("Squeezing broke norm.")
            return gpu_result
        except Exception as e:
            if e != "Squeezing broke norm.":
                logger.info(f"Error applying squeezing: {e}")
            return state

    def apply_annihilation(self, state):
        """
        Apply an annihilation operation to a given state.

        Parameters
        ----------
        state : torch.Tensor
            The quantum state to which the annihilation operation should be applied.

        Returns
        -------
        new_state : torch.Tensor
            The state after the annihilation operation has been applied.

        Raises
        ------
        ValueError
            If the annihilation operation changes the norm of the state by more than 0.05.
        """
        try:
            annihilation_op = self.quantum_ops.d
            gpu_result = torch.matmul(
                torch.matmul(annihilation_op, state), annihilation_op.T.conj()
            )
            gpu_result = gpu_result / torch.norm(gpu_result)
            return gpu_result
        except Exception as e:
            logger.error(f"Error applying annihilation: {e}")
            return state

    def apply_creation(self, state):
        """
        Apply a creation operation to a given state.

        Parameters
        ----------
        state : torch.Tensor
            The quantum state to which the creation operation should be applied.

        Returns
        -------
        new_state : torch.Tensor
            The state after the creation operation has been applied.

        Raises
        ------
        Exception
            If there is an error during the application of the creation operation,
            it logs the error and returns the original state.
        """

        try:
            creation_op = self.quantum_ops.d.T.conj()
            gpu_result = torch.matmul(
                torch.matmul(creation_op, state), creation_op.T.conj()
            )
            gpu_result = gpu_result / torch.norm(gpu_result)
            return gpu_result
        except Exception as e:
            logger.error(f"Error applying creation: {e}")
            return state

    def apply_photon_subtraction(self, rho):
        """
        Approximate photon subtraction:
            N_-(γ)[ρ] = (e^{2γ}-1) E a ρ a† E
        which is a non‐trace‐preserving map (conditional on detecting 1 click).
        """
        K = self.subtraction_kraus
        new_rho = K @ rho @ K.T.conj()
        return new_rho

    def apply_photon_addition(self, rho):
        """
        Approximate photon addition:
            N_+(γ)[ρ] = (e^{2γ}-1) E a† ρ a E
        which is also non‐trace‐preserving (conditional on the parametric gain).
        """
        K = self.addition_kraus
        new_rho = K @ rho @ K.T.conj()
        return new_rho

    def apply_annihilation_ket(self, ket):
        """
        Apply annihilation operation to a pure state (ket).
        
        Parameters
        ----------
        ket : torch.Tensor
            The quantum state ket (1D tensor)
            
        Returns
        -------
        torch.Tensor
            The state after annihilation operation
        """
        try:
            annihilation_op = self.quantum_ops.d
            new_ket = torch.mv(annihilation_op, ket)
            return new_ket
        except Exception as e:
            logger.error(f"Error applying annihilation to ket: {e}")
            return ket

    def apply_creation_ket(self, ket):
        """
        Apply creation operation to a pure state (ket).
        
        Parameters
        ----------
        ket : torch.Tensor
            The quantum state ket (1D tensor)
            
        Returns
        -------
        torch.Tensor
            The state after creation operation
        """
        try:
            creation_op = self.quantum_ops.d.T.conj()
            new_ket = torch.mv(creation_op, ket)
            return new_ket
        except Exception as e:
            logger.error(f"Error applying creation to ket: {e}")
            return ket
    
    def apply_breeding(self, state, rounds):
        """
        Applies the breeding operation to the given state and returns the result.

        Parameters
        ----------
        state : torch.Tensor
            The quantum state to which the breeding operation should be applied.
        rounds : int
            The number of rounds of breeding to apply.

        Returns
        -------
        torch.Tensor
            The state after the breeding operation has been applied.

        Raises
        ------
        ValueError
            If the breeding operation changes the norm of the state by more than 0.05.
        """
        try:
            logger.info(f"apply_breeding: State device before breeding: {state.device}")
            
            if self.use_ket_optimization and len(state.shape) == 1:
                # For ket-based optimization, we need a different approach
                # Since breeding is a complex operation involving measurements,
                # we'll use a simplified version for kets
                if rounds == 0:
                    result = state
                else:
                    # For now, apply a simple operation that preserves the ket nature
                    # This is a simplified version - in practice, breeding with kets
                    # would require more sophisticated handling
                    result = state  # Placeholder - could apply some unitary operation
            else:
                # Density matrix optimization
                result = self.quantum_ops.breeding_gpu(rounds, state, self.projector)
            
            logger.info(f"apply_breeding: State device after breeding: {result.device}")
            return result
        except Exception as e:
            logger.error(f"Error applying breeding: {e}")
            return state

    def calculate_operator_expectation(self, state):
        """
        Calculate the expectation value of the operator with respect to a given quantum state.

        Parameters
        ----------
        state : torch.Tensor
            The quantum state for which the expectation value of the operator is calculated.

        Returns
        -------
        float
            The real part of the expectation value as a NumPy array.

        Raises
        ------
        Exception
            If there is an error in calculating the expectation value, logs the error and returns infinity.
        """

        try:
            return torch.real(self.quantum_ops.expect(self.op, state)).cpu().numpy()
        except Exception as e:
            logger.error(f"Error calculating operator expectation: {e}")
            return np.inf


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
