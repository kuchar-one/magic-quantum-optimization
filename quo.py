from src.cuda_helpers import set_gpu_device
from src.data_helpers import (
    create_sequence_optimization_animation,
    analyze_sequence_result,
    save_sequence_metrics,
    format_sequence_report,
    create_boundary_state_animation,
    load_sequence_metrics,
    get_next_version_folder,
)
from src.nsga_ii import run_quantum_sequence_optimization
from src.qutip_quantum_ops import construct_initial_state, construct_operator
from src.utils import parse_target_superposition
import argparse
from numpy import round as np_round
import os


def main() -> int:
    """
    Main function to run the quantum operation sequence optimization.

    This function sets up the necessary directories, parses command-line arguments,
    constructs the initial state and operator, runs the optimization, and saves the results.
    """
    for path in ["output", "output/animation", "output/metrics", "cache/operators"]:
        os.makedirs(path, exist_ok=True)
    parser = argparse.ArgumentParser(
        description="Quantum Operation Sequence Optimization"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--N", type=int, default=30, help="Dimension of Fock space")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=10,
        help="Number of operations in sequence",
    )
    parser.add_argument("--pop_size", type=int, default=100, help="Population size")
    parser.add_argument(
        "--max_generations", type=int, default=100, help="Maximum number of generations"
    )
    parser.add_argument(
        "--initial_state",
        type=str,
        default="vacuum",
        help='Initial state for optimization: "vacuum", "fock", "coherent", "squeezed", "displaced", "cat", "squeezed_cat", or integer (Fock state with n photons)',
    )

    parser.add_argument(
        "--initial_state_params",
        type=complex,
        nargs="+",
        default=0.0,
        help='Parameters for the initial state. For "fock": photon number (integer). For other states: complex numbers, e.g., "1.0" or "1.0+2.0j"',
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-3, help="Tolerance for convergence"
    )
    parser.add_argument(
        "--target_superposition",
        type=str,
        nargs="+",
        required=True,
        help="Superposition coefficients for the target superposition (required)",
    )
    parser.add_argument(
        "--animate_only", action="store_true", help="Only create animation"
    )
    parser.add_argument(
        "--continuation", action="store_true", help="Continue optimization from existing metrics"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="nsga2",
        choices=["nsga2", "nsga3", "moead", "age", "age2", "rvea", "smsemoa", "ctaea", "unsga3", "rnsga2", "rnsga3"],
        help="Optimization algorithm to use (default: nsga2)",
    )
    args = parser.parse_args()

    # Parse target_superposition tokens into complex coefficients
    if getattr(args, "target_superposition", None) is not None:
        args.target_superposition = parse_target_superposition(args.target_superposition)

    try:
        # Create a string representation of the target superposition for the folder name
        target_str = "_".join([str(c) for c in args.target_superposition])[:50]  # Limit length
        base_folder = f"output/target_{target_str}_N={args.N}_sequence_length={args.sequence_length}_initial_state={args.initial_state}_pop_size={args.pop_size}_max_gen={args.max_generations}_tol={args.tolerance}_alg={args.algorithm}"
        if args.continuation:
            folder = get_next_version_folder(base_folder)
        else:
            folder = base_folder

        if not os.path.exists(folder):
            os.makedirs(folder)
        param = (
            args.initial_state_params
            if args.initial_state_params != list
            else args.initial_state_params[0]
        )
        if not args.animate_only:
            try:
                set_gpu_device(args.gpu)
                initial_state = construct_initial_state(
                    N=args.N, desc=args.initial_state, param=param
                )
            except Exception as e:
                print(f"Error constructing initial state: {e}")
                return 1

            try:
                operator = construct_operator(args.N, args.target_superposition)
                print(f"Constructed operator from target_superposition: {args.target_superposition}")
            except Exception as e:
                print(f"Error constructing operator: {e}")
                import traceback
                traceback.print_exc()
                return 1

            if args.continuation:
                metrics_filename = f"{base_folder}/sequence_optimization_metrics.json"
                if os.path.isfile(metrics_filename):
                    metrics = load_sequence_metrics(metrics_filename)
                    initial_population = metrics["X"]
                else:
                    print(f"Metrics file {metrics_filename} not found.")
                    return 1
            else:
                initial_population = None

            result, F_history = run_quantum_sequence_optimization(
                initial_state=initial_state[0],
                initial_probability=initial_state[1],
                operator=operator,
                N=args.N,
                sequence_length=args.sequence_length,
                pop_size=args.pop_size,
                max_generations=args.max_generations,
                device_id=args.gpu,
                tolerance=args.tolerance,
                initial_population=initial_population,
                algorithm=args.algorithm,
                use_ket_optimization=True,
            )
            # Create animation and save results
            create_sequence_optimization_animation(
                F_history,
                f"{folder}/sequence_optimization_animation.mp4",
            )
            metrics = analyze_sequence_result(result)
            save_sequence_metrics(
                metrics,
                f"{folder}/sequence_optimization_metrics.json",
            )
            print(format_sequence_report(metrics, solution_index=0))
        initial_state = construct_initial_state(
            np_round(1.5 * args.N).astype(int), desc=args.initial_state, param=param
        )
        create_boundary_state_animation(
            initial_state=initial_state[0],
            N=np_round(1.5 * args.N).astype(int),
            filename=f"{folder}/sequence_optimization_metrics.json",
            save_as=f"{folder}/boundary_state_animation.mp4",
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())