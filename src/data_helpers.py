import numpy as np
import json
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import qutip as qt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from joblib import Parallel, delayed
from src.qutip_quantum_ops import breeding, p0_projector


def get_next_version_folder(base_folder: str) -> str:
    """
    Get the next version folder by incrementing the version number.
    """
    version = 1
    while os.path.exists(f"{base_folder}_v{version}"):
        version += 1
    return f"{base_folder}_v{version}"


def analyze_sequence_result(result):
    """
    Analyze the result of the quantum operation sequence optimization.

    Parameters
    ----------
    result : pymoo.result.Result
        Result of the optimization, containing the Pareto front
        solutions and their objectives.

    Returns
    -------
        A dictionary containing various metrics of the optimization
    X = result.X
    # Recover probabilities
        the range of the Pareto front, the minimum and maximum
        values of the objectives, and the operation sequences for
        each solution.
    """
    # Get the Pareto front solutions
    F = result.F
    X = result.X
    # Recover probabilities
    F[:, 1] = -F[:, 1]

    # Sort points by operator expectation (first objective)
    sorted_indices = np.argsort(F[:, 0])
    sorted_F = F[sorted_indices]
    sorted_X = X[sorted_indices]

    # Create interpolation function for the boundary
    from scipy.interpolate import interp1d

    f_boundary = interp1d(
        sorted_F[:, 0],
        sorted_F[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=(np.nan, np.nan),
    )

    # Parse operation sequences for each solution
    operation_sequences = [parse_operation_sequence(x) for x in sorted_X]

    metrics = {
        # Objective space information
        "min_expectation": float(np.min(F[:, 0])),
        "max_expectation": float(np.max(F[:, 0])),
        "min_probability": float(np.min(F[:, 1])),
        "max_probability": float(np.max(F[:, 1])),
        # Number of Pareto optimal solutions
        "n_solutions": int(len(F)),
        # Range of the Pareto front
        "expectation_range": float(np.max(F[:, 0]) - np.min(F[:, 0])),
        "probability_range": float(np.max(F[:, 1]) - np.min(F[:, 1])),
        # Original results
        "X": sorted_X.tolist(),
        "F": sorted_F.tolist(),
        # Operation sequences
        "operation_sequences": operation_sequences,
    }

    return metrics


def parse_operation_sequence(params):
    """
    Parse the given parameters into a sequence of operations.

    Parameters
    ----------
    params : list
        A list of parameters, where each triplet of parameters
        describes an operation.

    Returns
    -------
    operations : list
        A list of dictionaries, where each dictionary describes
        an operation. The dictionary contains the type of the
        operation, the two parameters of the operation, a human-readable
        description of the operation, and a boolean indicating whether
        the operation was actually applied.
    """
    sequence_length = len(params) // 3
    operations = []

    for i in range(sequence_length):
        op_type = int(np.round(params[i * 3]))
        param1 = float(params[i * 3 + 1])
        param2 = float(params[i * 3 + 2])
        valid = True
        non_unitary = False

        op_desc = {
            0: "Unitary",
            1: "Displacement",
            2: "Squeezing",
            3: "Breeding",
            4: "Subtraction",
            5: "Addition",
        }.get(op_type, "Unknown")

        param_desc = ""
        if op_type == 1:
            param_desc = f"({param1:.3f} + {param2:.3f}j)"
        elif op_type == 2:
            param_desc = f"({param1/2:.3f} + {param2/2:.3f}j)"
        elif op_type == 3:
            rounds = max(1, min(5, int(np.round(param1))))
            param_desc = f"rounds={rounds}"
        else:
            param_desc = None

        if param_desc is not None:
            description = f"{op_desc}: {param_desc}"
        else:
            description = op_desc

        if op_type == 4:
            for j in range(0, i):
                if (
                    int(np.round(params[j * 3])) != 0
                    and int(np.round(params[j * 3])) != 4
                ):
                    non_unitary = True

            if not non_unitary:
                valid = False

        operations.append(
            {
                "type": op_desc,
                "type_id": op_type,
                "param1": param1,
                "param2": param2,
                "description": description,
                "valid": valid,  # Track if operation was actually applied
            }
        )

    return operations


def create_sequence_optimization_animation(F_history, filename):
    """
    Creates an animation of the quantum sequence optimization process.

    This function generates an animation that visualizes the evolution
    of operator expectations and total probabilities over generations
    during the optimization process. The animation is saved to the
    specified file.

    Parameters
    ----------
    F_history : list of numpy.ndarray
        A history of Pareto fronts, where each element corresponds to
        a generation and contains an array of objective values.
    filename : str
        The name of the file to which the animation will be saved.
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        ax.clear()
        F = F_history[frame]
        ax.scatter(F[:, 0], -F[:, 1], c="blue", alpha=0.6)
        ax.set_xlabel("Operator Expectation")
        ax.set_ylabel("Total probability")
        ax.set_title(f"Quantum Sequence Optimization (Generation {frame + 1})")
        ax.grid(True, alpha=0.3)

    anim = FuncAnimation(fig, update, frames=len(F_history), repeat=False)
    anim.save(filename, writer="ffmpeg", fps=30)
    plt.close()


def save_sequence_metrics(metrics, filename):
    """
    Save the quantum sequence optimization metrics to a JSON file.

    Parameters
    ----------
    metrics : dict
        A dictionary containing the metrics of the optimization to be saved.
    filename : str
        The name of the file to which the metrics will be saved. If the
        filename does not end with '.json', it will be automatically appended.

    """

    if not filename.endswith(".json"):
        filename += ".json"

    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Sequence metrics saved to {filename}")


def load_sequence_metrics(filename):
    """
    Load the quantum sequence optimization metrics from a JSON file.

    Parameters
    ----------
    filename : str
        The name of the file from which the metrics will be loaded. If the
        filename does not end with '.json', it will be automatically appended.

    Returns
    -------
    metrics : dict
        A dictionary containing the metrics of the optimization loaded from the
        file.

    """
    if not filename.endswith(".json"):
        filename += ".json"

    with open(filename, "r") as f:
        metrics = json.load(f)

    # Convert lists back to numpy arrays where needed
    metrics["X"] = np.array(metrics["X"])
    metrics["F"] = np.array(metrics["F"])

    return metrics


def format_sequence_report(metrics, solution_index=None):
    """
    Format the quantum sequence optimization metrics into a string report.

    Parameters
    ----------
    metrics : dict
        A dictionary containing the metrics of the optimization.
    solution_index : int, optional
        If provided, the detailed operation sequence for the Pareto optimal
        solution with the given index will be included in the report.

    Returns
    -------
    report : str
        A string containing the formatted report.

    """
    report = []
    report.append("Quantum Operation Sequence Optimization Report")
    report.append("=" * 50)

    report.append(f"\nNumber of Pareto optimal solutions: {metrics['n_solutions']}")
    report.append("\nObjective Ranges:")
    report.append(
        f"Operator Expectation: [{metrics['min_expectation']:.4f}, {metrics['max_expectation']:.4f}]"
    )
    report.append(
        f"Total probability: [{metrics['min_probability']:.4f}, {metrics['max_probability']:.4f}]"
    )

    if solution_index is not None:
        report.append(f"\nDetailed Operation Sequence for Solution {solution_index}:")
        sequence = metrics["operation_sequences"][solution_index]
        for i, op in enumerate(sequence, 1):
            report.append(f"{i}. {op['description']}")

    return "\n".join(report)


def create_boundary_state_animation(
    initial_state, filename, N, save_as="animation.mp4", blend=False
):
    """
    Create an animation of the quantum sequence optimization process.

    Parameters
    ----------
    initial_state : Qobj
        The initial state of the system.
    filename : str
        The filename containing the optimization metrics.
    N : int
        The number of Fock states in the Hilbert space.
    save_as : str, optional
        The filename to save the animation to. Defaults to 'animation.mp4'.
    blend : bool, optional
        Whether to blend the intermediate states. Defaults to False.

    Notes
    -----
    This function creates an animation of the quantum sequence optimization process.
    The animation shows the initial and final Wigner functions of the system, as
    well as the intermediate states and the operations applied to achieve the
    final state. The animation is saved to the specified filename.

    """
    a = qt.destroy(N)
    adag = qt.create(N)
    gamma = 0.1
    damping_operator = (-gamma * qt.num(N)).expm()
    subtraction_kraus = np.sqrt(np.exp(2*gamma) - 1.0) * (damping_operator * a)
    addition_kraus = np.sqrt(np.exp(2*gamma) - 1.0) * (damping_operator * adag)
    
    def format_operation(op):
        """
        Format a quantum operation into a string.

        Parameters
        ----------
        op : dict
            A dictionary containing the operation information.

        Returns
        -------
        str
            A string representation of the operation.
        """
        if not op["valid"]:
            return ""

        if op["type"] == "Displacement":
            return f"{op['type']}({op['param1']:.2f} + {op['param2']:.2f}i)"
        elif op["type"] == "Squeezing":
            return f"Sq.({op['param1']/2:.2f} + {op['param2']/2:.2f}i), ({np.round(10 * np.log10(np.exp(2 * (op['param1']**2 + op['param2']**2)**0.5)))} dB)"
        elif op["type"] == "Breeding":
            return f"{op['type']}"
        elif op["type"] == "Unitary":
            return ""
        else:
            return op["type"]

    def apply_operations_anim(state, operations):
        """
        Apply a sequence of quantum operations to an initial state.

        This function iterates over a list of operations and applies each valid
        operation to the provided initial quantum state. The operations include
        displacement, squeezing, annihilation, creation, and breeding. The function
        handles both ket and operator forms of the quantum state and ensures that
        the resulting state is normalized.

        Parameters
        ----------
        initial_state : Qobj
            The initial quantum state, which can be in ket or operator form.
        operations : list of dict
            A list of operations to apply. Each operation is a dictionary containing
            the operation type, parameters, and a validity flag.

        Returns
        -------
        Qobj
            The quantum state after all operations have been applied.


        ------
        ValueError
            If the state type is neither ket nor operator.
        """

        for op in operations:
            if op["type"] == "Displacement" and op["valid"] == True:
                # print(f"Applying displacement: {op['param1']}, {op['param2']}")
                alpha = complex(op["param1"], op["param2"])
                if state.isket:
                    state = (qt.displace(N, alpha) * state).unit()
                elif state.isoper:
                    state = (
                        qt.displace(N, alpha) * state * qt.displace(N, alpha).dag()
                    ).unit()
                else:
                    raise ValueError("Invalid state type")
            elif op["type"] == "Squeezing" and op["valid"] == True:
                # print(f"Applying squeezing: {op['param1']}, {op['param2']}")
                z = complex(op["param1"], op["param2"])
                if state.isket:
                    state = (qt.squeeze(N, z / 2) * state).unit()
                elif state.isoper:
                    state = (
                        qt.squeeze(N, z / 2) * state * qt.squeeze(N, z / 2).dag()
                    ).unit()
                else:
                    raise ValueError("Invalid state type")
            elif op["type"] == "Subtraction" and op["valid"] == True:
                """# print("Applying annihilation")
                if state.isket:
                    try:
                        state = (a * state).unit()
                    except Exception as e:
                        print(f"Error applying annihilation: {e}")
                        print(f"State: {state}")
                        print(f"Validity: {op['valid']}")
                elif state.isoper:
                    try:
                        state = (a * state * adag).unit()
                    except Exception as e:
                        print(f"Error applying annihilation: {e}")
                        print(f"State: {state}")
                        print(f"Validity: {op['valid']}")
                        print(f"Operation sequence: {operations}")
                else:
                    raise ValueError("Invalid state type")"""
                if state.isket:
                    state = qt.ket2dm(state)
                state = (subtraction_kraus * state * subtraction_kraus.dag()).unit()
                
            elif op["type"] == "Addition" and op["valid"] == True:
                """# print("Applying creation")
                if state.isket:
                    state = (adag * state).unit()
                elif state.isoper:
                    state = (adag * state * a).unit()
                else:
                    raise ValueError("Invalid state type")"""
                if state.isket:
                    state = qt.ket2dm(state)
                state = (addition_kraus * state * addition_kraus.dag()).unit()
                
            elif op["type"] == "Breeding" and op["valid"] == True:
                rounds = 1
                # print(f"Applying breeding: {rounds} rounds")
                state = breeding(N, rounds, state, projector).unit()
        return state

    metrics = json.load(open(filename, "r"))

    # Define the operation sequences
    operation_sequences = metrics["operation_sequences"]
    objectives = [tup for tup in metrics["F"]]

    sorted_metrics = [
        tup for _, tup in sorted(enumerate(objectives), key=lambda x: x[1][0])
    ]
    sort_indices = [i for i, _ in sorted(enumerate(objectives), key=lambda x: x[1][0])]

    operation_sequences = [operation_sequences[i] for i in sort_indices]

    projector = p0_projector(N)

    def compute_intermediate_states(seq, initial_state, N):
        """
        Compute the intermediate states in a sequence of operations.

        Parameters
        ----------
        seq : list
            List of operations in the sequence.
        initial_state : Qobj
            Initial quantum state.
        N : int
            Dimension of the Hilbert space.

        Returns
        -------
        list
            List of intermediate states, starting with the initial state, and
            ending with the final state.
        """
        states = [initial_state]
        current_state = initial_state
        for op in seq:
            if format_operation(op):  # Only apply valid operations
                try:
                    current_state = apply_operations_anim(states[-1], [op])
                    states.append(current_state)
                except Exception as e:
                    print(
                        f"Error applying operation: {e}, state {states[-1]}, op {op} in sequence {seq}."
                    )
        return states

    print(f"Filename = {save_as}, animating {len(operation_sequences)} sequences...")
    intermediate_states = Parallel(n_jobs=-1)(
        delayed(compute_intermediate_states)(seq, initial_state, N)
        for seq in tqdm(operation_sequences, desc="Precomputing intermediate states...")
    )

    # Calculate total frames (sum of operations per sequence + 1 for initial state per sequence)
    total_frames = sum(len(states) for states in intermediate_states)

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 2])

    # Create subplots
    ax_initial = fig.add_subplot(gs[0, 0])
    ax_final = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, 0])
    ax_pareto = fig.add_subplot(gs[1, 1])

    xvec = np.linspace(-8, 8, 100)
    yvec = np.linspace(-8, 8, 100)

    cmap = plt.colormaps.get_cmap("inferno")

    class PlateauTwoSlopeNorm(colors.TwoSlopeNorm):
        def __init__(self, vcenter, plateau_size, vmin=None, vmax=None):
            """
            A modified TwoSlopeNorm that maintains a constant color within
            a specified range around vcenter before transitioning to the endpoints.

            Parameters
            ----------
            vcenter : float
                The central value that defines the plateau midpoint
            plateau_size : float
                The total width of the plateau region (vcenter ± plateau_size/2)
            vmin : float, optional
                The minimum value in the normalization
            vmax : float, optional
                The maximum value in the normalization
            """
            super().__init__(vcenter=vcenter, vmin=vmin, vmax=vmax)
            self.plateau_size = plateau_size

        def __call__(self, value, clip=None):
            """
            Map values to the interval [0, 1], maintaining a constant value
            within the plateau region.
            """
            result, is_scalar = self.process_value(value)
            self.autoscale_None(result)

            if not self.vmin <= self.vcenter <= self.vmax:
                raise ValueError("vmin, vcenter, vmax must increase monotonically")

            # Define plateau boundaries
            plateau_lower = self.vcenter - self.plateau_size / 2
            plateau_upper = self.vcenter + self.plateau_size / 2

            # Create interpolation points including plateau region
            x_points = [self.vmin, plateau_lower, plateau_upper, self.vmax]
            y_points = [0, 0.5, 0.5, 1]

            result = np.ma.masked_array(
                np.interp(result, x_points, y_points, left=-np.inf, right=np.inf),
                mask=np.ma.getmask(result),
            )

            if is_scalar:
                result = np.atleast_1d(result)[0]
            return result

        def inverse(self, value):
            if not self.scaled():
                raise ValueError("Not invertible until both vmin and vmax are set")

            plateau_lower = self.vcenter - self.plateau_size / 2
            plateau_upper = self.vcenter + self.plateau_size / 2

            x_points = [0, 0.5, 0.5, 1]
            y_points = [self.vmin, plateau_lower, plateau_upper, self.vmax]

            return np.interp(value, x_points, y_points, left=-np.inf, right=np.inf)

    norm = PlateauTwoSlopeNorm(vcenter=0, plateau_size=0.03, vmin=-0.23, vmax=0.23)

    # Pre-calculate Pareto front metrics
    metric1_values = [m[0] for m in sorted_metrics]
    metric2_values = [m[1] for m in sorted_metrics]

    def update(frame):
        """
        Update the visualization for a given animation frame.

        This function updates the plots of Wigner functions and Pareto front
        for a specific frame during the animation of quantum operations.

        Parameters
        ----------
        frame : int
            The current frame number in the animation sequence.

        Returns
        -------
        tuple
            A tuple containing the contour plots for initial, final, and
            current Wigner functions.
        """

        cumulative_frames = np.cumsum([len(states) for states in intermediate_states])
        seq_index = np.searchsorted(cumulative_frames, frame, side="right")
        step_index = frame - (cumulative_frames[seq_index - 1] if seq_index > 0 else 0)

        # Get the current sequence, state, and metrics
        sequence = operation_sequences[seq_index]
        states = intermediate_states[seq_index]
        current_state = states[step_index]
        final_state = states[-1]
        current_metric1 = sorted_metrics[seq_index][0]
        current_metric2 = sorted_metrics[seq_index][1]

        # Clear previous plots
        ax_initial.clear()
        ax_final.clear()
        ax_text.clear()
        ax_pareto.clear()

        # Calculate Wigner functions
        W_initial = qt.wigner(initial_state, xvec, yvec)
        W_final = qt.wigner(final_state, xvec, yvec)
        W_current = qt.wigner(current_state, xvec, yvec)

        # Plot initial Wigner function
        cont1 = ax_initial.contourf(xvec, yvec, W_initial, 100, cmap=cmap, norm=norm)
        ax_initial.set_title("Initial Wigner Function")
        ax_initial.grid(False)

        # Plot final Wigner function (always show the final state here)
        cont2 = ax_final.contourf(xvec, yvec, W_final, 100, cmap=cmap, norm=norm)
        ax_final.set_title("Final Wigner Function (Static)")
        ax_final.grid(False)

        # Display operations list and intermediate states in the bottom left
        operations_text = "Applied Operations:\n"
        k = 1
        for _, op in enumerate(sequence, 1):
            op_str = format_operation(op)
            if op_str:
                operations_text += f"{k}. {op_str}\n"
                k += 1

        operations_text += "\nCurrent Progress:\n"
        operations_text += f"Step {step_index}/{k-1}\n"

        num_lines = len(operations_text.split("\n"))
        mid = num_lines // 2
        col1_text = "\n".join(operations_text.split("\n")[:mid])
        col2_text = "\n".join(operations_text.split("\n")[mid:])

        ax_text.text(
            0.05,
            0.95,
            col1_text,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=8,
        )
        ax_text.text(
            0.55,
            0.95,
            col2_text,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=8,
        )

        W_subplot = ax_text.inset_axes([0.05, 0.1, 0.9, 0.6])
        cont3 = W_subplot.contourf(xvec, yvec, W_current, 100, cmap=cmap, norm=norm)
        W_subplot.set_title(f"Intermediate State (Step {step_index})", fontsize=9)
        W_subplot.grid(False)

        ax_text.axis("off")

        # Plot Pareto front
        ax_pareto.scatter(metric1_values, metric2_values, c="blue", alpha=0.5, s=20)
        ax_pareto.scatter(current_metric1, current_metric2, c="red", s=100)
        ax_pareto.set_xlabel("<op>")
        ax_pareto.set_ylabel("probability")
        ax_pareto.set_yscale("log")
        ax_pareto.grid(True)
        ax_pareto.set_title("Pareto Front")

        plt.tight_layout()
        return (cont1, cont2, cont3)

    # Animate with progress bar
    with tqdm(total=total_frames, desc="Generating animation") as pbar:

        def update_with_progress(frame):
            """
            Helper function to update animation with progress bar.

            Parameters
            ----------
            frame : int
                The current frame number.

            Returns
            -------
            result : tuple
                The result of calling the update function.
            """
            result = update(frame)
            pbar.update(1)
            return result

        ani = FuncAnimation(
            fig, update_with_progress, frames=total_frames, interval=200, blit=False
        )

        # Save animation
        if save_as.endswith(".gif"):
            ani.save(save_as, writer="pillow")
        elif save_as.endswith(".mp4"):
            ani.save(save_as, writer="ffmpeg", fps=5)
        else:
            raise ValueError(
                "save_as must be either 'animation.gif' or 'animation.mp4'"
            )

    plt.close()
