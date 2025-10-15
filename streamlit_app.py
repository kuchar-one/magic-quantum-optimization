import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import sys
import os
import io
import base64
import json
import re
import threading
from io import StringIO
from PIL import Image
import queue

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.qutip_quantum_ops import construct_operator, construct_initial_state
from src.nsga_ii import run_quantum_sequence_optimization, QuantumOperationSequence, OptimizationCallback
from src.data_helpers import analyze_sequence_result, format_sequence_report, create_sequence_optimization_animation, create_boundary_state_animation, save_sequence_metrics, get_next_version_folder
from src.cuda_helpers import aggressive_memory_cleanup, print_memory_status, check_memory_and_cleanup
import qutip as qt
from pymoo.core.callback import Callback



def save_streamlit_optimization_results(result, F_history, algorithm, N, sequence_length, pop_size, max_generations, tolerance, target_superposition, initial_state, initial_state_param=0.0):
    """
    Save optimization results with versioning, similar to quo.py
    
    Parameters
    ----------
    result : pymoo result object
        Optimization result
    F_history : list
        History of F values
    algorithm : str
        Algorithm name
    N, sequence_length, pop_size, max_generations, tolerance : int/float
        Optimization parameters
    target_superposition : tuple
        Target superposition parameters
    initial_state : str
        Initial state type
        
    Returns
    -------
    str
        Path to the saved results folder
    """
    # Create base folder name (same format as quo.py)
    target_str = "_".join([str(c) for c in target_superposition])[:50]  # Limit length
    base_folder = f"output/target_{target_str}_N={N}_sequence_length={sequence_length}_initial_state={initial_state}_pop_size={pop_size}_max_gen={max_generations}_tol={tolerance}_alg={algorithm}"
    
    # Get next version folder
    folder = get_next_version_folder(base_folder)
    
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Save metrics
    metrics = analyze_sequence_result(result)
    save_sequence_metrics(metrics, f"{folder}/sequence_optimization_metrics.json")
    
    # Create animations
    create_sequence_optimization_animation(F_history, f"{folder}/sequence_optimization_animation.mp4")
    
    # Create boundary state animation (this is what the Streamlit app displays)
    try:
        from src.qutip_quantum_ops import construct_initial_state
        import numpy as np
        
        # Create initial state for animation (use larger dimension to avoid truncation)
        animation_N = int(np.round(1.5 * N))
        initial_state_qobj, _ = construct_initial_state(animation_N, initial_state, initial_state_param)
        
        create_boundary_state_animation(
            initial_state=initial_state_qobj,
            N=animation_N,
            filename=f"{folder}/sequence_optimization_metrics.json",
            save_as=f"{folder}/boundary_state_animation.mp4"
        )
    except Exception as e:
        print(f"Warning: Could not create boundary state animation: {e}")
    
    return folder

def main():
    st.set_page_config(
        page_title="Magic Quantum Sequence Optimization",
        page_icon="🔮",
        layout="wide"
    )
    
    # Initialize session state
    if 'operator' not in st.session_state:
        st.session_state.operator = None
    if 'ground_state' not in st.session_state:
        st.session_state.ground_state = None
    if 'optimization_running' not in st.session_state:
        st.session_state.optimization_running = False
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    
    # Add custom CSS for styling
    st.markdown("""
    <style>
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }

    .progress-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔮 Magic Quantum Sequence Optimization")
    st.markdown("Optimize quantum operation sequences to prepare target superposition states using breeding operations.")
    
    # Sidebar for parameters
    st.sidebar.header("Optimization Parameters")
    
    # Algorithm selection (independent from parameter presets)
    st.sidebar.markdown("#### 🧬 Algorithm Selection")
    algorithm = st.sidebar.selectbox(
        "Optimization Algorithm",
        ["nsga2", "nsga3", "moead", "age", "age2", "rvea", "smsemoa", "ctaea", "unsga3", "rnsga2", "rnsga3"],
        index=0,
        help="Choose the multi-objective optimization algorithm"
    )
    
    # Parameter presets based on benchmark results
    st.sidebar.markdown("#### 📊 Algorithm Parameter Presets")
    
    # Define parameter configurations based on benchmark results
    parameter_presets = {
        "Custom": {
            "description": "Set all parameters manually below",
            "crossover_eta": None, "crossover_prob": None,
            "mutation_eta": None, "mutation_prob": None
        },
        "🎯 Best Quality (Baseline)": {
            "description": "Best overall performance: -0.674970 expectation (UNSGA3), -0.592684 (AGE)",
            "crossover_eta": 1.0, "crossover_prob": 0.8,
            "mutation_eta": 3.0, "mutation_prob": 0.95
        },
        "⚡ Fastest (Fast Convergence)": {
            "description": "Fastest convergence: 35.94s (SMSEMOA), 6.05s (AGE2)",
            "crossover_eta": 2.0, "crossover_prob": 0.85,
            "mutation_eta": 5.0, "mutation_prob": 0.9
        },
        "🚀 Aggressive (High Exploration)": {
            "description": "Maximum exploration: High mutation/crossover rates",
            "crossover_eta": 0.5, "crossover_prob": 0.95,
            "mutation_eta": 1.0, "mutation_prob": 0.99
        },
        "🔬 High Diversity": {
            "description": "Maximum diversity: Extreme exploration parameters",
            "crossover_eta": 0.3, "crossover_prob": 0.9,
            "mutation_eta": 0.5, "mutation_prob": 0.98
        },
        "📚 Literature Standard": {
            "description": "Common values from NSGA-II research papers",
            "crossover_eta": 2.0, "crossover_prob": 0.9,
            "mutation_eta": 5.0, "mutation_prob": 0.1
        },
        "🔄 High Crossover": {
            "description": "Emphasizes recombination over mutation",
            "crossover_eta": 1.5, "crossover_prob": 0.95,
            "mutation_eta": 20.0, "mutation_prob": 0.7
        },
        "🧬 High Mutation": {
            "description": "Emphasizes exploration through mutation",
            "crossover_eta": 5.0, "crossover_prob": 0.7,
            "mutation_eta": 2.0, "mutation_prob": 0.95
        }
    }
    
    parameter_preset = st.sidebar.selectbox("Parameter Configuration", list(parameter_presets.keys()), index=1)
    
    if parameter_preset != "Custom":
        preset = parameter_presets[parameter_preset]
        st.sidebar.info(f"**{preset['description']}**")
        
        # Set preset values
        preset_crossover_eta = preset["crossover_eta"]
        preset_crossover_prob = preset["crossover_prob"]
        preset_mutation_eta = preset["mutation_eta"]
        preset_mutation_prob = preset["mutation_prob"]
        
        # Show benchmark performance info
        if parameter_preset == "🎯 Best Quality (Baseline)":
            st.sidebar.success("✅ **Benchmark Results:**\n- Magic: -0.674970 (UNSGA3)\n- Stateprep: -0.592684 (AGE)")
        elif parameter_preset == "⚡ Fastest (Fast Convergence)":
            st.sidebar.success("✅ **Benchmark Results:**\n- Magic: 35.94s (SMSEMOA)\n- Stateprep: 6.05s (AGE2)")
        
        # Disable manual controls when preset is selected
        st.sidebar.markdown("*Parameters set by preset*")
    else:
        preset_crossover_eta = None
        preset_crossover_prob = None
        preset_mutation_eta = None
        preset_mutation_prob = None
    
    # Physical parameters
    st.sidebar.markdown("#### 🔬 Physical Parameters")
    N = st.sidebar.slider("Hilbert Space Dimension (N)", 5, 50, 20, 5)
    sequence_length = st.sidebar.slider("Sequence Length", 3, 20, 10, 1)
    
    # Optimization parameters
    st.sidebar.markdown("#### ⚙️ Optimization Parameters")
    pop_size = st.sidebar.number_input(
        "Population Size", 
        min_value=5, 
        value=100, 
        step=1, 
        help="Number of individuals in each generation (arbitrary positive integer)"
    )
    
    max_generations = st.sidebar.number_input(
        "Max Generations", 
        min_value=1, 
        value=100, 
        step=1, 
        help="Maximum number of generations to run (arbitrary positive integer)"
    )
    
    tolerance = st.sidebar.number_input("Tolerance", 1e-6, 1e-2, 1e-3, 1e-4, format="%.2e")
    
    # Algorithm-specific parameters
    st.sidebar.markdown("#### 🧬 Algorithm Parameters")
    st.sidebar.markdown("*These control crossover and mutation behavior*")
    
    crossover_eta = st.sidebar.number_input(
        "Crossover η (Distribution Index)", 
        min_value=0.1, 
        max_value=50.0, 
        value=preset_crossover_eta if preset_crossover_eta is not None else 1.0, 
        step=0.1, 
        help="Lower = more aggressive crossover, Higher = more conservative",
        disabled=(parameter_preset != "Custom")
    )
    
    crossover_prob = st.sidebar.number_input(
        "Crossover Probability", 
        min_value=0.1, 
        max_value=1.0, 
        value=preset_crossover_prob if preset_crossover_prob is not None else 0.8, 
        step=0.05, 
        help="Probability of crossover operation",
        disabled=(parameter_preset != "Custom")
    )
    
    mutation_eta = st.sidebar.number_input(
        "Mutation η (Distribution Index)", 
        min_value=0.1, 
        max_value=50.0, 
        value=preset_mutation_eta if preset_mutation_eta is not None else 3.0, 
        step=0.1, 
        help="Lower = more aggressive mutation, Higher = more conservative",
        disabled=(parameter_preset != "Custom")
    )
    
    mutation_prob = st.sidebar.number_input(
        "Mutation Probability", 
        min_value=0.1, 
        max_value=1.0, 
        value=preset_mutation_prob if preset_mutation_prob is not None else 0.95, 
        step=0.05, 
        help="Probability of mutation operation",
        disabled=(parameter_preset != "Custom")
    )
    
    # Initial state
    initial_state = st.sidebar.selectbox(
        "Initial State",
        ["vacuum", "fock", "coherent", "squeezed", "displaced", "cat", "squeezed_cat"],
        index=0
    )
    
    # Show photon number input for Fock state
    initial_state_param = 0.0
    if initial_state == "fock":
        initial_state_param = st.sidebar.number_input(
            "Number of Photons",
            min_value=0,
            max_value=N-1,
            value=1,
            step=1,
            help=f"Number of photons in the Fock state (must be < N={N})"
        )
    elif initial_state in ["coherent", "squeezed", "displaced", "cat", "squeezed_cat"]:
        initial_state_param = st.sidebar.number_input(
            f"Parameter for {initial_state} state",
            min_value=-10.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help=f"Parameter value for {initial_state} state"
        )
    
    # Target superposition input
    st.sidebar.header("Target Superposition")
    st.sidebar.markdown("Enter complex coefficients for the target superposition state.")
    
    # Add format information
    with st.sidebar.expander("ℹ️ Target Superposition Format", expanded=False):
        st.markdown("""
        **Supported formats:**
        - **Real numbers**: `1.0`, `0.5`
        - **Imaginary numbers**: `1.0j`, `0.5j`
        - **Complex numbers**: `1.0+1.0j`, `0.5-0.866j`
        - **Polar form**: `1.0*e^(i*pi/4)`
        
        **Examples:**
        - Equal superposition: a=`1.0`, b=`1.0`
        - Phase difference: a=`1.0`, b=`1.0j`
        - Complex superposition: a=`1.0`, b=`1.0+0.5j`
        - Polar form: a=`1.0`, b=`e^(i*pi/4)`
        - Square root: a=`sqrt(2)/2`, b=`sqrt(2)/2*j`
        - Trigonometric: a=`cos(pi/4)`, b=`sin(pi/4)*j`
        - Advanced polar: a=`1.0`, b=`2*e^(i*pi/3)`
        """)

    # Text input fields for complex expressions
    st.sidebar.markdown("**Enter coefficients as text expressions:**")

    # Use session state to remember values
    if 'coeff_a_text' not in st.session_state:
        st.session_state.coeff_a_text = "1.0"
    if 'coeff_b_text' not in st.session_state:
        st.session_state.coeff_b_text = "1.0"

    col1, col2 = st.sidebar.columns(2)
    with col1:
        coeff_a_text = st.text_input(
            "a (real only)", 
            value=st.session_state.coeff_a_text,
            help="Real coefficient only. Examples: 1.0, 0.5, sqrt(2)/2"
        )
    with col2:
        coeff_b_text = st.text_input(
            "b (complex)", 
            value=st.session_state.coeff_b_text,
            help="Complex coefficient. Examples: 1.0, 1.0j, 1.0+1.0j, e^(i*pi/4)"
        )

    # Update session state
    st.session_state.coeff_a_text = coeff_a_text
    st.session_state.coeff_b_text = coeff_b_text

    # Parse the coefficients using the existing parse function
    try:
        from src.utils import parse_target_superposition
        target_superposition = parse_target_superposition([coeff_a_text, coeff_b_text])
    except Exception as e:
        st.sidebar.error(f"❌ Error parsing coefficients: {str(e)}")
        # Fallback to default values
        target_superposition = (1.0, 1.0)
    
    # GPU settings
    gpu_id = st.sidebar.selectbox("GPU Device", [0, 1, 2, 3], index=0)
    use_ket_optimization = st.sidebar.checkbox("Use Ket Optimization", value=True, help="Use ket-based optimization (faster) instead of density matrix optimization")
    verbose = st.sidebar.checkbox("Verbose Output", value=True, help="Enable verbose optimization output in the console")
    enable_signal_handler = st.sidebar.checkbox("Enable Signal Handler", value=False, help="Enable graceful interrupt handling with Ctrl+C (disable for Streamlit)")
    
    # Normalize coefficients if needed
    normalization = np.sqrt(target_superposition[0]**2 + abs(target_superposition[1])**2)
    if normalization > 1e-10:  # Avoid division by zero
        normalized_a = target_superposition[0] / normalization
        normalized_b = target_superposition[1] / normalization
        target_superposition = (normalized_a, normalized_b)
        
        # Show warning if coefficients were not normalized
        if abs(normalization - 1.0) > 1e-6:
            st.sidebar.warning(f"⚠️ Coefficients were automatically normalized from {normalization:.3f} to 1.0")

    # Display current coefficients for debugging
    st.sidebar.markdown("### Current Coefficients")
    st.sidebar.info(f"**a (real):** {target_superposition[0]:.3f}  \n**b:** {target_superposition[1]:.3f}  \n**Normalization:** {np.sqrt(target_superposition[0]**2 + abs(target_superposition[1])**2):.3f}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">🎯 Target Operator Ground State</div>', unsafe_allow_html=True)
        
        # Auto-update operator when coefficients change
        target_superposition_key = f"{target_superposition}_{N}"
        
        # Check if we need to update the operator
        if (st.session_state.operator is None or 
            st.session_state.get('last_target_superposition_key') != target_superposition_key):
            try:
                with st.spinner("Constructing operator and computing ground state..."):
                    operator = construct_operator(N, target_superposition)
                    ground_state = operator.groundstate()[1]  # Get the ground state vector
                    
                    st.session_state.operator = operator
                    st.session_state.ground_state = ground_state
                    st.session_state.last_target_superposition_key = target_superposition_key
                    
                    st.success(f"✅ Operator constructed successfully!")
                    st.info(f"Ground state eigenvalue: {operator.groundstate()[0]:.6f}")
                    
                    # Debug: Show what coefficients were used
                    st.info(f"**Coefficients used:** a={target_superposition[0]:.3f}, b={target_superposition[1]:.3f}")
                    
            except Exception as e:
                st.error(f"❌ Error constructing operator: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Manual update button (optional)
        if st.button("🔄 Force Update Operator", key="force_update_op"):
            try:
                with st.spinner("Force updating operator and computing ground state..."):
                    operator = construct_operator(N, target_superposition)
                    ground_state = operator.groundstate()[1]  # Get the ground state vector
                    
                    st.session_state.operator = operator
                    st.session_state.ground_state = ground_state
                    st.session_state.last_target_superposition_key = target_superposition_key
                    
                    st.success(f"✅ Operator force updated successfully!")
                    st.info(f"Ground state eigenvalue: {operator.groundstate()[0]:.6f}")
                    
                    # Debug: Show what coefficients were used
                    st.info(f"**Coefficients used:** a={target_superposition[0]:.3f}, b={target_superposition[1]:.3f}")
                    
            except Exception as e:
                st.error(f"❌ Error force updating operator: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Display ground state if available
        if st.session_state.ground_state is not None:
            try:
                # Create Wigner function plot
                from src.plotter import plot_single_state
                fig, ax, cax = plot_single_state(
                    st.session_state.ground_state, 
                    f"Ground State (N={N})",
                    figsize=(6, 5),
                    xvec=np.linspace(-6, 6, 100),
                    yvec=np.linspace(-6, 6, 100)
                )
                
                # Convert matplotlib figure to image
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                img = Image.open(buf)
                
                st.image(img, caption="Ground State Wigner Function", width="stretch")
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Error plotting ground state: {str(e)}")
        
        st.markdown('<div class="section-header">🚀 Optimization Control</div>', unsafe_allow_html=True)
        
        # Start optimization button
        if st.button("🚀 Start Optimization", type="primary"):
            if st.session_state.operator is None:
                st.error("❌ Please construct the operator first!")
            else:
                st.session_state.optimization_running = True
                st.session_state.progress_data = []
                st.session_state.optimization_results = None
                st.session_state.current_generation = 0
                st.session_state.generation_data = []

        # Add stop button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("⏹️ Stop Optimization", disabled=not st.session_state.optimization_running):
                st.session_state.optimization_running = False
                st.session_state.graceful_stop_requested = True
                st.warning("🛑 Optimization stop requested - finishing current generation...")

        # Progress display
        if st.session_state.optimization_running:
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.markdown("### 🔄 Optimization Progress")
            
            # Create placeholders for progress updates
            progress_bar = st.progress(0)
            status_text = st.empty()
            progress_table = st.empty()
            
            try:
                # Set GPU device
                from src.cuda_helpers import set_gpu_device
                set_gpu_device(gpu_id)
                
                # Construct initial state
                initial_state_qobj, initial_probability = construct_initial_state(N, initial_state, initial_state_param)
                
                # Use operator from session state
                operator = st.session_state.operator
                
                st.success(f"✅ Using operator from target superposition: {target_superposition}")
                
                # Create a custom callback for real-time updates
                class StreamlitOptimizationCallback(Callback):
                    def __init__(self, progress_bar, status_text, progress_table, session_state, max_generations):
                        super().__init__()
                        self.progress_bar = progress_bar
                        self.status_text = status_text
                        self.progress_table = progress_table
                        self.session_state = session_state
                        self.max_generations = max_generations
                        self.generation_data = []
                        self.data = {"F": []}
                        self.interrupted = False
                    
                    def notify(self, algorithm):
                        """Called after each generation"""
                        gen = algorithm.n_gen
                        pop = algorithm.pop
                        
                        # Store F data
                        F = pop.get("F")
                        if F is not None:
                            self.data["F"].append(F.copy())
                        
                        # Update progress
                        progress = min(gen / self.max_generations, 1.0)
                        self.progress_bar.progress(progress)
                        
                        # Update status
                        self.status_text.text(f"🔄 Generation {gen}/{self.max_generations}")
                        
                        # Check for interrupt request (graceful stop)
                        if not self.session_state.optimization_running or getattr(self.session_state, 'graceful_stop_requested', False):
                            print(f"\n✅ Gracefully stopping optimization at generation {gen}")
                            if F is not None and len(F) > 0:
                                print(f"   Final results: {len(F)} solutions, best expectation: {np.min(F[:, 0]):.6f}, best probability: {np.max(-F[:, 1]):.6f}")
                            # Force termination immediately
                            algorithm.termination.force_termination = True
                            # Store that we requested an interrupt
                            self.interrupted = True
                        
                        # Collect generation data for table
                        if F is not None and len(F) > 0:
                            min_expectation = np.min(F[:, 0])
                            max_probability = -np.min(F[:, 1])  # Convert back from negative
                            n_solutions = len(F)
                            
                            self.generation_data.append({
                                'Generation': gen,
                                'Min Expectation': f"{min_expectation:.6f}",
                                'Max Probability': f"{max_probability:.6f}",
                                'Pareto Solutions': n_solutions
                            })
                            
                            # Limit generation data size to prevent memory leaks
                            if len(self.generation_data) > 100:  # Keep only last 100 generations
                                self.generation_data = self.generation_data[-100:]
                            
                            # Display progress table (last 10 generations)
                            if self.generation_data:
                                df = pd.DataFrame(self.generation_data)
                                df_display = df.tail(10)
                                self.progress_table.dataframe(df_display, width="stretch", hide_index=True)
                
                # Create callback
                callback = StreamlitOptimizationCallback(progress_bar, status_text, progress_table, st.session_state, max_generations)
                
                # Check if we should continue with optimization
                if not getattr(st.session_state, 'graceful_stop_requested', False):
                    result, F_history = run_quantum_sequence_optimization(
                        initial_state=initial_state_qobj,
                        initial_probability=initial_probability,
                        operator=operator,
                        N=N,
                        sequence_length=sequence_length,
                        pop_size=pop_size,
                        max_generations=max_generations,
                        device_id=gpu_id,
                        tolerance=tolerance,
                        algorithm=algorithm,
                        use_ket_optimization=use_ket_optimization,
                        verbose=verbose,
                        enable_signal_handler=enable_signal_handler,
                        callback=callback,
                        custom_crossover={"eta": crossover_eta, "prob": crossover_prob},
                        custom_mutation={"eta": mutation_eta, "prob": mutation_prob}
                    )
                else:
                    # If graceful stop was requested before optimization started, skip it
                    st.warning("🛑 Optimization was stopped before starting")
                    result = None
                    F_history = []
                
                # Get real F_history from callback data
                if result is not None:
                    F_history = callback.data["F"] if callback.data["F"] else []
                else:
                    F_history = []
                
                # Update session state
                st.session_state.optimization_results = (result, F_history)
                st.session_state.optimization_running = False
                
                # Check if optimization was stopped gracefully
                was_gracefully_stopped = getattr(st.session_state, 'graceful_stop_requested', False) or callback.interrupted
                
                # Show completion message
                progress_bar.progress(1.0)
                if was_gracefully_stopped:
                    if result is not None and len(F_history) > 0:
                        status_text.text(f"✅ Optimization stopped gracefully! ({len(F_history)} generations)")
                        st.success(f"🛑 Optimization stopped by user after {len(F_history)} generations")
                        should_save_results = True
                    else:
                        status_text.text("🛑 Optimization stopped before starting")
                        st.warning("🛑 Optimization was stopped before it could begin")
                        should_save_results = False
                else:
                    status_text.text(f"✅ Optimization completed! ({len(F_history)} generations)")
                    st.success("🎉 Optimization completed successfully!")
                    should_save_results = True
                
                # Reset graceful stop flag
                if hasattr(st.session_state, 'graceful_stop_requested'):
                    st.session_state.graceful_stop_requested = False
                
                # Save results with versioning (only if we have results to save)
                if should_save_results:
                    with st.spinner("Saving results with versioning..."):
                        # Save results to versioned folder
                        folder = save_streamlit_optimization_results(
                            result=result,
                            F_history=F_history,
                            algorithm=algorithm,
                            N=N,
                            sequence_length=sequence_length,
                            pop_size=pop_size,
                            max_generations=max_generations,
                            tolerance=tolerance,
                            target_superposition=target_superposition,
                            initial_state=initial_state,
                            initial_state_param=initial_state_param
                        )
                        
                        # Store the results folder path
                        st.session_state.results_folder = folder
                        st.session_state.animation_file = os.path.join(folder, "boundary_state_animation.mp4")
                
                
            except Exception as e:
                st.error(f"❌ Error running optimization: {str(e)}")
                st.session_state.optimization_running = False
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Results section
        if st.session_state.optimization_results is not None:
            result, F_history = st.session_state.optimization_results
            
            # Only show results if we have valid results
            if result is not None and len(F_history) > 0:
                st.markdown('<div class="section-header">📈 Optimization Results</div>', unsafe_allow_html=True)
                
                # Analyze results
                metrics = analyze_sequence_result(result)
                
                # Convert F and X back to numpy arrays for plotting and download
                metrics['F'] = np.array(metrics['F'])
                metrics['X'] = np.array(metrics['X'])
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Pareto Solutions", metrics['n_solutions'])
                with col2:
                    st.metric("Best Expectation", f"{metrics['min_expectation']:.4f}")
                with col3:
                    st.metric("Best Probability", f"{metrics['max_probability']:.4f}")
                with col4:
                    st.metric("Final Generation", len(F_history))
                
                # Plot Pareto front evolution
                st.markdown("### 📊 Pareto Front Evolution")
            
                # Create evolution plot with smart color scheme (matching StatePrep)
                fig = go.Figure()
                
                # Use a color palette that works well for many generations
                import plotly.colors as pc
                colors = pc.qualitative.Set3  # Good for many categories
                
                # Add traces for each generation (show ALL generations)
                for i, F in enumerate(F_history):
                    # Use different colors for different generations
                    color = colors[i % len(colors)]
                    
                    # Make later generations more prominent
                    opacity = 0.4 + 0.6 * (i / max(1, len(F_history) - 1))
                    size = 6 + 2 * (i / max(1, len(F_history) - 1))
                    
                    fig.add_trace(go.Scatter(
                        x=F[:, 0],
                        y=-F[:, 1],  # Convert back from negative probability
                        mode='markers',
                        name=f'Generation {i + 1}',
                        marker=dict(
                            size=size,
                            color=color,
                            opacity=opacity
                        )
                    ))
                
                fig.update_layout(
                    title="Pareto Front Evolution",
                    xaxis_title="Operator Expectation (minimize)",
                    yaxis_title="Total Probability (maximize)",
                    width=800,
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed report
                st.markdown("### 📋 Detailed Report")
                report = format_sequence_report(metrics, solution_index=0)
                st.text(report)
                
                # Download results
                st.markdown("### 💾 Download Results")
                
                # Create downloadable data
                results_data = {
                    'X': metrics['X'].tolist(),
                    'F': metrics['F'].tolist(),
                    'sequence_params': metrics.get('sequence_params', []),
                    'parameters': {
                        'algorithm': algorithm,
                        'N': N,
                        'sequence_length': sequence_length,
                        'pop_size': pop_size,
                        'max_generations': max_generations,
                        'tolerance': tolerance,
                        'target_superposition': str(target_superposition),
                        'initial_state': initial_state
                    }
                }
                
                # Convert to JSON for download
                import json
                json_str = json.dumps(results_data, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json_str,
                    file_name=f"magic_optimization_results_N{N}_seq{sequence_length}.json",
                    mime="application/json"
                )
                
                # Display animation
                st.markdown("### 🎬 Animation")
                if hasattr(st.session_state, 'animation_file'):
                    animation_file = st.session_state.animation_file
                    if os.path.exists(animation_file):
                        st.video(animation_file)
                    else:
                        st.info("Animation file not found. Check the output folder for generated animations.")
                else:
                    st.info("Animation file not available.")

    with col2:
        st.header("System Status")
        
        # Display GPU memory status
        try:
            from src.cuda_helpers import print_memory_status
            memory_info = print_memory_status()
            st.text(memory_info)
        except:
            st.warning("Could not retrieve GPU memory status")
        
        # Display current parameters
        st.subheader("Current Parameters")
        preset_info = f"\n**📊 Parameter Preset:** {parameter_preset}" if parameter_preset != "Custom" else ""
        st.write(f"**Algorithm:** {algorithm}")
        st.write(f"**N:** {N}")
        st.write(f"**Sequence Length:** {sequence_length}")
        st.write(f"**Population Size:** {pop_size}")
        st.write(f"**Max Generations:** {max_generations}")
        initial_state_display = f"{initial_state}"
        if initial_state == "fock":
            initial_state_display += f" (n={int(initial_state_param)} photons)"
        elif initial_state in ["coherent", "squeezed", "displaced", "cat", "squeezed_cat"]:
            initial_state_display += f" (param={initial_state_param})"
        st.write(f"**Initial State:** {initial_state_display}{preset_info}")
        st.write(f"**Target Superposition:** {target_superposition}")
        
        # Algorithm parameters
        st.write("**Algorithm Parameters:**")
        st.write(f"- Crossover: η={crossover_eta}, prob={crossover_prob}")
        st.write(f"- Mutation: η={mutation_eta}, prob={mutation_prob}")
        
        # Show benchmark info if using a preset
        if parameter_preset != "Custom":
            if parameter_preset == "🎯 Best Quality (Baseline)":
                st.success("🎯 **Best Quality Preset** - Optimized for solution quality")
            elif parameter_preset == "⚡ Fastest (Fast Convergence)":
                st.success("⚡ **Fastest Preset** - Optimized for speed")
            elif parameter_preset == "🚀 Aggressive (High Exploration)":
                st.success("🚀 **Aggressive Preset** - Maximum exploration")
        
        # Display help
        st.subheader("Help")
        st.markdown("""
        **Target Superposition Format:**
        - Simple: `1.0, 1.0`
        - Complex: `1.0, 1.0j`
        - Polar: `1.0*e^(i*pi/4)`
        - Mixed: `0.5+0.866j, 1.0`
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    🔮 Magic Quantum Sequence Optimization | Built with Streamlit
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
