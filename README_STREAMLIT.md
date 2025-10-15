# Magic Quantum Sequence Optimization - Streamlit App

A web-based interface for optimizing quantum operation sequences to prepare target superposition states using breeding operations.

## Features

- **Interactive Parameter Control**: Adjust optimization parameters through a web interface
- **Real-time Monitoring**: Watch optimization progress with live plots and metrics
- **Multiple Algorithms**: Choose from 11 different optimization algorithms (NSGA2, NSGA3, MOEA/D, etc.)
- **Target Superposition Input**: Specify complex target states using various formats
- **GPU Acceleration**: Utilize CUDA for fast quantum operations
- **Visualization**: Real-time Pareto front evolution and final results
- **Animation Generation**: Automatic creation of optimization sequence animations

## Quick Start

1. **Activate the virtual environment**:
   ```bash
   cd /home/vojtech/cloud/python/code/magic/stateprep
   source .venv/bin/activate
   ```

2. **Launch the Streamlit app**:
   ```bash
   cd ../magic
   ./launch_app.sh
   ```

3. **Open your browser** and go to `http://localhost:8501`

## Usage

### Setting Parameters

- **Algorithm**: Choose from 11 optimization algorithms
- **Hilbert Space Dimension (N)**: Size of the quantum state space (5-50)
- **Sequence Length**: Number of operations in the sequence (3-20)
- **Population Size**: Number of solutions per generation (20-200)
- **Max Generations**: Maximum optimization iterations (10-500)
- **Initial State**: Starting quantum state (vacuum, coherent, squeezed, etc.)

### Target Superposition

Specify your target superposition state using complex number notation:

- **Simple**: `1.0, 1.0` (equal superposition)
- **Complex**: `1.0, 1.0j` (with phase)
- **Polar**: `1.0*e^(i*pi/4)` (polar form)
- **Mixed**: `0.5+0.866j, 1.0` (mixed real/complex)

### Running Optimization

1. Set your desired parameters in the sidebar
2. Click "🚀 Start Optimization"
3. Monitor progress in real-time
4. View final results and download animations

## Output

The app generates:

- **Metrics**: Pareto front statistics and best solutions
- **Plots**: Real-time evolution and final Pareto front
- **Animation**: MP4 video showing the optimization sequence
- **Data**: JSON files with detailed results

## File Structure

```
magic/
├── streamlit_app.py          # Main Streamlit application
├── launch_app.sh            # Launch script
├── requirements_streamlit.txt # Python dependencies
├── README_STREAMLIT.md      # This file
└── src/                     # Source code modules
    ├── nsga_ii.py          # Optimization algorithms
    ├── qutip_quantum_ops.py # Quantum operations
    ├── cuda_helpers.py     # GPU utilities
    └── data_helpers.py     # Data processing
```

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**: Reduce population size or sequence length
2. **Import Errors**: Ensure virtual environment is activated
3. **Animation Not Generated**: Check file permissions in output directory

### Performance Tips

- Use smaller N values for faster optimization
- Reduce population size for quicker results
- Enable GPU acceleration for better performance

## Advanced Usage

### Custom Target States

For complex target states, use the custom input option:

```
# Bell state
1/sqrt(2), 1/sqrt(2)

# GHZ state  
1/sqrt(3), 1/sqrt(3), 1/sqrt(3)

# Custom phase
1.0, 0.5+0.866j
```

### Algorithm Selection

- **NSGA2**: Good general-purpose algorithm
- **NSGA3**: Better for many objectives
- **MOEA/D**: Efficient for specific problems
- **AGE**: Good convergence properties

## Dependencies

- Python 3.8+
- PyTorch with CUDA support
- QuTiP for quantum operations
- PyMOO for optimization
- Streamlit for web interface
- Plotly for interactive plots

## License

This project is part of the Magic quantum optimization suite.
