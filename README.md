# Magic Quantum Sequence Optimization

A comprehensive quantum optimization framework for preparing target superposition states using breeding operations, annihilation, creation, displacement, and squeezing operations. This project implements advanced multi-objective optimization algorithms with GPU acceleration for quantum state preparation.

## 🚀 Features

- **Multi-Objective Optimization**: 11 different PyMOO algorithms (NSGA2, NSGA3, MOEA/D, AGE, AGE2, RVEA, SMSEMOA, CTAEA, UNSGA3, RNSGA2, RNSGA3)
- **Ket-Based Optimization**: Pure state quantum optimization using kets instead of density matrices
- **GPU Acceleration**: CUDA-accelerated quantum operations for high-performance computing
- **Target Superposition Input**: Flexible specification of target quantum states using complex coefficients
- **Breeding Operations**: Advanced quantum breeding protocols for state preparation
- **Interactive Web Interface**: Streamlit-based GUI for parameter control and real-time monitoring
- **Comprehensive Benchmarking**: Built-in algorithm comparison and performance analysis
- **Graceful Interruption**: Clean termination with result saving
- **Advanced Caching**: Memory-efficient CUDA matrix caching
- **Animation Generation**: Automatic creation of optimization sequence visualizations

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Algorithm Comparison](#algorithm-comparison)
- [Streamlit Interface](#streamlit-interface)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- Linux/macOS/Windows

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd magic
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: For GPU acceleration, PyTorch will automatically install with CUDA support if available.

4. **Verify installation**:
   ```bash
   python quo.py --help
   ```

### GPU Setup (Optional but Recommended)

For CUDA acceleration, install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Quick Start

### Basic Optimization

```bash
python quo.py --N 20 --sequence_length 10 --pop_size 100 --max_generations 50 --algorithm nsga2 --target_superposition "1.0" "1.0"
```

### Interactive Web Interface

```bash
# Simple launcher (recommended)
python run_app.py

# Advanced launcher with options
python launch_magic_app.py

# Shell script launcher
./launch_app.sh

# Direct Streamlit
streamlit run streamlit_app.py
```

### Algorithm Comparison

```bash
python benchmark_algorithms.py --N 15 --sequence_length 5 --pop_size 50 --max_generations 20 --num_runs 3
```

## 📖 Usage

### Command Line Interface

The main optimization script `quo.py` provides comprehensive control over optimization parameters:

```bash
python quo.py [OPTIONS]
```

#### Core Parameters

- `--N`: Hilbert space dimension (default: 20)
- `--sequence_length`: Number of operations in sequence (default: 10)
- `--pop_size`: Population size for optimization (default: 100)
- `--max_generations`: Maximum number of generations (default: 100)
- `--algorithm`: Optimization algorithm (default: nsga2)

#### Target State Parameters

- `--target_superposition`: Target superposition coefficients (e.g., "1.0" "1.0j")
- `--initial_state`: Initial state type (vacuum, coherent, squeezed, displaced, cat, squeezed_cat)

#### Advanced Parameters

- `--gpu`: GPU device ID (default: 0)
- `--tolerance`: Convergence tolerance (default: 1e-3)
- `--animate_only`: Only generate animation from existing results

#### Example Commands

```bash
# Basic optimization
python quo.py --N 20 --sequence_length 10 --pop_size 100 --max_generations 50

# Complex target state
python quo.py --target_superposition "1.0" "1.0j" "0.5+0.866j"

# Different algorithm
python quo.py --algorithm nsga3 --pop_size 200 --max_generations 100

# GPU optimization
python quo.py --gpu 1 --N 30 --sequence_length 15
```

### Target Superposition Format

The `--target_superposition` parameter accepts complex number notation:

- **Real numbers**: `"1.0"`, `"0.5"`
- **Imaginary numbers**: `"1.0j"`, `"0.5j"`
- **Complex numbers**: `"1.0+1.0j"`, `"0.5-0.866j"`
- **Polar form**: `"1.0*e^(i*pi/4)"`

### Available Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `nsga2` | Non-dominated Sorting Genetic Algorithm II | General purpose |
| `nsga3` | NSGA-III with reference directions | Many objectives |
| `moead` | Multi-objective Evolutionary Algorithm based on Decomposition | Decomposed problems |
| `age` | Adaptive Geometry Estimation | Adaptive optimization |
| `age2` | AGE-II variant | Improved convergence |
| `rvea` | Reference Vector Guided Evolutionary Algorithm | Reference-based |
| `smsemoa` | S-Metric Selection Evolutionary Multi-objective Algorithm | S-metric optimization |
| `ctaea` | Constraint Tournament Archive Evolutionary Algorithm | Constrained problems |
| `unsga3` | Unified NSGA-III | Unified approach |
| `rnsga2` | Robust NSGA-II | Robust optimization |
| `rnsga3` | Robust NSGA-III | Robust many-objective |

## 🔬 API Reference

### Core Classes

#### `QuantumOperationSequence`

Main optimization problem class for quantum sequence optimization.

```python
from src.nsga_ii import QuantumOperationSequence

problem = QuantumOperationSequence(
    N=20,
    sequence_length=10,
    operator=operator,
    initial_state=initial_state,
    use_ket_optimization=True
)
```

#### `run_quantum_sequence_optimization`

Main optimization function.

```python
from src.nsga_ii import run_quantum_sequence_optimization

result, F_history = run_quantum_sequence_optimization(
    initial_state=initial_state,
    operator=operator,
    N=20,
    sequence_length=10,
    pop_size=100,
    max_generations=50,
    algorithm="nsga2",
    use_ket_optimization=True
)
```

### Quantum Operations

#### `construct_operator`

Construct quantum operator from target superposition.

```python
from src.qutip_quantum_ops import construct_operator

operator = construct_operator(N=20, target_superposition=(1.0, 1.0j))
```

#### `construct_initial_state`

Create initial quantum state.

```python
from src.qutip_quantum_ops import construct_initial_state

state, probability = construct_initial_state(N=20, desc="vacuum", param=0.0)
```

### GPU Operations

#### `set_gpu_device`

Set CUDA device for optimization.

```python
from src.cuda_helpers import set_gpu_device

set_gpu_device(0)  # Use GPU 0
```

#### `aggressive_memory_cleanup`

Clean up GPU memory.

```python
from src.cuda_helpers import aggressive_memory_cleanup

aggressive_memory_cleanup()
```

## 📊 Algorithm Comparison

The `benchmark_algorithms.py` script provides comprehensive algorithm comparison:

### Running Benchmarks

```bash
# Quick comparison
python benchmark_algorithms.py --algorithms nsga2 nsga3 moead

# Comprehensive benchmark
python benchmark_algorithms.py --N 20 --sequence_length 10 --pop_size 100 --max_generations 50 --num_runs 5

# Custom parameters
python benchmark_algorithms.py --N 30 --sequence_length 15 --pop_size 200 --max_generations 100 --target_superposition "1.0" "1.0j"
```

### Benchmark Output

The benchmark generates:

- **JSON Results**: Detailed performance metrics
- **Text Report**: Human-readable comparison
- **Visualization Plots**: Performance comparison charts
- **Statistics**: Success rates, convergence times, solution quality

### Metrics Evaluated

- **Runtime**: Total optimization time
- **Convergence**: Generation when best solution found
- **Solution Quality**: Best expectation value and probability
- **Robustness**: Success rate across multiple runs
- **Hypervolume**: Pareto front quality metric

## 🖥️ Streamlit Interface

The interactive web interface provides real-time optimization monitoring:

### Launching the Interface

#### Simple Launcher (Recommended)
```bash
python run_app.py
```
- Easy to use, similar to StatePrep's demo_app.py
- Automatic virtual environment detection
- Clear instructions and feature overview

#### Advanced Launcher
```bash
python launch_magic_app.py [OPTIONS]
```
- Advanced options: `--port`, `--host`, `--theme`
- Dependency checking
- GPU detection
- Project structure validation

#### Shell Script Launcher
```bash
./launch_app.sh
```
- Traditional shell script approach
- Simple and reliable

#### Direct Streamlit
```bash
streamlit run streamlit_app.py
```
- Direct Streamlit command
- Manual virtual environment activation required

### Features

- **Parameter Control**: Adjust all optimization parameters via GUI
- **Real-time Monitoring**: Live plots of optimization progress
- **Algorithm Selection**: Choose from all available algorithms
- **Target State Input**: Flexible complex number input
- **Results Visualization**: Interactive plots and animations
- **System Status**: GPU memory and performance monitoring

### Interface Components

1. **Sidebar**: Parameter control and algorithm selection
2. **Main Area**: Optimization control and results display
3. **Real-time Plots**: Live optimization progress
4. **Final Results**: Comprehensive solution analysis
5. **Animation Player**: Generated optimization sequences

## ⚙️ Configuration

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0  # Specify GPU device
export OMP_NUM_THREADS=4       # CPU threads
export MKL_NUM_THREADS=4       # Math kernel threads
```

### Configuration Files

Create `config.json` for default parameters:

```json
{
    "default_N": 20,
    "default_sequence_length": 10,
    "default_pop_size": 100,
    "default_max_generations": 50,
    "default_algorithm": "nsga2",
    "default_gpu_id": 0,
    "default_tolerance": 1e-3
}
```

### GPU Memory Management

The system automatically manages GPU memory, but you can control it manually:

```python
from src.cuda_helpers import print_memory_status, aggressive_memory_cleanup

# Check memory usage
print_memory_status()

# Clean up memory
aggressive_memory_cleanup()
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce population size: `--pop_size 50`
- Reduce sequence length: `--sequence_length 5`
- Use smaller N: `--N 15`
- Clean GPU memory: `python reset_gpu.py`

#### 2. Import Errors

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solutions**:
- Activate virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version`

#### 3. Algorithm Convergence Issues

**Error**: Poor optimization results

**Solutions**:
- Increase generations: `--max_generations 200`
- Increase population size: `--pop_size 200`
- Try different algorithm: `--algorithm nsga3`
- Adjust tolerance: `--tolerance 1e-4`

#### 4. Animation Generation Fails

**Error**: Animation not generated

**Solutions**:
- Check file permissions in output directory
- Ensure sufficient disk space
- Verify matplotlib backend: `export MPLBACKEND=Agg`

### Performance Optimization

#### GPU Optimization

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Monitor GPU usage
nvidia-smi -l 1
```

#### CPU Optimization

```bash
# Set thread count
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

#### Memory Optimization

```python
# Clean up between runs
aggressive_memory_cleanup()

# Monitor memory usage
print_memory_status()
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📁 Project Structure

```
magic/
├── src/                          # Source code
│   ├── nsga_ii.py               # Main optimization algorithms
│   ├── qutip_quantum_ops.py     # Quantum operations
│   ├── cuda_quantum_ops.py      # GPU-accelerated operations
│   ├── cuda_helpers.py          # GPU utilities
│   ├── data_helpers.py          # Data processing and visualization
│   ├── plotter.py               # Visualization utilities
│   ├── sampling.py              # Custom sampling algorithms
│   ├── logging.py               # Logging configuration
│   └── utils.py                 # Shared utility functions
├── output/                       # Results and outputs
│   ├── benchmarks/              # Algorithm comparison results
│   └── target_*/                # Optimization results
├── cache/                        # Cached operators
│   └── operators/               # Pre-computed quantum operators
├── quo.py                       # Main optimization script
├── streamlit_app.py             # Web interface
├── benchmark_algorithms.py      # Algorithm comparison
├── run_app.py                   # Simple launcher script
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `python -m pytest tests/`
5. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Include unit tests for new features

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_optimization.py

# Run with coverage
python -m pytest --cov=src tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyMOO**: Multi-objective optimization framework
- **QuTiP**: Quantum optics toolbox
- **PyTorch**: Deep learning framework with CUDA support
- **Streamlit**: Web application framework
- **NumPy/SciPy**: Scientific computing libraries

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@example.com

## 🔄 Changelog

### Version 2.0.0
- ✅ Ket-based optimization implementation
- ✅ 11 PyMOO algorithms integration
- ✅ Streamlit web interface
- ✅ Comprehensive algorithm benchmarking
- ✅ Advanced CUDA caching
- ✅ Target superposition input system
- ✅ Graceful interruption handling

### Version 1.0.0
- ✅ Initial density matrix optimization
- ✅ Basic breeding operations
- ✅ GPU acceleration
- ✅ Animation generation

---

**Magic Quantum Sequence Optimization** - Preparing quantum states with precision and efficiency! 🔮✨
