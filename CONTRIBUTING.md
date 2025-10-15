# Contributing to Magic Quantum Sequence Optimization

Thank you for your interest in contributing to Magic! This document provides guidelines and information for contributors.

## Getting Started

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/magic-quantum-optimization.git
   cd magic-quantum-optimization
   ```

3. **Set up development environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install in development mode
   pip install -e .
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test file
python -m pytest tests/test_optimization.py
```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use type hints for all function parameters and return values
- Add comprehensive docstrings using Google style
- Keep line length under 88 characters (Black formatter default)

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
# Run linter
flake8 src/ tests/
```

### Type Checking

We use [mypy](https://mypy.readthedocs.io/) for type checking:

```bash
# Run type checker
mypy src/
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation if needed
   - Ensure all tests pass

3. **Run quality checks**:
   ```bash
   # Format code
   black src/ tests/
   
   # Run linter
   flake8 src/ tests/
   
   # Run type checker
   mypy src/
   
   # Run tests
   python -m pytest tests/
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

### Submitting a Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub with:
   - Clear title describing the changes
   - Detailed description of what was changed and why
   - Reference any related issues
   - Screenshots or examples if applicable

### Pull Request Guidelines

- **One feature per PR**: Keep pull requests focused on a single feature or bug fix
- **Write clear commit messages**: Use imperative mood ("Add feature" not "Added feature")
- **Update documentation**: Include updates to README, docstrings, or other docs
- **Add tests**: Include tests for new functionality
- **Keep PRs small**: Break large changes into smaller, manageable PRs

## Development Guidelines

### Adding New Features

1. **Design first**: Plan the feature and its API before coding
2. **Write tests**: Add tests for new functionality
3. **Update documentation**: Update README, docstrings, and other docs
4. **Consider performance**: Ensure new features don't significantly impact performance
5. **Handle errors gracefully**: Add proper error handling and logging

### Bug Fixes

1. **Reproduce the bug**: Create a minimal test case that reproduces the issue
2. **Write a failing test**: Add a test that demonstrates the bug
3. **Fix the bug**: Implement the fix
4. **Verify the fix**: Ensure the test now passes
5. **Update documentation**: Update any relevant documentation

### Code Review

All code changes require review before merging. Reviewers will check for:

- **Correctness**: Does the code work as intended?
- **Style**: Does it follow the project's style guidelines?
- **Performance**: Are there any performance concerns?
- **Security**: Are there any security issues?
- **Documentation**: Is the code well-documented?

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected behavior** vs actual behavior
4. **Environment details** (OS, Python version, dependencies)
5. **Error messages** or logs
6. **Minimal code example** if possible

### Feature Requests

When requesting features, please include:

1. **Clear description** of the desired feature
2. **Use case** explaining why this feature would be useful
3. **Proposed implementation** if you have ideas
4. **Alternatives considered** and why they weren't suitable

## Documentation

### Code Documentation

- **Docstrings**: Use Google style docstrings for all public functions and classes
- **Comments**: Add inline comments for complex logic
- **Type hints**: Include type hints for all function parameters and return values

### User Documentation

- **README**: Keep the main README up to date
- **Examples**: Provide clear examples of how to use new features
- **API docs**: Document any new public APIs

## Testing

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test how components work together
- **Performance tests**: Ensure performance requirements are met
- **End-to-end tests**: Test complete workflows

### Writing Tests

```python
def test_function_name():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

## Performance Considerations

### GPU Memory

- **Clean up GPU memory** after operations
- **Use memory-efficient algorithms** when possible
- **Monitor memory usage** during development

### CPU Performance

- **Profile code** to identify bottlenecks
- **Use vectorized operations** when possible
- **Consider parallel processing** for CPU-bound tasks

## Security

### Code Security

- **Validate inputs** to prevent injection attacks
- **Use secure random number generation** for cryptographic operations
- **Handle sensitive data** appropriately

### Dependencies

- **Keep dependencies updated** to avoid security vulnerabilities
- **Audit dependencies** regularly
- **Use trusted sources** for package installation

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backwards-compatible manner
- **PATCH**: Backwards-compatible bug fixes

### Release Checklist

1. **Update version** in setup.py and __init__.py
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** to ensure everything works
4. **Update documentation** if needed
5. **Create release tag** on GitHub
6. **Build and upload** to PyPI (if applicable)

## Community

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Pull Requests**: For code contributions

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- **Be respectful** and considerate of others
- **Be patient** with newcomers and questions
- **Focus on constructive feedback**
- **Help others learn** and grow

## Getting Help

If you need help:

1. **Check the documentation** first
2. **Search existing issues** for similar problems
3. **Ask questions** in GitHub Discussions
4. **Create an issue** if you find a bug

Thank you for contributing to Magic Quantum Sequence Optimization! 🚀
