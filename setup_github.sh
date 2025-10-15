#!/bin/bash

# Magic Quantum Sequence Optimization - GitHub Repository Setup Script
# This script helps set up a new GitHub repository for the project

set -e  # Exit on any error

echo "🔮 Magic Quantum Sequence Optimization - GitHub Setup"
echo "=================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files
echo "Adding files to git..."
git add .
echo "✅ Files added to git"

# Make initial commit
echo "Making initial commit..."
git commit -m "Initial commit: Magic Quantum Sequence Optimization v2.0.0

- Comprehensive quantum optimization framework
- 11 PyMOO algorithms integration
- Streamlit web interface
- GPU acceleration with CUDA
- Advanced caching and memory management
- Real-time optimization monitoring
- Animation generation
- Comprehensive documentation"
echo "✅ Initial commit made"

# Create main branch
echo "Creating main branch..."
git branch -M main
echo "✅ Main branch created"

echo ""
echo "🎉 Local repository setup complete!"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub (e.g., magic-quantum-optimization)"
echo "2. Add the remote origin:"
echo "   git remote add origin https://github.com/yourusername/magic-quantum-optimization.git"
echo "3. Push to GitHub:"
echo "   git push -u origin main"
echo ""
echo "Or run:"
echo "   gh repo create magic-quantum-optimization --public --source=. --remote=origin --push"
echo ""
echo "For more information, see README.md"
