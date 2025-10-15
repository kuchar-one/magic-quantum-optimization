#!/usr/bin/env python3
"""
Simple launcher script for the Magic Quantum Sequence Optimization Streamlit App
Similar to StatePrep's demo_app.py
"""

import subprocess
import sys
import os

def main():
    print("🔮 Magic Quantum Sequence Optimization - Streamlit App")
    print("=" * 60)
    print()
    
    print("📱 Starting the web interface...")
    print("🌐 The app will be available at: http://localhost:8501")
    print()
    print("🎯 Features:")
    print("  • Interactive parameter input")
    print("  • Real-time optimization progress")
    print("  • 11 PyMOO algorithms")
    print("  • Ket-based quantum optimization")
    print("  • Target superposition input")
    print("  • Results visualization and animation")
    print()
    print("🚀 Quick Start:")
    print("  1. Set target superposition in sidebar")
    print("  2. Choose optimization algorithm")
    print("  3. Configure parameters")
    print("  4. Click 'Start Optimization'")
    print("  5. Watch progress and analyze results")
    print()
    print("💡 Recommended algorithms:")
    print("  • SMSEMOA - Best overall performance")
    print("  • NSGA2 - Best solution quality")
    print("  • UNSGA3 - Balanced performance")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_app.py"):
        print("❌ Error: streamlit_app.py not found!")
        print("   Please run this script from the magic project directory.")
        return 1
    
    # Check if virtual environment exists
    if os.path.exists(".venv"):
        print("🐍 Using virtual environment...")
        cmd = ["bash", "-c", "source .venv/bin/activate && streamlit run streamlit_app.py"]
    else:
        print("🐍 Using system Python...")
        cmd = ["streamlit", "run", "streamlit_app.py"]
    
    try:
        print("🚀 Launching Streamlit app...")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
