#!/bin/bash
#
# This script finds the pip-installed ptxas binary and symlinks it
# to the virtual environment's bin/ directory for TensorFlow to find.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Check if VIRTUAL_ENV is set ---
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: No virtual environment detected."
    echo "Please activate your virtual environment first (e.g., 'source .venv/bin/activate')"
    exit 1
fi

echo "Virtual environment found at: $VIRTUAL_ENV"

# --- 2. Find the NVIDIA package directory ---
echo "Locating nvidia package directory..."
NVIDIA_PKG_DIR=$(python -c "import nvidia; print(nvidia.__path__[0])")

if [ -z "$NVIDIA_PKG_DIR" ] || [ "$NVIDIA_PKG_DIR" == "None" ]; then
    echo "ERROR: Could not find the 'nvidia' package directory."
    echo "Is tensorflow[and-cuda] installed?"
    exit 1
fi
echo "Found nvidia package directory: $NVIDIA_PKG_DIR"

# --- 3. Find the ptxas binary ---
echo "Searching for ptxas binary..."
PTXAS_SOURCE=$(find "$NVIDIA_PKG_DIR" -name ptxas -type f -print -quit)

if [ -z "$PTXAS_SOURCE" ]; then
    echo "ERROR: Could not find ptxas binary inside $NVIDIA_PKG_DIR."
    echo "       The 'tensorflow[and-cuda]' package may be incomplete."
    echo "       Try running: pip install --force-reinstall \"tensorflow[and-cuda]\""
    exit 1
fi
echo "SUCCESS: Found ptxas at: $PTXAS_SOURCE"

# --- 4. Define target and create link ---
PTXAS_TARGET="$VIRTUAL_ENV/bin/ptxas"
echo "Linking to: $PTXAS_TARGET"
ln -sf "$PTXAS_SOURCE" "$PTXAS_TARGET"

# --- 5. Verify the link ---
echo "--------------------------------------------------------------------"
echo "Verifying link (output should show an arrow ->):"
ls -l "$PTXAS_TARGET"
echo "Script finished successfully."