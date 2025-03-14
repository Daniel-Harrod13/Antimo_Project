#!/bin/bash

# Commodity Market Price Prediction & Analysis Platform
# Conda Environment Setup Script

# Set environment name
ENV_NAME="commodity"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -q $ENV_NAME; then
    echo "Creating new conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=3.9
else
    echo "Conda environment $ENV_NAME already exists"
fi

# Activate environment
echo "Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Setup project
echo "Setting up project directories..."
python main.py setup

echo ""
echo "Setup complete! You can now run the project with the following commands:"
echo ""
echo "1. Activate the environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "2. Generate sample data:"
echo "   python main.py collect"
echo ""
echo "3. Train models:"
echo "   python main.py train"
echo ""
echo "4. Run the dashboard:"
echo "   python main.py dashboard"
echo ""
echo "The dashboard will be available at http://localhost:8501" 