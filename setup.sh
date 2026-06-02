#!/bin/bash

set -e

PYTHON=/opt/homebrew/bin/python3.12

echo "Using Python: $PYTHON"

echo "Removing old venv if present..."
rm -rf venv

echo "Creating virtual environment..."
$PYTHON -m venv venv --without-pip

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing pip manually..."
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
rm get-pip.py

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install torch gymnasium gym pygame

echo ""
echo "✅ Done! Activate with: source venv/bin/activate"