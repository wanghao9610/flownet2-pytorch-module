#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
TORCH=$($PYTHON -c "import os; import torch; print(os.path.dirname(torch.__file__))")

echo "Compiling resample2d kernels by nvcc..."
rm -rf *_cuda.egg-info build dist __pycache__
$PYTHON setup.py install --user
