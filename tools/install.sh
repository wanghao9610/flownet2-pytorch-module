#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
TORCH=$($PYTHON -c "import os; import torch; print(os.path.dirname(torch.__file__))")

echo "Compiling correlation kernels by nvcc..."
cd ../ops/correlatiols
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user

echo "Compiling resample2d kernels by nvcc..."
cd ../resample2d
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user

echo "Compiling channel_norm kernels by nvcc..."
cd ../channelnorm
rm -rf *_cuda.egg-info build dist __pycache__
python3 setup.py install --user
