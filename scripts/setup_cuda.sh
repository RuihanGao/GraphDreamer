#!/bin/bash
source /data/ruihan/anaconda3/etc/profile.d/conda.sh
conda activate GraphDreamer

# Set up CUDA environment variables
export CUDA_HOME=/usr/local/cuda-11.7 # default on capybara machine is cuda12
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH