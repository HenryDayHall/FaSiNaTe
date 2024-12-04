#!/usr/bin/env bash

# Script to setup key4hep stable envrionment that allows to use (py)torch, onnx and Marlin
# together with DD4hep.
#
#

source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-11-28

echo "Finding torch"
TORCH_PATH=$(dirname $(python -c 'import torch; print(f"{torch.__file__}")'))
echo "Found torch at ${TORCH_PATH}"
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:${TORCH_PATH}/share/cmake
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${TORCH_PATH}/lib

echo "Finding onnx"
ONNXRUNTIME_PATH=$(dirname $(python -c 'import onnxruntime; print(f"{onnxruntime.__file__}")'))
echo "Found onnx at ${ONNXRUNTIME_PATH}"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(realpath ${ONNXRUNTIME_PATH}/../../../../)/lib64  # CentOS7
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(realpath ${ONNXRUNTIME_PATH}/../../../../)/lib    # Ubuntu
