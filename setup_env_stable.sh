#!/usr/bin/env bash

# Script to setup key4hep stable envrionment that allows to use (py)torch, onnx and Marlin
# together with DD4hep.
#
#

source /cvmfs/ilc.desy.de/key4hep/releases/ca872f9c34/x86_64-ubuntu22.04-gcc11.3.0-opt/key4hep-stack/2023-09-28-hmnio4/setup.sh

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
