#!/bin/bash
# applicable to GPU

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_gpu.sh DATA_PATH DEVICE_TARGET"
echo "DEVICE_TARGET could be GPU or Ascend"
echo "For example: bash run_gpu.sh /path/dataset GPU"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
DEVICE_TARGET=$2
export DATA_PATH=${DATA_PATH}
export DEVICE_TARGET=${DEVICE_TARGET}

echo "start training"
mpirun -n 2 pytest -s -v ./resnet50_distributed_training_pipeline.py #>train.log 2>&1 &
