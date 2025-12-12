#!/bin/bash
set -e
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh  RANK_SIZE"
echo "For example: bash run.sh 8"
echo "This example is expected to run on the Ascend environment."
echo "=============================================================================================================="
RANK_SIZE=$1

mpirun -n $RANK_SIZE python fusion_example.py