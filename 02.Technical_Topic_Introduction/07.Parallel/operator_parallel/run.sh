#!/bin/bash
set -e
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh  RANK_SIZE"
echo "For example: bash run.sh 8"
echo "This example is expected to run on the Ascend environment."
echo "=============================================================================================================="

if ! [[ $1 =~ ^[1-8]$ ]]; then
  echo "[error]Please run the script as: bash run.sh 2"
  exit 0
fi

mpirun -n $1 python train.py $1