#!/bin/bash

echo "======================================================================================================"
echo "Please run the script as: "
echo "bash run.sh RANK_SIZE"
echo "For example, bash run.sh 8"
echo "======================================================================================================"

if ! [[ $1 =~ ^[1-8]$ ]]; then
  echo "[error]Please run the script as: bash run.sh 2"
  exit 0
fi

cur_path=$(pwd)
export DATA_PATH=${cur_path}/cifar-10-batches-bin/

if [ -d "${DATA_PATH}" ]; then
    echo "data_path:${DATA_PATH}"
else
    echo "download http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
    tar xvf cifar-10-binary.tar.gz
fi

# 检测是否有pytest，安装pytest
pip list |grep pytest || pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple

rm -rf device
mkdir device
cp ./resnet50_distributed_training.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n $1 pytest -s -v ./resnet50_distributed_training.py > train.log 2>&1 &
