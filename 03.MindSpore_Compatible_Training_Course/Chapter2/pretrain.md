# 实验简介

本实验以Qwen2.5-7B-Instruct模型和alpaca数据集为例, 展示使用MindSpore和MindSpeed-Core-MS进行模型预训练的流程。

# 环境准备

本实验使用的主要环境信息如下:

| 依赖软件              | 版本     |
|:------------------|:-------|
| CANN              | 8.2RC1 |
| Python            | 3.10   |
| MindSpore         | 2.7.1  |
| MindSpeed-Core-MS | r0.4.0 |

[dockerfile_unified](../dockerfiles/dockfile_unified)中打入了**CANN**与**Python**, 开发者可基于此镜像或任何包含指定**CANN**和**Python**版本的环境中完成实验。
其他依赖安装参考以下步骤。

## MindSpore安装

安装MindSpore 2.7.1版本，安装教程请参考[MindSpore快速安装](https://www.mindspore.cn/install)。

执行以下命令

```bash

python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"

```
如果输出

```text

MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!

```
说明MindSpore安装成功。


## MindSpeed-Core-MS及相关依赖安装


```bash

# 安装MindSpeed-Core-MS转换工具
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git  -b r0.4.0

# 使用MindSpeed-Core-MS内部脚本提供配置环境
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh llm
```

# Qwen2.5-7B-Instruct预训练流程

## 权重转换

1. 权重以及模型文件下载

执行以下命令下载

```bash

pip install modelscope
cd MindSpeed-LLM
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./qwen2.5_7b_hf

```

2. 权重转换
 
在权重转换脚本`examples/mindspore/qwen25/ckpt_convert_qwen25_hf2mcore.sh`中配置`tp并行大小--target-tensor-parallel-size`，`pp并行大小--target-pipeline-parallel-size`，`模型加载路径--load-dir`，`模型保存路径--save-dir`，`模型tokenizer路径--tokenizer-model`，参考脚本如下：

```bash
python mindspeed_llm/mindspore/convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 2 \
       --target-pipeline-parallel-size 2 \
       --add-qkv-bias \
       --load-dir ./qwen2.5_7b_hf/ \
       --save-dir ./model_weights/qwen2.5_mcore/ \
       --tokenizer-model ./qwen2.5_7b_hf/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16 \
       --ai-framework mindspore \
```

执行以下命令权重转换

```bash

cd MindSpeed-LLM
bash examples/mindspore/qwen25/ckpt_convert_qwen25_hf2mcore.sh

```


运行脚本后，预期会看到类似以下的日志输出，表示权重转换成功：

```bash

successfully saved checkpoint from iteration 1 to ./model_weights/qwen2.5_mcore/
INFO:root:Done!

```
## 数据预处理(以Alpaca数据集为例)

1. 数据集下载

数据集下载可以基于[网页](https://huggingface.co/datasets/tatsu-lab/alpaca/tree/main/data)直接下载，也可以基于以下命令行下载

```bash

mkdir dataset
cd dataset/
wget https://hf-mirror.com/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

```
2. 数据集预处理

在预训练数据预处理脚本examples/mindspore/qwen25/data_convert_qwen25_pretrain.sh中配置好数据输入路径--input，数据输出路径--output-prefix、tokenizer模型路径--tokenizer-name-or-path， 参考脚本如下：

```bash
python ./preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./qwen2.5_7b_hf/ \
        --output-prefix ./dataset/alpaca \
        --tokenizer-type PretrainedFromHF \
        --workers 4 \
        --log-interval 1000
```
执行以下命令处理数据集

```bash

bash examples/mindspore/qwen25/data_convert_qwen25_pretrain.sh

```

预训练数据集处理结果如下

```bash

./dataset/alpaca_text_document.bin
./dataset/alpaca_text_document.idx

```
> 注意：预训练时，数据集路径 --data-path 参数传入 ./dataset/alpaca_text_document 即可

## 预训练

预训练脚本examples/mindspore/qwen25/pretrain_qwen25_7b_32k_ms.sh中配置模型加载路径CKPT_LOAD_DIR，模型保存路径CKPT_SAVE_DIR，数据集路径DATA_PATH，tokenizer路径TOKENIZER_PATH

场景一：无重计算参考脚本如下

```bash
CKPT_LOAD_DIR="./model_weights/qwen2.5_mcore/iter_0000001"
CKPT_SAVE_DIR="./pretrain_ckpt/nonrecompute"
DATA_PATH="./dataset/alpaca_text_document"
TOKENIZER_PATH="./qwen2.5_7b_hf"

TP=2
PP=2
SEQ_LEN=10240
MBS=1
GBS=64

GPT_ARGS="
    ...
    --train-iters 3 \
    ...
"

msrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --ai-framework mindspore \
    | tee logs/pretrain_qwen25_7b_32k.log
```

场景二：重计算参考脚本如下


```bash
CKPT_LOAD_DIR="./model_weights/qwen2.5_mcore/iter_0000001"
CKPT_SAVE_DIR="./pretrain_ckpt/compute"
DATA_PATH="./dataset/alpaca_text_document"
TOKENIZER_PATH="./qwen2.5_7b_hf"

TP=2
PP=2
SEQ_LEN=10240
MBS=1
GBS=64

GPT_ARGS="
    ...
    --train-iters 3 \
    ...
    --recompute-granularity full \     #开启完全重计算 
    --recompute-method uniform \  #重计算方式为uniform
    --recompute-num-layers 1 \  #每1层进行一次重计算
"

msrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --ai-framework mindspore \
    | tee logs/pretrain_qwen25_7b_32k.log
```

执行以下命令进行预训练


```bash

bash examples/mindspore/qwen25/pretrain_qwen25_7b_32k_ms.sh

```
> 注意：该脚本中的TP/PP需要和权重转换脚本ckpt_convert_qwen25_hf2mcore.sh中的TP/PP保持一致


查看日志信息

运行日志保存在./worker_x.log中