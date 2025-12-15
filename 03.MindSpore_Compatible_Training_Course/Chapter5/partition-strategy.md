# 实验简介
 本实验在Qwen2.5-7B-Instruct预训练的实验的基础上增加模型切分策略调优部分，环境配置，模型下载，数据集下载和预处理等步骤参考[pretrain.md](../Chapter1/pretrain.md)

##  权重转换
修改权重转换脚本`examples/mindspore/qwen25/ckpt_convert_qwen25_hf2mcore.sh`，以TP2PP4为例，不同切分策略主要修改target-tensor-parallel-size，target-pipeline-parallel-size的大小和模型保存路径save-dir
- TP2PP4参考脚本如下：
```bash
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 2 \
       --target-pipeline-parallel-size 4 \
       --add-qkv-bias \
       --load-dir ./model_from_hf/qwen2.5_7b_hf/ \
       --save-dir ./convert-model/tp2pp4/ \
       --tokenizer-model ./model_from_hf/qwen2.5_7b_hf/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16
```

- TP2PP4+VPP参考脚本如下：
```bash
python convert_ckpt.py \
       ... 
       --target-tensor-parallel-size 2 \
       --target-pipeline-parallel-size 4 \
       --save-dir ./convert-model/tp2pp4_vpp/ \
       --noop-layers 2,3,5,8 \
       ...
```



执行命令
```bash
bash examples/mindspore/qwen25/ckpt_convert_qwen25_hf2mcore.sh
```

## 预训练
在预训练脚本`examples/mindspore/qwen25/pretrain_qwen25_7b_32k_ms.sh`中配置模型加载路径`CKPT_LOAD_DIR`，模型保存路径`CKPT_SAVE_DIR`，数据集路径`DATA_PATH`，tokenizer路径`TOKENIZER_PATH`

- TP2PP4参考脚本如下：
```bash
CKPT_LOAD_DIR="./convert-model/tp2pp4/iter_0000001"
CKPT_SAVE_DIR="./output-model/tp2pp4"
DATA_PATH="./dataset/alpaca_text_document"
TOKENIZER_PATH="./model_from_hf/qwen2.5_7b_hf"

TP=2
PP=4
SEQ_LEN=4096

GPT_ARGS="
    ...
    --train-iters 10 \
    ...
"
```
- TP4PP2参考脚本如下：
```bash
CKPT_LOAD_DIR="./convert-model/tp4pp2/iter_0000001"
CKPT_SAVE_DIR="./output-model/tp4pp2"
DATA_PATH="./dataset/alpaca_text_document"
TOKENIZER_PATH="./model_from_hf/qwen2.5_7b_hf"

TP=4
PP=2
SEQ_LEN=4096

GPT_ARGS="
    ...
    --train-iters 10 \
    ...
"
```
- TP4PP2重计算参考脚本如下：
```bash
CKPT_LOAD_DIR="./convert-model/tp4pp2/iter_0000001"
CKPT_SAVE_DIR="./output-model/tp4pp2"
DATA_PATH="./dataset/alpaca_text_document"
TOKENIZER_PATH="./model_from_hf/qwen2.5_7b_hf"

TP=4
PP=2
SEQ_LEN=4096

GPT_ARGS="
    ...
    --train-iters 10 \
    ...
    --bf16 \
    --recompute-granularity full \   
    --recompute-method uniform \  
    --recompute-num-layers 1 \  
"
```

- TP2PP4+VPP参考脚本如下：  
Qwen2.5-7B-Instruct模型的num-layers为28，无法被TP,PP,VPP相乘的大小16整除，需要插4个空层增加到32
```bash
CKPT_LOAD_DIR="./convert-model/tp2pp4_vpp/iter_0000001"
CKPT_SAVE_DIR="./output-model/tp2pp4"
DATA_PATH="./dataset/alpaca_text_document"
TOKENIZER_PATH="./model_from_hf/qwen2.5_7b_hf"

TP=2
PP=4
SEQ_LEN=4096

GPT_ARGS="
    ...
    --num-layers 32  \
    --train-iters 10 \
    ...
    --bf16 \
    --noop-layers 2,3,5,8 \
    --num-layers-per-virtual-pipeline-stage 2
"
```

- TP2PP2+RingAttention参考脚本如下：
```bash
CKPT_LOAD_DIR="./convert-model/tp2pp2/iter_0000001"
CKPT_SAVE_DIR="./output-model/tp2pp_Ring_Attention"
DATA_PATH="./dataset/alpaca_text_document"
TOKENIZER_PATH="./model_from_hf/qwen2.5_7b_hf"

TP=2
PP=2
SEQ_LEN=16384

GPT_ARGS="
    ...
    --train-iters 10 \
    ...
    --bf16 \
    --context-parallel-size 2 \
    --seq-length 16384 \
    --use-cp-send-recv-overlap \
    --context-parallel-algo megatron_cp_algo
"
```

- TP2PP2+Ulysses参考脚本如下：
```bash
CKPT_LOAD_DIR="./convert-model/tp2pp2/iter_0000001"
CKPT_SAVE_DIR="./output-model/tp2pp_Ulysses"
DATA_PATH="./dataset/alpaca_text_document"
TOKENIZER_PATH="./model_from_hf/qwen2.5_7b_hf"

TP=2
PP=2
SEQ_LEN=16384

GPT_ARGS="
    ...
    --train-iters 10 \
    ...
    --bf16 \
    --context-parallel-size 2 \
    --context-parallel-algo ulysses_cp_algo
"
```

执行命令
```bash
bash examples/mindspore/qwen25/pretrain_qwen25_7b_32k_ms.sh
```


运行日志保存在`MindSpeed-LLM/worker_7.log`中

##  实验结果
| 配置 | 吞吐量 | 峰值显存 | 主要影响因素 |
| :--- | :--- | :--- | :--- |
| TP2PP4 | 119 | 35366 | 模型被切成4段，每段之间的气泡比例更大，缺乏张量并行，导致计算不均衡，拖慢速度。 |
| TP2PP4+VPP | 126 | 37087 | VPP 优化了流水线调度，减少了气泡，吞吐量略高于 tp2pp4；显存占用因算法开销略有增加。 |
| TP4PP2 | 126 | 27951 | PP=2 显著减少了流水线气泡，TP=4 提高了计算效率，吞吐量得到提高；显存占用因 PP 减少而大幅降低。 |
| TP4PP2+Recompute | 99 | 26779 | 重计算以时间换空间，显著降低了显存占用（低于 tp4pp2），但增加了前向计算量，导致吞吐量明显下降。 |

