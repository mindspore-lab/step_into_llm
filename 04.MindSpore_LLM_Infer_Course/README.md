<div align=center>
  <h1>vLLM-MindSpore推理部署课程</h1>
  <p><a href="./README.md">View English</a></p>
</div>

本课程旨在帮助开发者快速掌握基于MindSpore Transformers和vLLM-MindSpore的大模型推理部署技术，涵盖从环境搭建、服务化启动到高级特性配置的全流程实践。

## 📢 最新消息

- 2025-12-21 「课程更新」：新增vLLM-MindSpore服务化部署课程，支持Multi-LoRA与Function Call特性。（[查看详情](#)）
- 2025-12-18 「功能优化」：项目仓完成重构，优化文档结构，新增混合并行部署示例。（[查看详情](#)）
- 2025-12-10 「Bug修复」：修复vLLM-MindSpore启动脚本中的参数解析问题，感谢社区贡献。（[查看详情](#)）

## 前置知识

在学习本门课程之前，您需要掌握：

- Python基础
- Linux命令基础
- Jupyter基础
- Docker镜像使用

您可以通过前置学习考试（*待上线*）进行自检。

## 环境准备

为确保项目仓中实践代码可正常运行，推荐以下环境准备方式。更详细的环境准备指导详见[Wiki](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/wiki/Set-Up-Development-Environment)。

### 直接安装依赖

请先确保 Python 版本符合[课程要求](#版本维护)后，进入仓库根目录，执行：

```bash
pip install requirements.txt
```

### 使用Docker镜像（*待发布*）

为方便开发者更加便捷地进行代码实践，节约环境准备的时间，我们提供了预装好的基础Dockerfile文件。课程的所有镜像可从[dockerfile](./dockerfile/)获取。本课程镜像文件信息如下，开发者可根据实际需求进行拉取：

镜像基础使用教程详见环境准备Wiki中的[Docker镜像使用](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/wiki/Set-Up-Development-Environment)部分。

## 课程内容

| 序号 | 课节    | 简介             | 视频                  |
| :-- | :------ | :--------------- | :----------------------- |
| 1   | vLLM-MindSpore服务化部署 | 学习vLLM-MindSpore的安装、启动及参数配置。 | [观看](https://www.bilibili.com/video/BV1Ys1aBxEFD/?share_source=copy_web&vd_source=fd4588b77d7b0209a532d9279088f606) |
| 2   | 大模型推理高级特性 | 深入理解Chunked Prefill、Prefix Caching等优化技术。 | [观看](https://www.bilibili.com/video/BV1Ys1aBxEFD/?share_source=copy_web&vd_source=fd4588b77d7b0209a532d9279088f606) |
| 3   | 混合并行与量化推理 | 掌握混合并行部署及模型量化推理的最佳实践。 | [观看](https://www.bilibili.com/video/BV1Ys1aBxEFD/?share_source=copy_web&vd_source=fd4588b77d7b0209a532d9279088f606) |

# 服务化部署与评测指导

## 目录
- [安装部署](#安装部署)
- [服务化启动参数](#服务化启动参数)
- [关键特性配置](#关键特性配置)
  - [Prefix Caching](#prefix-caching)
  - [Chunked Prefill](#chunked-prefill)
  - [Multi-LoRA](#multi-lora)
  - [Function Call](#function-call)
- [混合并行服务化部署示例](#混合并行服务化部署示例)
- [量化推理](#量化推理)
- [评测工具使用](#评测工具使用)
  - [精度评测](#精度评测)
  - [性能评测](#性能评测)
- [启动服务](#启动服务)
- [发送请求](#发送请求)


## 安装部署

详细安装步骤请参考官方文档：[安装指南](https://www.mindspore.cn/vllm_mindspore/docs/zh-CN/master/getting_started/installation/installation.html)


## 关键特性配置

### Prefix Caching

#### 示例命令
```bash
python -m vllm.entrypoints.openai.api_server \
--model /path/to/model \
--tensor-parallel-size=2 \
--enable-prefix-caching
```

### Chunked Prefill

#### 示例命令

```bash
export VLLM_USE_V1=0

vllm-mindspore serve Qwen3-8B \
  --trust-remote-code \
  --max-num-seqs 32 \
  --max-model-len 16384 \
  --block-size 128 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill
```

### Multi-LoRA
#### Multi-LoRA服务化部署示例

**启动服务**
```bash
vllm-mindspore serve /home/ckpt/qwen2.5-7b-hf \
--tensor_parallel_size=1 \
--gpu-memory-utilization=0.9 \
--max_model_len=800 \
--enable-lora \
--lora-modules lora1='/home/ckpt/qwen2.5-7b-lora-law' lora2='/home/ckpt/qwen2.5-7b-lora-medical' \
--max_loras=3 \
--max_lora_rank=64
```

**服务化示例**
```bash
# 请求1
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d \
'{"model": "/home/ckpt/qwen2.5-7b-hf", "prompt": "违章停车与违法停车是否有区别？", \
"model": "lora1", "max_tokens": 20, "temperature": 0, "top_p": 1.0, "top_k": -1, \
"repetition_penalty": 1.0}'

# 请求2
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d \
'{"model": "/home/ckpt/qwen2.5-7b-hf", "prompt": "血液有哪些成分？", "model": "lora2", \
"max_tokens": 20, "temperature": 0, "top_p": 1.0, "top_k": -1, \
"repetition_penalty": 1.0}'
```

### Function Call
#### Function Call 服务化部署示例

**启动服务**
```bash
vllm-mindspore serve /hf_quant_ckpt/V3-0324-A8W8 \
--trust_remote_code \
--tensor_parallel_size=16 \
--max-num-seqs=192 \
--max_model_len=65536 \
--max-num-batched-tokens=4096 \
--block-size=128 \
--gpu-memory-utilization=0.9 \
--quantization golden-stick \
--enable-chunked-prefill \
--enable-auto-tool-choice \
--tool-call-parser deepseek_v3 \
--chat-template tool_chat_template.jinja
```

## 混合并行服务化部署示例

#### 示例1：Ray启动
**Step1：主节点启动ray**
```bash
ray start --head --port=<port-to-ray>
```

**Step2：从节点通过ray链接主节点**
```bash
ray start --address=<head_node_ip>:<port>
ray status
```

**Step3：主节点启动服务**
```bash
vllm-mindspore serve /data/dsr1-w8a8 \
--trust-remote-code \
--max-num-seqs=256 \
--max-model-len=16384 \
--max-num-batched-tokens 2048 \
--block-size=128 \
--gpu-memory-utilization=0.93 \
--tensor-parallel-size 4 \
--data-parallel-size 4 \
--data-parallel-size-local 2 \
--data-parallel-backend=ray \
--enable-expert-parallel \
--additional-config '{"expert_parallel": 4}' \
--quantization ascend
```

#### 示例2：Multiprocess启动
**Step1：主节点启动命令**
```bash
vllm-mindspore serve /data/dsr1-w8a8/ \
--trust_remote_code --host 0.0.0.0 --port 8000 \
--max-num-seqs 256 --max-num-batched-tokens 2048 \
--max_model_len 16384 --block-size 128 \
--gpu-memory-utilization 0.93 --quantization ascend \
--tensor-parallel-size 4 --data-parallel-size 4 \
--data-parallel-size-local 2 --data-parallel-start-rank 0 \
--data-parallel-address {head_node_ip} \
--data-parallel-rpc-port 8888 \
--enable-expert-parallel \
--additional-config '{"expert_parallel": 4}'
```

**Step2：从节点启动命令**
```bash
vllm-mindspore serve /data/dsr1-w8a8/ \
--trust_remote_code --host 0.0.0.0 --port 8000 \
--max-num-seqs 256 --max-num-batched-tokens 2048 \
--max_model_len 16384 --block-size 128 \
--gpu-memory-utilization 0.93 --quantization ascend \
--tensor-parallel-size 4 --data-parallel-size 4 \
--data-parallel-size-local 2 --data-parallel-start-rank 2 \
--data-parallel-address {head_node_ip} \
--data-parallel-rpc-port 8888 \
--enable-expert-parallel --headless \
--additional-config '{"expert_parallel": 4}'
```


## 量化推理

#### 调用命令

**金箍棒量化的权重：**
```bash
vllm-mindspore serve Qwen3-32B-gs-A8W8 \
--port 8888 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.75 \
--max-num-batched-tokens 4096 \
--block_size 32 \
--quantization golden-stick
```

**ModelSlim量化的权重：**
```bash
vllm-mindspore serve Qwen3-32B-modelslim \
--port 8888 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.75 \
--max-num-batched-tokens 4096 \
--block_size 32 \
--quantization ascend
```

## 评测工具使用

ais_bench 是昇腾提供的 AI 性能测试工具，用于对模型进行基准测试。

**仓库地址：** [https://gitee.com/aisbench/benchmark](https://gitee.com/aisbench/benchmark)

### 精度评测
```bash
# 修改配置文件
# 配置文件位于 ais_bench 仓库中：ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
# 建议先克隆仓库并安装依赖

# 启动评测
# run_benchmark.py 是工具包中的主程序脚本
python run_benchmark.py --models vllm_api_general --datasets gsm8k_gen_0_shot_cot_str
```

### 性能评测
```bash
# 修改配置文件
# 配置文件位于 ais_bench 仓库中：ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py

# 启动性能评测
python run_benchmark.py --mode perf
```

## 启动服务
```bash
vllm-mindspore serve ${MODEL_PATH} \
--trust-remote-code \
--max-num-seqs 32 \
--max-model-len 4096 \
--block-size 128 \
--gpu-memory-utilization 0.9
```

## 发送请求
```bash
curl -X POST http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "/home/ckpt/Qwen3-8B",
  "temperature": 0,
  "max_tokens": 20,
  "prompt": "Qwen3 is"
}'
```

## 版本维护

项目随昇思MindSpore及vllm-mindspore迭代同步发布版本，本项目仓每**半年**进行版本发布。

| 版本名  | Python | MindSpore | MindFormers | vllm|
| :----- | :----- |:------ |:------ | :------ |
| r0.3.0 | 3.11    | r2.7.0    | r1.6.0    | v0.8.3|

## 常见问题（FAQ）

详见Wiki中[FAQ](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/wiki/Developer-FAQ)。

## 贡献与反馈

欢迎各位开发者通过 [Issue](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/issues) 提交建议或 bug 反馈，也可直接发起 [PR](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/pulls) 进行Bug修复或代码贡献（提交前请参考提交规范，由Committer @username 完成评审合入），你的每一份参与都能让本项目更加完善。

### 提交规范

详见WIKI：[Issue与PR提交规范](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/wiki/Contributing-Guidelines)

### 贡献者展示

向本项目的贡献者们致以最诚挚的感谢！

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/mindspore-lab/step_into_llm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=mindspore-lab/step_into_llm" />
  </a>
</div>
