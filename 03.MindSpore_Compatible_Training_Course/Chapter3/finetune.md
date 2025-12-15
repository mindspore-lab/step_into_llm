# 实验简介
本实验以Qwen2.5-VL-7B-Instruct模型的Lora微调为例，展示使用MindSpeed-Core-MS进行微调训练的整体流程
# 环境准备
本实验使用的主要环境信息如下:

| 依赖软件              | 版本     |
|:------------------|:-------|
| CANN              | 8.2RC1 |
| Python            | 3.10   |
| MindSpore         | 2.7.1  |
| MindSpeed-Core-MS | r0.4.0 |

[dockerfile_unified](./dockerfiles/dockfile_unified)中打入了**CANN**与**Python**, 开发者可基于此镜像或任何包含指定**CANN**和**Python**版本的环境中完成实验。
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
git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.4.0

# 使用MindSpeed-Core-MS内部脚本自动拉取相关代码仓并一键适配、提供配置环境
cd MindSpeed-Core-MS
pip install -r requirements.txt
source auto_convert.sh mm
```
# 使用MindSpeed-MM微调Qwen2.5-VL-7B-Instruct模型流程
## 1. 模型下载
```bash
cd MindSpeed-MM
mkdir -p ckpt/hf_path && cd ckpt/hf_path
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir ./Qwen2.5-VL-7B-Instruct
cd ../..
```

## 2. 权重转换
```bash
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
  --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
  --cfg.parallel_config.tp_size 1

# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
  ```

## 3. 数据集准备及处理
- 下载COCO2017数据集[COCO2017](http://images.cocodataset.org/zips/train2017.zip)，并解压到项目目录下的./data/COCO2017文件夹中
- 获取图片数据集的描述文件[LLaVA-Instruct-150K](https://modelscope.cn/datasets/AI-ModelScope/LLaVA-Instruct-150K/resolve/master/llava_instruct_150k.json)，下载至./data/路径下

```bash
mkdir -p data/COCO2017 && cd data/COCO2017
wget http://images.cocodataset.org/zips/train2017.zip
unzip train.zip
cd ..
wget https://modelscope.cn/datasets/AI-ModelScope/LLaVA-Instruct-150K/resolve/master/llava_instruct_150k.json
cd ..
python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py
  ```
- 在./data路径下将生成文件mllm_format_llava_instruct_data.json，目录结构如下：
```bash
├── data
    ├── COCO2017
    │   └──train2017
    │   
    ├── llava_instruct_150k.json
    └── mllm_format_llava_instruct_data.json
```

## 4. 微调


- 运行微调命令
```bash
mkdir save_dir
bash examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b_lora.sh
```
运行日志保存在msrun_log/worker_7.log中