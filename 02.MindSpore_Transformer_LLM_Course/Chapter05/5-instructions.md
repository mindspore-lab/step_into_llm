# 使用指令汇总

## 获取 Hugging Face 权重
```bash
git clone https://hf-mirror.com/Qwen/Qwen3-0.6B
```

## 获取数据集
```bash
mkdir qwen-datasets
cd qwen-datasets

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.idx
```

## 运行任务
```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config xx.yaml"
```

## 合并权重
```bash
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs ./output/strategy \
  --mindspore_ckpt_dir ./output/checkpoint \
  --output_dir ./unified_trained_qwen3_60_sf \
  --file_suffix "60_1" \
  --has_redundancy True \
  --filter_out_param_prefix "adam_" \
  --max_process_num 16
```

## qwen3-0.6B 反转权重
```bash
python toolkit/weight_convert/qwen3/reverse_mcore_qwen3_weight_to_hf.py \
  --mindspore_ckpt_path ./unified_trained_qwen3_60_sf/60_1_ckpt_convert/unified_safe \
  --huggingface_ckpt_path ./hf_sf \
  --num_layers 28 \
  --num_attention_heads 16 \
  --num_query_groups 8 \
  --kv_channels 128 \
  --ffn_hidden_size 3072 \
  --dtype 'bf16'
```