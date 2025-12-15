# 实验简介
本实验在Qwen2.5-7B-Instruct预训练的实验的基础上增加profiler性能调优部分，环境配置，模型下载，数据集下载和预处理，权重转换等步骤参考[pretrain.md](../Chapter1/pretrain.md)
## profiler使能预训练
在预训练脚本`examples/mindspore/qwen25/pretrain_qwen25_7b_32k_ms.sh`配置中添加profiler相关参数


```bash
...
msrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --profile \
    --profile-step-start 5 \
    --profile-step-end 6 \
    --profile-save-path ./logs/prof \
    --profile-level level1 \
    --profile-with-cpu \
    --profile-with-memory \
    --profile-with-stack \
     --profile-ranks -1 \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --ai-framework mindspore \
    | tee logs/pretrain_qwen25_7b_32k.log
```


执行命令
```bash
bash examples/mindspore/qwen25/pretrain_qwen25_7b_32k_ms.sh
```

##  查看结果 
运行日志保存在`./logs/prof`中   
MindStudio-Insight下载链接：https://www.hiascend.com/developer/download/community/result?module=pt+sto+cann   
MindStudio-Insight使用方法：https://www.mindspore.cn/tutorials/zh-CN/r2.7.0/debug/profiler.html#%E6%A6%82%E8%A7%88%E7%95%8C%E9%9D%A2%E6%80%BB%E8%A7%88%E6%95%B0%E6%8D%AE%E6%83%85%E5%86%B5