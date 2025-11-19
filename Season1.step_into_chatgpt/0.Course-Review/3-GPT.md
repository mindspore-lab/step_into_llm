# GPT 课程回顾

预训练语言模型主要分为两条技术路线，一条是上节课讲解的，以BERT为代表的Encoder-Only架构，另一条便是我们本节课介绍的，以GPT为代表的Decoder-Only结构。

GPT-1是更早于BERT提出了预训练语言模型（Pre-train+Fine-tune）的思路，并以此为七点，后续衍生出GPT-2、GPT-3等模型，正式开启了大型语言模型的时代。

接下来我们对课程进行简单回顾，迎接下一节公开课的进一步深入。

## 1. 课程回顾

- Semi-Supervised Learning
- Unsupervised Pretraining
    - 模型预训练优化目标
    - 模型结构
- Supervised Fine-tuning
    - 模型finetuning优化目标
    - 下游任务及对应的输入处理

## 2. 课程实践

课程中我们讲解了如何finetune GPT模型实现分类（classification）任务，大家可以在另三类下游任务中选择一种进行实践，注意不同的下游任务需要对模型进行不同的处理。

<div align="center"><img src="..\3.GPT\assets\finetune_tasks.png" alt="gpt-finetune"></div>