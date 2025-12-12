<div align="center">
  <h1>昇思MindSpore大模型专题课程</h1>


</div>

### 课程特色
- ***探究前沿***：解读技术热点，解构热点模型
- ***应用实践***：理论实践相结合，手把手指导开发
- ***专家解读***：多领域专家，多元解读
- ***开源共享***：课程免费，课件代码开源

### 📢 最新消息

- 2025-12-20 「课程更新」：新增DeepSeek系列模型解读（[查看详情](xxxx)）

### 教研团队

<div align="center"><img src="./assets/teachers.jpg" alt="teachers"></div>

### 课程介绍

本课程由浅入深地逐步深入大模型技术，以模型为主线介绍经典模型的结构、训练、推理中的创新技术，同时结合MindSpore大模型套件进行实践。
本课程包含两部分：

一、业界模型介绍：深入剖析业界模型结构、训练、推理等方面的创新技术。

1. 经典模型技术剖析:

| 章节序号 | 章节名称 | 课程简介                                        | 视频 |                                                课件及代码                          |
|:----:|:----:|:--------------------------------------------|:----:|:---------------------------------------------------------------------------------------------------:|
| 第一讲  |       BERT        | 基于Transformer Encoder的BERT模型设计：MLM和NSP任务。BERT进行下游任务微调的范式。                                                                               | [link](https://www.bilibili.com/video/BV1xs4y1M72q/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) |                             [link](./01.Industry_Model_Introduction/01.Classic_Model_Technical_Analysis/01.BERT/) | 
| 第二讲  |        GPT        | 基于Transformer Decoder的GPT模型设计：Next token prediction。GPT下游任务微调范式。| [link](https://www.bilibili.com/video/BV1Gh411w7HC/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](./01.Industry_Model_Introduction/01.Classic_Model_Technical_Analysis/02.GPT/)                              | 
| 第三讲  |       GPT2        | GPT2的核心创新点，包括Task Conditioning和Zero shot learning；模型实现细节基于GPT1的改动。                                                                      |  [link](https://www.bilibili.com/video/BV1Ja4y1u7xx/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) |                             [link](./01.Industry_Model_Introduction/01.Classic_Model_Technical_Analysis/03.GPT2/)                             |
| 第四讲  |      ChatGLM      | GLM模型结构，从GLM到ChatGLM的演变，ChatGLM推理部署代码演示| [link](https://www.bilibili.com/video/BV1ju411T74Y/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429)  |                             [link](./01.Industry_Model_Introduction/01.Classic_Model_Technical_Analysis/04.ChatGLM/)                             |
| 第五讲  |     ChatGLM2      | ChatGLM2技术解析，ChatGLM2推理部署代码演示，ChatGLM3特性介绍| [link](https://www.bilibili.com/video/BV1Ew411W72E/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429)  |                            [link](./01.Industry_Model_Introduction/01.Classic_Model_Technical_Analysis/05.ChatGLM2/)                             |
| 第六讲  |       LLAMA       | LLaMA背景及羊驼大家族介绍，LLaMA模型结构解析，LLaMA推理部署代码演示| [link](https://www.bilibili.com/video/BV1nN41157a9/?spm_id_from=333.999.0.0) |                              [link](./01.Industry_Model_Introduction/01.Classic_Model_Technical_Analysis/06.LLaMA)                              |  [link](https://mp.weixin.qq.com/s/9QdP062-agcIbsR0_a-b3g)  |
| 第七讲  |      LLAMA2       | 介绍LLAMA2模型结构，走读代码演示LLAMA2 chat部署| [link](https://www.bilibili.com/video/BV1Me411z7ZV/?spm_id_from=333.999.0.0) |                             [link](./01.Industry_Model_Introduction/01.Classic_Model_Technical_Analysis/07.LLaMA2/)                              |                             
| 第八讲  |      CPM-Bee      | 介绍CPM-Bee预训练、推理、微调及代码现场演示 |[link](https://www.bilibili.com/video/BV1VZ4y1n7t9/?spm_id_from=333.999.0.0)  | [link](./01.Industry_Model_Introduction/01.Classic_Model_Technical_Analysis/08.CPM-Bee/) |
| 第九讲  |      RWKV1-4      | RNN的没落和Transformers的崛起 万能的Transformers？Self-attention的弊端 “拳打”Transformer的新RNN-RWKV 基于MindNLP的RWKV模型实践 |   [link](https://www.bilibili.com/video/BV1K4421w7Ha/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) |                                                  [link](./01.Industry_Model_Introduction/01.Classic_Model_Technical_Analysis/09.RWKV/ )                                               |
| 第十讲 |        MOE        | MoE的前世今生 MoE的实现基础：AlltoAll通信； Mixtral 8x7b: 当前最好的开源MoE大模型，MoE与终身学习，基于昇思MindSpore的Mixtral 8x7b推理演示。           |   [link](https://www.bilibili.com/video/BV1jH4y177DL/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](./01.Industry_Model_Introduction/01.Classic_Model_Technical_Analysis/10.Mixtral/) |

2. 伙伴创新模型分享:

|  章节序号  |            章节名称            | 课程简介                                        | 视频 | 课件及代码 |
|:------:|:--------------------------:|:--------------------------------------------|:----:|:----:|
|  第一讲   |          CodeGeex          | 代码预训练发展沿革。Code数据的预处理。CodeGeex代码预训练大模型。                                                                                                  |  [link](https://www.bilibili.com/video/BV1Em4y147a1/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](./01.Industry_Model_Introduction/02.Partner_Innovation_Model_Sharing/01.CodeGeeX/) |
|  第二讲   |        多模态预训练大模型           | 紫东太初多模态大模型的设计、数据处理和优势；语音识别的理论概述、系统框架和现状及挑战。  | [link](https://www.bilibili.com/video/BV1wg4y1K72r/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | / |
|  第三讲   |       多模态遥感智能解译基础模型        | 本次课程由中国科学院空天信息创新研究院研究员 实验室副主任 孙显老师讲解多模态遥感解译基础模型，揭秘大模型时代的智能遥感技术的发展与挑战、遥感基础模型的技术路线与典型场景应用| [link](https://www.bilibili.com/video/BV1Be41197wY/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429)  | /|
|  第四讲   |            鹏城脑海            | 鹏城·脑海200B模型是具有2千亿参数的自回归式语言模型，在中国算力网枢纽节点'鹏城云脑II'千卡集群上基于昇思MindSpore的多维分布式并行技术进行长期大规模训练。模型聚焦中文核心能力，兼顾英文和部分多语言能力，目前完成了1.8T token量的训练 | [link](https://www.bilibili.com/video/BV1AT4y1p7bJ/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429)   |  / |
|  第五讲   | 书生.浦语大模型开源全链工具链简介与智能体开发体验  | 在本期课程中，我们有幸邀请到了书生.浦语社区技术运营、技术布道师闻星老师，以及昇思MindSpore技术布道师耿力老师，来详细解读书生.浦语大模型开源全链路工具链，演示如何对书生.浦语进行微调、推理以及智能体开发实操。| [link](https://www.bilibili.com/video/BV1K4421w7Ha/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) |    /     | 

二、技术专题介绍：围绕某一技术主题做全面、深入的讲解。

| 章节序号 | 章节名称 | 课程简介                                        | 视频 | 课件及代码 |
|:----:|:----:|:--------------------------------------------|:----:|:----:|
| 第一讲  |   Prompt Tuning   | Pretrain-finetune范式到Prompt tuning范式的改变。Hard prompt和Soft prompt相关技术。只需要改造描述文本的prompting。                                                 | [link](https://www.bilibili.com/video/BV1Wg4y1K77R/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](./02.Technical_Topic_Introduction/01.Prompt_Tuning/) |
| 第二讲  |  Instruct Tuning  | Instruction tuning的核心思想：让模型能够理解任务描述（指令）。Instruction tuning的局限性：无法支持开放域创新性任务、无法对齐LM训练目标和人类需求。Chain-of-thoughts：通过在prompt中提供示例，让模型“举一反三”。 | [link](https://www.bilibili.com/video/BV1cm4y1e7Cc/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](./02.Technical_Topic_Introduction/02.Instruct_Tuning/) |
| 第三讲  |Prompt Engineering | Prompt engineering：1.什么是Prompt？2.如何定义一个Prompt的好坏或优异? 3.如何撰写优质的Prompt？4.如何产出一个优质的Prompt？ 5.浅谈一些我们在进行Prompt的时候遇到的问题。   | [link](https://www.bilibili.com/video/BV1aD421W73q/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) |  / |
| 第四讲  |       RLHF        | RLHF核心思想：将LLM和人类行为对齐。RLHF技术分解：LLM微调、基于人类反馈训练奖励模型、通过强化学习PPO算法实现模型微调。                                                                     | [link](https://www.bilibili.com/video/BV15a4y1c7dv/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](./02.Technical_Topic_Introduction/04.RLHF/) |
| 第五讲  |     文本生成解码原理      | 以MindNLP为例，讲解搜索与采样技术原理和实现| [link](https://www.bilibili.com/video/BV1QN4y117ZK/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429)  | [link](./02.Technical_Topic_Introduction/05.Text_Generation_Decoding/) |
| 第六讲  |     高效参数微调      | 介绍Lora、（P-Tuning）原理及代码实现 | [link](https://www.bilibili.com/video/BV11D421j7fZ/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429)  | [link](/02.Technical_Topic_Introduction/06.PEFT/) |
| 第七讲  |   MindSpore自动并行   | 以MindSpore分布式并行特性为依托的数据并行、模型并行、Pipeline并行、内存优化等技术。                                                                                      |  [link](https://www.bilibili.com/video/BV1VN41117AG/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](./02.Technical_Topic_Introduction/07.Parallel/) |
| 第八讲  |  多维度混合并行自动搜索优化策略  | 议题一·时间损失模型及改进多维度二分法/议题二·APSS算法应用   | [上](https://www.bilibili.com/video/BV1if421X7jB/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429)   [下](https://www.bilibili.com/video/BV1QM4m1z7FV/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | / |

## 贡献与反馈

欢迎各位开发者通过 [Issue](https://github.com/mindspore-lab/step_into_llm/issues) 提交建议或 bug 反馈，也可直接发起 [PR](https://github.com/mindspore-lab/step_into_llm/pulls) 进行Bug修复或代码贡献（提交前请参考提交规范，由Committer @username 完成评审合入），你的每一份参与都能让本项目更加完善。

### 提交规范

详见WIKI：[Issue与PR提交规范](https://github.com/mindspore-lab/step_into_llm/wiki/Contributing-Guidelines)

### 贡献者展示

向本项目的贡献者们致以最诚挚的感谢！

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/mindspore-lab/step_into_llm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=mindspore-lab/step_into_llm" />
  </a>
</div>