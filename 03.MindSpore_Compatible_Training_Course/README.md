<div align=center>
  <h1>MindSpore兼容式分布式训练原理和实践</h1>

  <p><a href="./README.md">View English</a></p>
</div>

本课程围绕大模型预训练、微调、强化学习等场景，介绍了基于MindSpeed-Core-MS套件的全流程开发过程，使得学员能够了解MindSpeed-Core-MS相关概念、特性和开发流程，并初步具备在不同场景下大模型训练的能力。

## 📢 最新消息

- 2025-11-30 「课程更新」：新增章节1-5，包含完整视频、课件及代码案例。（[查看详情](xxxx)）

[//]: # (- 2025-10-18 「功能优化」：项目仓完成重构，查找课程资源更清晰，新增PR检查门禁，合入内容更规范。（[查看详情]&#40;xxx&#41;）)

[//]: # (- 2025-10-10 「Bug修复」：修复xxxxxx问题，感谢@username的PR贡献。（[查看详情]&#40;xxxx&#41;）)

## 前置要求

本课程为MindSpore系列课程的中级课程, 读者在掌握Transformers结构、单机上模型微调后学习本门课程更佳。

## 环境准备

该课程使用的Docker环境可从[dockerfiles](./dockerfiles/)获取。
主要环境信息如下:

| 环境组件                                                                        | 版本信息   |
|:----------------------------------------------------------------------------------------------|:-------|
| [CANN](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1) | 8.2RC1 |
| Python                                                                                        | \>=3.9 |
| [MindSpeed-Core-MS](https://gitcode.com/Ascend/MindSpeed-Core-MS/tree/r0.4.0)                 | r0.4.0 |


## 课程内容

| 序号 | 课节                            | 简介                                   | 课程资源                                              |
|:---|:------------------------------|:-------------------------------------|:--------------------------------------------------|
| 1  | MindSpore兼容式大模型训推套件概览         | 介绍MindSpore兼容方案架构与能力                 | [PPT](./Chapter1) · [手册](./Chapter1) · [视频](跳转链接) |
| 2  | 基于MindSpeed-Core-MS的模型预训练实践   | 介绍使用MindSpeed-Core-MS进行大模型预训练的流程与实践  | [PPT](./Chapter2) · [手册](./Chapter2) · [视频](跳转链接) |
| 3  | 基于MindSpeed-Core-MS指令微调原理与实践  | 介绍使用MindSpeed-Core-MS进行大模型指令微调的流程与实践 | [PPT](./Chapter3) · [手册](./Chapter3) · [视频](跳转链接) |
| 4  | 基于MindSpeed-Core-MS的强化学习原理与实践 | 介绍使用MindSpeed-Core-MS进行强化学习的流程与实践    | [PPT](./Chapter4) · [手册](./Chapter4) · [视频](跳转链接) |
| 5  | 内存调优&性能调优介绍与实践 | 介绍内存调优与性能调优的方法论与实践对比                 | [PPT](./Chapter5) · [手册](./Chapter5) · [视频](跳转链接) |

## 版本维护

项目随昇思[MindSpore](https://www.mindspore.cn/install)及[MindSpeed-Core-MS](https://gitcode.com/Ascend/MindSpeed-Core-MS/tree/master)代同步发布版本，本项目仓每**半年**进行版本发布。

| 版本名  | Python | MindSpore | MindSpeed-Core-MS |
| :----- |:-------|:----------|:------------------|
| master | \>=3.9 | 2.7.1     | r0.4.0            |


## 常见问题（FAQ）

详见Wiki中[FAQ](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/wiki/FAQ)。

## 贡献与反馈

欢迎各位开发者通过 [Issue](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/issues) 提交建议或 bug 反馈，也可直接发起 [PR](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/pulls) 进行Bug修复或代码贡献（提交前请参考提交规范，由Committer @username 完成评审合入），你的每一份参与都能让本项目更加完善。

### 提交规范

详见WIKI：[Issue与PR提交规范](https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/wiki/%E6%8F%90%E4%BA%A4%E8%AF%B4%E6%98%8E)

### 贡献者展示

向本项目的贡献者们致以最诚挚的感谢！

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=mindspore-courses/MindSpore-Compatible-Distributed-Training-Principles-and-Practices" />
  </a>
</div>
