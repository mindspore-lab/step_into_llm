"""
完整的MRPC数据集PEFT训练脚本,修复了标签问题
"""

import os
import argparse
import json
import copy
import mindspore
from mindspore import context, Tensor, ops
from mindspore.dataset import NumpySlicesDataset, SequentialSampler
from mindspore.common.parameter import Parameter
from mindspore.nn import AdamWeightDecay
from mindnlp.engine import Evaluator
from mindnlp.metrics import Accuracy
from mindnlp.common.grad import value_and_grad
from mindnlp.dataset import load_dataset
from mindnlp.transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from mindnlp.peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm

# 导入我们的辅助函数
from fix_mrpc_training import (
    print_dataset_keys,
    improved_forward_fn,
    improved_train_step,
    examine_batch,
    prepare_mrpc_batch
)

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """
            InputExample表示单个输入示例
            包含一个全局唯一标识符（guid）、文本 A（text_a）、可选的文本 B（text_b）和标签（label）
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"    

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label, input_len):
        """
            InputFeatures 表示模型输入特征
            包含输入 ID（input_ids）、注意力掩码（attention_mask）、标记类型 ID（token_type_ids）、标签（label）和输入长度（input_len）
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.input_len = input_len
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def convert_dataset_to_examples(ds):
    """
        Convert dataset to examples.
        将数据集ds转换为 InputExample 实例列表examples
        使用 mindspore.dataset 的迭代器遍历数据集，并将每个样本转换为 InputExample 对象。
    """
    examples = []
    iter0 = ds.create_tuple_iterator()
    for i, (text_a, text_b, label, idx, label_text) in enumerate(iter0):
        examples.append(
            InputExample(guid=i, text_a=str(text_a.asnumpy()), text_b=str(text_b.asnumpy()), label=int(label))
        )
    
    return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
        Truncates a sequence pair in place to the maximum length.
        即保持文本对的意义，同时截断文本对，使其总长度不超过指定的最大长度max_length
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        # 优先选择文本更长的文本进行截断
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, tokenizer, max_seq_length=512):
    """
        将 InputExample 实例列表转换为 InputFeatures 实例列表。
        使用 tokenizer 对文本进行编码，生成模型输入所需的特征。
    """
    features = []

    for ex_index, example in enumerate(examples):
        tokenizer.return_token = True
        tokens_a = tokenizer(example.text_a)
        tokens_b  = None
        if example.text_b:
            tokens_b = tokenizer(example.text_b)
        if tokens_b is not None:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        token_type_ids = []
        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)

        if tokens_b is not None:
            for token in tokens_b[1:]:
                tokens.append(token)
                token_type_ids.append(1)

        tokenizer.return_token=False
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        
        label_id = example.label

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label_id,
                          input_len=input_len)
        )
    return features

def load_examples(tokenizer, max_seq_length, mrpc_datas):
    """load_examples using load_dataset
        加载数据集并转换为模型训练所需的特征：
            首先加载 MRPC 数据集的指定部分（训练或测试）
            然后调用 convert_examples_to_features 函数转换为模型输入所需的特征
    """
    
    train_examples = convert_dataset_to_examples(mrpc_datas)

    features = convert_examples_to_features(train_examples, tokenizer, max_seq_length=max_seq_length)

    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    all_attention_mask = [f.attention_mask for f in features]
    all_token_type_ids = [f.token_type_ids for f in features]
    all_lens = [f.input_len for f in features]
    all_labels = [f.label for f in features]
    dataset = ((all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels))

    return dataset

def get_dataloader_from_ds(ds, batch_size):
    train_sampler = SequentialSampler()  # 应用 SequentialSampler 以顺序方式采样数据
    col_names = ['input_ids', 'attention_mask', 'token_type_ids', 'lens', 'labels']
    train_dataloader = NumpySlicesDataset(ds, sampler=train_sampler, column_names=col_names)  # 使用 NumpySlicesDataset 包装数据集
    train_dataloader = train_dataloader.batch(batch_size)  # 根据指定批次大小 进行 批处理

    return train_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train PEFT model on MRPC dataset")
    parser.add_argument("--save_dir", default="./saved_models", help="Directory to save model checkpoints")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--model_name", default="gpt2", help="Base model name")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    args = parser.parse_args()
    
    # 设置运行模式
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    
    # 测试后保存模型，所以tokenizer需要在这个作用域可用
    global tokenizer
    
    print("加载MRPC数据集...")
    # 加载MRPC数据集
    mrpc_dict = load_dataset("SetFit/mrpc")
    mrpc_train = mrpc_dict['train']
    mrpc_valid = mrpc_dict['validation']
    mrpc_test = mrpc_dict['test']
    
    print("加载tokenizer...")
    # 加载tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    # 添加特殊token
    special_tokens_dict = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"添加了 {num_added_toks} 个特殊token")
    
    print("处理数据集...")
    # 处理数据集
    train_ds = load_examples(tokenizer, args.max_seq_len, mrpc_train)
    valid_ds = load_examples(tokenizer, args.max_seq_len, mrpc_valid)
    test_ds = load_examples(tokenizer, args.max_seq_len, mrpc_test)
    
    # 转换为dataloader
    train_dataloader = get_dataloader_from_ds(train_ds, args.batch_size)
    valid_dataloader = get_dataloader_from_ds(valid_ds, args.batch_size)
    test_dataloader = get_dataloader_from_ds(test_ds, args.batch_size)
    
    print("加载模型...")
    # 加载模型
    model = GPT2ForSequenceClassification.from_pretrained(args.model_name, num_labels=2, force_download=False, use_safetensors=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(model.config.vocab_size + num_added_toks)
    
    # 配置PEFT (LoRA)
    if args.use_lora:
        print("应用LoRA配置...")
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, fan_in_fan_out=True)
        model = get_peft_model(model, peft_config)
        print("可训练参数信息:")
        model.print_trainable_parameters()
    
    # 获取参数
    params = []
    if hasattr(model, "get_trainable_parameters"):
        params = model.get_trainable_parameters()
    elif hasattr(model, "trainable_params"):
        params = model.trainable_params()

    print(f"找到 {len(params)} 个可训练参数")

    # 转换参数
    converted_params = []
    for param in params:
        if hasattr(param, "data") and hasattr(param, "name"):
            converted_param = Parameter(param.data, name=param.name)
        else:
            converted_param = param
        converted_params.append(converted_param)

    print(f"转换后的参数数量: {len(converted_params)}")

    # 使用转换后的参数
    optimizer = AdamWeightDecay(params=converted_params, learning_rate=args.lr)
    metric = Accuracy()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, 'gpt2_mrpc_finetune.ckpt')
    best_ckpt_path = os.path.join(args.save_dir, 'gpt2_mrpc_finetune_best.ckpt')

    # 预处理数据集，移除不需要的'lens'列
    def remove_lens_column(dataset):
        """移除数据集中的'lens'列"""
        columns_to_keep = [col for col in dataset.get_col_names() if col != 'lens']
        return dataset.project(columns=columns_to_keep)

    # 打印数据集列名
    print("训练数据集列名:", train_dataloader.get_col_names())
    
    # 处理数据集
    train_dataset = remove_lens_column(train_dataloader)
    valid_dataset = remove_lens_column(valid_dataloader)
    test_dataset = remove_lens_column(test_dataloader)

    print("处理后训练数据集列名:", train_dataset.get_col_names())
    
    # 定义前向函数(使用改进的版本)
    def forward_fn(model, data):
        """前向计算函数"""
        # 只在第一个批次时启用详细输出
        verbose = getattr(forward_fn, 'first_batch', False)
        if verbose:
            forward_fn.first_batch = False
        
        # 使用我们的改进版本
        return improved_forward_fn(model, data, verbose=verbose)
    
    # 设置第一个批次标志
    forward_fn.first_batch = True

    # 使用value_and_grad包装前向计算函数
    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters)

    # 定义训练步骤(使用改进的版本)
    def train_step(model, optimizer, data):
        """执行一个训练步骤"""
        # 首先处理批次数据,确保包含所需字段
        processed_data = prepare_mrpc_batch(data)
        # 使用改进的训练步骤
        return improved_train_step(model, optimizer, processed_data, grad_fn)

    # 定义评估函数
    def evaluate(model, dataset, metric):
        """评估模型"""
        metric.clear()
        model.set_train(False)
        
        for data in dataset.create_dict_iterator():
            # 处理批次数据,确保包含所需字段
            processed_data = prepare_mrpc_batch(data)
            
            # 确保标签单独传递
            if 'labels' in processed_data:
                labels = processed_data.pop('labels')
            elif 'label' in processed_data:
                labels = processed_data.pop('label')
            else:
                print("警告: 评估数据中没有标签!")
                continue
            
            try:
                # 前向计算
                outputs = model(**processed_data)
                
                # 更新指标
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, "logits"):
                    # 处理SequenceClassifierOutputWithPast对象
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                metric.update(logits, labels)
            except Exception as e:
                print(f"评估时出错: {e}")
                continue
        
        # 计算指标
        result = metric.eval()
        model.set_train(True)
        
        return result

    # 训练循环
    model.set_train(True)
    best_acc = 0
    total_steps = args.num_epochs * train_dataset.get_dataset_size()

    print(f"开始训练: {args.num_epochs}个epoch, 共{total_steps}步")

    # 在开始训练前检查一个批次数据
    print("检查数据格式...")
    first_batch = next(train_dataset.create_dict_iterator())
    examine_batch(first_batch)

    for epoch in range(args.num_epochs):
        # 训练一个epoch
        model.set_train(True)
        train_loss = 0
        train_steps = 0
        
        # 重置第一个批次标志（每个epoch只在第一个批次详细输出）
        forward_fn.first_batch = epoch == 0
        
        progress_bar = tqdm(train_dataset.create_dict_iterator(), total=train_dataset.get_dataset_size())
        for batch in progress_bar:
            loss = train_step(model, optimizer, batch)
            
            # 错误处理
            if isinstance(loss, mindspore.Tensor):
                try:
                    loss_value = loss.asnumpy()
                    train_loss += loss_value
                except:
                    print(f"警告: 无法转换损失值 {loss}")
                    train_loss += 1000.0  # 使用一个大的默认值
            else:
                # 处理SequenceClassifierOutputWithPast类型的返回值
                try:
                    if hasattr(loss, "loss"):
                        loss_value = loss.loss.asnumpy()
                        train_loss += loss_value
                    else:
                        print(f"警告: 返回对象没有loss属性 {type(loss)}")
                        train_loss += 1000.0  # 使用一个大的默认值
                except Exception as e:
                    print(f"警告: 处理损失值时出错 {e}")
                    train_loss += 1000.0  # 使用一个大的默认值
                
            train_steps += 1
            
            # 更新进度条
            progress_bar.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
            progress_bar.set_postfix(loss=train_loss/train_steps)
            
            # 在debug模式下,训练几个批次后停止
            if args.debug and train_steps >= 5:
                print("Debug模式: 提前停止训练")
                break
        
        # 计算该epoch的平均损失
        avg_loss = train_loss / train_steps
        print(f"Epoch {epoch+1}/{args.num_epochs}, 平均损失: {avg_loss:.4f}")
        
        # 保存模型检查点
        try:
            # 使用模型自带的save_pretrained方法替代mindspore.save_checkpoint
            save_dir = os.path.join(args.save_dir, f"epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            print(f"模型已保存至 {save_dir}")
        except Exception as e:
            print(f"保存模型时出错: {e}")
        
        # 评估
        acc = evaluate(model, valid_dataset, metric)
        print(f"验证集准确率: {acc:.4f}")
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            try:
                # 使用模型自带的save_pretrained方法替代mindspore.save_checkpoint
                best_save_dir = os.path.join(args.save_dir, "best_model")
                os.makedirs(best_save_dir, exist_ok=True)
                model.save_pretrained(best_save_dir)
                print(f"找到更好的模型! 已保存至 {best_save_dir}")
            except Exception as e:
                print(f"保存最佳模型时出错: {e}")
        
        # 在debug模式下,只运行一个epoch
        if args.debug:
            print("Debug模式: 提前停止训练")
            break

    # 在测试集上评估
    print("在测试集上评估...")
    test_acc = evaluate(model, test_dataset, metric)
    print(f"测试集准确率: {test_acc:.4f}")

    # 保存最终模型和tokenizer
    try:
        final_save_dir = os.path.join(args.save_dir, "final_model")
        os.makedirs(final_save_dir, exist_ok=True)
        
        # 保存模型
        model.save_pretrained(final_save_dir)
        
        # 保存tokenizer
        tokenizer.save_pretrained(final_save_dir)
        
        print(f"最终模型和tokenizer已保存至 {final_save_dir}")
        
        # 创建README文件，记录训练信息
        readme_path = os.path.join(final_save_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"# MRPC数据集上的GPT2模型\n\n")
            f.write(f"## 训练信息\n")
            f.write(f"- 模型: {args.model_name}\n")
            f.write(f"- 数据集: MRPC\n")
            f.write(f"- 训练轮数: {args.num_epochs}\n")
            f.write(f"- 批次大小: {args.batch_size}\n")
            f.write(f"- 学习率: {args.lr}\n")
            f.write(f"- 验证集准确率: {best_acc:.4f}\n")
            f.write(f"- 测试集准确率: {test_acc:.4f}\n")
            
        print("训练信息已记录到README.md")
    except Exception as e:
        print(f"保存最终模型时出错: {e}")

if __name__ == "__main__":
    main() 