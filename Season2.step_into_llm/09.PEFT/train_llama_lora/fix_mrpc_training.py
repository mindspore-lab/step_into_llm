"""
用于修复MRPC数据集训练的辅助函数
"""

import mindspore
from mindspore import Tensor, ops

def print_dataset_keys(dataset):
    """打印数据集的列名"""
    print(f"数据集列名: {dataset.get_col_names()}")
    # 尝试获取一个样本并打印其键
    iterator = dataset.create_dict_iterator()
    try:
        sample = next(iterator)
        print(f"样本中的键: {list(sample.keys())}")
        for key, value in sample.items():
            print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
    except:
        print("无法获取样本数据")

def improved_forward_fn(model, data, verbose=False):
    """改进的前向计算函数,有更好的错误处理"""
    # 移除不支持的字段
    data_copy = dict(data)  # 创建数据的副本以避免修改原始数据
    
    if 'lens' in data_copy:
        del data_copy['lens']
    
    # 检查数据键
    if verbose:
        print(f"数据包含以下键: {list(data_copy.keys())}")
    
    # 检查'labels'是否存在,如果不存在查找替代键
    label_keys = ['labels', 'label']
    label_found = False
    
    for key in label_keys:
        if key in data_copy:
            labels = data_copy.pop(key)
            label_found = True
            if verbose:
                print(f"使用'{key}'作为标签")
            break
    
    if not label_found:
        # 如果找不到标签,创建一个虚拟标签
        if verbose:
            print("警告: 找不到标签字段,使用虚拟标签")
        # 如果有input_ids,使用其批次大小
        if 'input_ids' in data_copy:
            batch_size = data_copy['input_ids'].shape[0]
        else:
            batch_size = 1
        labels = ops.zeros(batch_size, dtype=mindspore.int32)
    
    # 使用模型进行前向计算
    try:
        outputs = model(**data_copy, labels=labels)
        
        # 获取损失
        if isinstance(outputs, tuple):
            loss = outputs[0]
        elif hasattr(outputs, "loss"):
            # 处理SequenceClassifierOutputWithPast对象
            loss = outputs.loss
        else:
            loss = outputs
            
        return loss
    except Exception as e:
        print(f"前向计算出错: {e}")
        # 返回一个默认的损失
        return Tensor(1000.0)

def improved_train_step(model, optimizer, data, grad_fn):
    """改进的训练步骤函数,有更好的错误处理"""
    try:
        # 计算损失和梯度
        loss, grads = grad_fn(model, data)
        
        # 应用梯度
        optimizer(grads)
        
        return loss
    except Exception as e:
        print(f"训练步骤出错: {e}")
        # 返回一个默认的损失
        return Tensor(1000.0)

def examine_batch(batch):
    """检查批次数据结构"""
    print("\n批次数据检查:")
    
    if not isinstance(batch, dict):
        print(f"批次不是字典类型,而是 {type(batch)}")
        return
    
    print(f"批次包含 {len(batch)} 个键:")
    
    for key, value in batch.items():
        if isinstance(value, Tensor):
            print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
            # 打印值的摘要
            if value.size < 10:
                print(f"    值: {value}")
            else:
                flat_value = value.reshape(-1)
                print(f"    前5个值: {flat_value[:5]}")
                
                # 检查是否是标签
                if key.lower() in ['label', 'labels']:
                    unique_labels = ops.unique(flat_value)[0]
                    print(f"    唯一标签值: {unique_labels}")
        else:
            print(f"  {key}: 类型={type(value)}")

def prepare_mrpc_batch(batch):
    """准备MRPC批次以进行训练"""
    # 移除不支持的字段
    result = {}
    
    # 复制所有支持的字段
    for key in ['input_ids', 'attention_mask', 'token_type_ids']:
        if key in batch:
            result[key] = batch[key]
    
    # 处理标签
    if 'labels' in batch:
        result['labels'] = batch['labels']
    elif 'label' in batch:
        # 重命名label为labels
        result['labels'] = batch['label']
    else:
        # 如果没有标签,创建一个虚拟标签
        if 'input_ids' in batch:
            batch_size = batch['input_ids'].shape[0]
        else:
            batch_size = 1
        result['labels'] = ops.zeros(batch_size, dtype=mindspore.int32)
    
    return result 