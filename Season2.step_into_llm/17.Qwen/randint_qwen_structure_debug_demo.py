import mindspore
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir+"/mindnlpv041") #克隆mindnlp到本地重命名然后设置环境变量开始源码调试，用以观察Qwen极简结构 export PYTHONPATH=$PYTHONPATH:/home/tridu33/workspace/qwen2/mindnlpv041

from mindnlpv041.mindnlp.transformers.models.qwen2 import Qwen2Config, Qwen2Model
# 设置单线程
from mindspore._c_expression import disable_multi_thread
disable_multi_thread()
# 设置同步
import mindspore as ms
mindspore.runtime.launch_blocking()


def run_qwen2():
    # https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/config.json
    qwen2config = Qwen2Config(vocab_size = 151936,
                              hidden_size = 2048,
                              intermediate_size = 11008,
                              num_hidden_layers = 2, # 原始配置是36，改小一点加速调试过程
                              num_attention_heads = 16, # 16个注意力头
                              #每一头的 hidden_dim=hidden_size/num_attention_heads 嵌入维度必须能够被头数整除的原因是为了确保每个头获得相等长度的输入，从而进行独立的注意力计算。嵌入维度被头数整除还有一个重要的原因是减少计算量。
                              num_key_value_heads = 2,
                              max_position_embeddings = 32768, # 模型可以处理的最大序列长度。
                              )
    qwen2model = Qwen2Model(config=qwen2config)
    print(qwen2config)
    input_ids = mindspore.ops.randint(0, qwen2config.vocab_size, (2,16)) # 大小任意:batch为2，序列长度qlen为16
    # 最初的文本经过tokenizer生成input_ids,再经过编码得到hidden_states
    res = qwen2model(input_ids)
    print(res) # last_hidden_state=Tensor(shape=[2, 16, 2048]..., dtype=Float32), past_key_values=(), attentions=(), hidden_states=())


if __name__ == "__main__":
    run_qwen2()
