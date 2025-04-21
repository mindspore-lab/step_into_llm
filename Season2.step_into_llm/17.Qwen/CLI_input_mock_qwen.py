import mindspore
import numpy as np
from mindspore import dtype as mstype
import mindspore.ops as ops
from mindspore import Tensor
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM
import faulthandler

faulthandler.enable()

model_id = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, mirror='modelscope')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    ms_dtype=mindspore.float16,
    mirror='modelscope'
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="ms"
)
attention_mask = Tensor(np.ones(input_ids.shape), mstype.float32)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|endoftext|>")
]
outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=20,
    eos_token_id=terminators,
    do_sample=False,
    # do_sample=True,
    # temperature=0.6,
    # top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(outputs)
print(tokenizer.decode(response, skip_special_tokens=True))

