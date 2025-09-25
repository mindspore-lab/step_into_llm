# pylint: disable=missing-module-docstring
from mindnlp.transformers import pipeline

# Text Generation Demo
# generator = pipeline(model="openai-community/gpt2")
generator = pipeline(
    model="openai-community/gpt2", 
    model_kwargs={
        'use_safetensors': False  
    }
)
outputs = generator("I can't believe you did such a ", do_sample=False)
print(outputs)

# [{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

# Chat Demo
chat1 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a test"},
            {"role": "assistant", "content": "This is a reply"},
        ]
chat2 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a second test"},
            {"role": "assistant", "content": "This is a reply"},
        ]

# 进行对话时，tokenizer默认没有设置`chat_template`， 需要自行设置
from mindnlp.transformers import  AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

generator = pipeline("text-generation", model="openai-community/gpt2", tokenizer=tokenizer,
                    model_kwargs={
                    'use_safetensors': False  
                })
                
outputs = generator(chat1, do_sample=False, max_new_tokens=10)
print(outputs)

outputs = generator([chat1, chat2], do_sample=False, max_new_tokens=10)
print(outputs)
