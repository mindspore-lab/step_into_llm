import mindspore
from mindnlp.transformers import RwkvForCausalLM, AutoTokenizer

expected_output = "Hello my name is Jasmine and I am a newbie to the"

model_id = "RWKV/rwkv-4-169m-pile"
tokenizer = AutoTokenizer.from_pretrained(model_id)
input_ids = tokenizer("Hello my name is", return_tensors="ms").input_ids
model = RwkvForCausalLM.from_pretrained(model_id, ms_dtype=mindspore.float16)

output = model.generate(input_ids, max_new_tokens=10)
output_sentence = tokenizer.decode(output[0].tolist())
print(output_sentence, expected_output)
