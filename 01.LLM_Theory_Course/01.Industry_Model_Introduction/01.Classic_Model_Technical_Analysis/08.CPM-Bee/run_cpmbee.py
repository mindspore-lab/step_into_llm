from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("openbmb/cpm-bee-2b")
model = AutoModelForCausalLM.from_pretrained("openbmb/cpm-bee-2b")
result = model.generate({"input": "今天天气不错，", "<ans>": ""}, tokenizer)
print(result)
