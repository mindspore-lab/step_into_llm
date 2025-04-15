import sys
import os
# Add current directory to Python path for local module imports
sys.path.insert(0, os.path.abspath("."))  
from ms_mindnlp.transformers.models.llama.modeling_llama import LlamaModel
from ms_mindnlp.transformers.models.llama.configuration_llama import LlamaConfig
import mindspore as ms
from mindspore import dtype, ops
import debugpy

debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach...")

debugpy.wait_for_client()
print("Debugger is attached.")

# import inspect
# llama_config_file_path = inspect.getfile(LlamaConfig)
# print(f"{llama_config_file_path}")

ms.set_context(mode=ms.PYNATIVE_MODE)

def run():
    """Main execution function for LLaMA model inference demo"""
    config = LlamaConfig(
        vocab_size=32000,  # Tokenizer vocabulary size
        hidden_size=4096,  # Hidden layer dimension
        intermediate_size=11008,  # FFN layer inner dimension
        num_hidden_layers=2,  # Number of transformer blocks
        num_attention_heads=32,  # Parallel attention heads
        num_key_value_heads=2,  # KV heads for grouped-query attention
        max_position_embeddings=2048,  # Maximum sequence length
    )
    model = LlamaModel(config=config)
    # Generate random input tensor: (batch_size=2, seq_length=16)
    input_ids = ops.randint(0, config.vocab_size, (2, 16), dtype=dtype.int32)
    output = model(input_ids=input_ids)
    print("inference")
    print(output)

if __name__ == "__main__":
    run()