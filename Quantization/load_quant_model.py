from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

# Load the model with the quantization config
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", 
    quantization_config=quantization_config, 
    torch_dtype='auto',
    low_cpu_mem_usage=True
)

# Get the memory footprint (usually in MB or bytes)
memory_footprint_bytes = model_8bit.get_memory_footprint()

# Convert memory footprint to GB
memory_footprint_gb = memory_footprint_bytes / (1024 ** 3)

# Print memory footprint in GB
print(f"Memory Footprint: {memory_footprint_gb:.2f} GB")
