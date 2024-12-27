import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Supported bfloat16: {torch.cuda.is_bf16_supported()}")