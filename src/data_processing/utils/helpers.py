import torch

def get_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        return round(gpu_memory_bytes / (2**30))
    return 0

def recommend_model_config(gpu_memory_gb):
    if gpu_memory_gb < 5.1:
        return None, None
    elif gpu_memory_gb < 8.1:
        return "google/gemma-2b-it", True
    elif gpu_memory_gb < 19.0:
        return "google/gemma-2b-it", False
    else:
        return "google/gemma-7b-it", False