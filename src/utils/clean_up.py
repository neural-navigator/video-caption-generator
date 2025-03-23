import gc
import torch

def cleanup(obj):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    del obj
    gc.collect()
    torch.cuda.empty_cache()
