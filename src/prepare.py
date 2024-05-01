import logging
import torch
import warnings

logger = logging.getLogger("torch.distributed.nn.jit.instantiator")
logger.setLevel(logging.ERROR)

def get_local_debug():
    import socket
    hostname = socket.gethostname()
    if hostname == 'ps018':
        local_debug = True
    else:
        local_debug = False
    return local_debug

warnings.filterwarnings(
    "ignore", "The PyTorch API of nested tensors is in prototype stage*"
)

warnings.filterwarnings("ignore", "Converting mask without torch.bool dtype to bool*")

# torch.set_float32_matmul_precision("high")
