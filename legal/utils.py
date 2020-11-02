import random
import numpy as np
import torch
seed=0

random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True
# # Remove randomness (may be slower on Tesla GPUs)
# # https://pytorch.org/docs/stable/notes/randomness.html
# if seed == 0:
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False