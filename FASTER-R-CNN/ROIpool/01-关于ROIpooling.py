import torch
from torchvision.ops import RoIPool

a = torch.randn(1, 3, 20, 20)
pool = RoIPool((7, 7), 2)
pool()