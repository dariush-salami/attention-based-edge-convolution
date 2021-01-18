import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
from torch.nn import BatchNorm1d as BN

src = torch.arange(25, dtype=torch.float).reshape(1, 1, 5, 5).requires_grad_()  # 1 x 1 x 5 x 5 with 0 ... 25
indices = torch.tensor([[-1, -1], [.5, .5]], dtype=torch.float).reshape(1, 1, -1, 2)  # 1 x 1 x 2 x 2
output = F.grid_sample(src, indices)
print(src)
print(output)

a = torch.arange(30, dtype=torch.float64).reshape(10, 3)
indices = torch.arange(5, dtype=torch.long).reshape(-1, 1).repeat(1, 2).reshape(-1)
print(a)
print(indices)
print(scatter(a, indices, dim_size=5, dim=-2, reduce='max'))

A = torch.tensor([[0.2008, 0.0400, -0.0931, 1],
                  [0.2167, 0.0458, -0.1069, 2],
                  [0.1959, 0.0189, -0.0909, 30],
                  [-1.1217, -0.2696, 2.3543, 20],
                  [-0.0379, 0.0223, 0.1487, 14],
                  [-1.1447, -0.2898, 2.3234, 1]])

B = torch.tensor([[0.0, 1.0, 2.0, 3.0],
                  [3.0, 2.0, 1.0, 0.0],
                  [1.5, 1.5, 5, 200]])

batch_norm_layer = BN(4)
print(batch_norm_layer(B))
