import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter

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


A = torch.from_numpy(np.array([[0, 1.5, 2, 20], [-1.5, 0, 1.5, 0]]))
A -= A.min(1, keepdim=True)[0]
A /= A.max(1, keepdim=True)[0]

print(A)