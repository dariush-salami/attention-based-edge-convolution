import torch
import torch.nn.functional as F

loss = F.nll_loss(torch.tensor([[
    0.5, 0.25, 0.25
]]), torch.tensor([0]))
print(loss)
