import torch
from torch_multi_head_attention import MultiHeadAttention


def scatted_concat(source, index):
    unique_indices, _ = torch.sort(torch.unique(index), dim=0)
    result_shape = list([unique_indices[-1].item() + 1]) + list(source.size())
    result_shape[1] = int(result_shape[1] / result_shape[0])
    result = torch.empty(result_shape)
    for u_index in unique_indices:
        result[u_index] = source[(index == u_index).nonzero().reshape(-1)]
    return result


example_input = torch.rand(20, dtype=torch.float64).float()
indices = torch.argsort(example_input, dim=0, descending=False)
print(example_input[indices])