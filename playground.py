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


example_input = torch.tensor([range(0, 20)], dtype=torch.float64).reshape(10, 2).float()
index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
print(example_input.scatter_(0, index, example_input))
example_input = example_input.reshape(2, 2, 5)

multihead_attn = MultiHeadAttention(5, 1)
print(example_input.size())

attn_output = multihead_attn(example_input, example_input, example_input)

print(example_input)
print(attn_output)
