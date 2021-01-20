import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
from torch.nn import BatchNorm1d as BN
from torch_cluster import knn


def make_proper_data(data, sequence_number, batch, self_loop=False, T=1):
    source, source_batch, target, target_batch = data, batch, data.clone(), None
    index_mapper = torch.arange(0, len(data), device=data.device)
    batch_size = len(torch.unique(batch))
    frame_number = len(torch.unique(sequence_number))
    point_number = len(data) // (batch_size * frame_number)
    source_batch = (batch * frame_number + sequence_number - 1).long()
    target_batch = source_batch.clone()
    target = target.reshape(batch_size, 1, frame_number, -1, data.shape[-1])
    index_mapper = index_mapper.reshape(batch_size, 1, frame_number, -1, 1)
    target = target.repeat(1, frame_number, 1, 1, 1).reshape(batch_size, frame_number * frame_number, -1, data.shape[-1])
    index_mapper = index_mapper.repeat(1, frame_number, 1, 1, 1).reshape(batch_size, frame_number * frame_number, -1, 1)
    if self_loop:
        mask = torch.tril(torch.ones((frame_number, frame_number), device=data.device))
    else:
        mask = torch.tril(torch.ones((frame_number, frame_number), device=data.device), diagonal=-1)
        mask[0][0] = 1
    mask -= torch.tril(torch.ones((frame_number, frame_number), device=data.device), diagonal=-T - 1)
    mask = mask.reshape(-1)
    target = target[:, mask == 1]
    index_mapper = index_mapper[:, mask == 1]
    target_batch = target_batch.reshape(-1, 1).repeat(1, frame_number).reshape(batch_size, -1, point_number)
    target_batch = target_batch[:, mask == 1]
    return source, source_batch, target.reshape(-1, data.shape[-1]), target_batch.reshape(-1), index_mapper.reshape(-1)


num_points = 3
num_frames = 3
batch_size = 3
x = torch.rand((batch_size * num_frames * num_points, 3))
sequence_number = torch.arange(1, num_frames + 1).reshape(1, -1, 1).repeat(batch_size, 1, num_points).reshape(-1)
batch = torch.arange(0, batch_size).reshape(-1, 1).repeat(1, num_points * num_frames).reshape(-1)
source_data, source_batch, target_data, target_batch, index_mapper = make_proper_data(x, sequence_number, batch, False, 100)

edge_index = knn(target_data, source_data, 2, target_batch, source_batch)
edge_index[1] = index_mapper[edge_index[1]]
print(x)
print(sequence_number)
print(batch)
print(edge_index)
print(batch[edge_index[1]] - batch[edge_index[0]])
print(sequence_number[edge_index[1]] - sequence_number[edge_index[0]])
