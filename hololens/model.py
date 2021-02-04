from torch.nn import Dropout
from torch_geometric.nn import global_max_pool
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import math

from hololens.multi_head_attention import MultiHeadAttention

try:
    from torch_cluster import knn
except ImportError:
    knn = None


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.tensor([1., 0., 0., 0., 1., 0., 0., 0., 1.], requires_grad=True).to(x.device) \
            .view(1, 9).repeat(
            batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


def make_proper_data(data: Tensor, sequence_number: Tensor, batch: Tensor, self_loop: bool = False, T: int = 1):
    source, source_batch, target, target_batch = data, batch, data.clone(), None
    index_mapper = torch.arange(0, len(data), device=data.device)
    batch_size = len(torch.unique(batch))
    frame_number = len(torch.unique(sequence_number))
    point_number = len(data) // (batch_size * frame_number)
    source_batch = (batch * frame_number + sequence_number - 1).long()
    target_batch = source_batch.clone()
    target = target.reshape(batch_size, 1, frame_number, -1, data.shape[-1])
    index_mapper = index_mapper.reshape(batch_size, 1, frame_number, -1, 1)
    target = target.repeat(1, frame_number, 1, 1, 1).reshape(batch_size, frame_number * frame_number, -1,
                                                             data.shape[-1])
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


class GeneralizedTemporalSelfAttentionDynamicEdgeConv(MessagePassing):
    def __init__(self, nn: Callable, T: int, attention_in_features: int, head_num: int, k: int,
                 aggr: str = 'max',
                 num_workers: int = 1, spatio_temporal_factor: float = 0, **kwargs):
        super(GeneralizedTemporalSelfAttentionDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`GeneralizedTemporalSelfAttentionDynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.multihead_attn = MultiHeadAttention(attention_in_features, head_num)
        self.k = k
        self.num_workers = num_workers
        self.spatio_temporal_factor = spatio_temporal_factor
        self.T = T
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.multihead_attn)
        reset(self.nn)

    def forward(
            self, x: Tensor,
            sequence_number: Tensor,
            batch: Tensor, ) -> Tensor:
        assert x is not None
        assert sequence_number is not None
        assert batch is not None
        knn_input = torch.cat((x, sequence_number.reshape(-1, 1)), 1)
        knn_input -= knn_input.min(0, keepdim=True)[0]
        knn_input /= knn_input.max(0, keepdim=True)[0]
        knn_input[:, -1] *= self.spatio_temporal_factor * math.sqrt(x.shape[-1])
        source_data, source_batch, target_data, target_batch, index_mapper = make_proper_data(data=knn_input,
                                                                                              sequence_number=sequence_number,
                                                                                              batch=batch,
                                                                                              T=self.T)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `GeneralizedTemporalSelfAttentionDynamicEdgeConv`.'

        edge_index = knn(target_data, source_data, self.k, target_batch, source_batch,
                         num_workers=self.num_workers)
        edge_index[1] = index_mapper[edge_index[1]]
        # propagate_type: (x: PairTensor, batch: Tensor)
        return self.propagate(edge_index, x=x, size=None, batch=batch)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  batch: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        original_shape = inputs.shape
        # We assume K is fixed and the index tensor is sorted!
        attention_input_shape = list([int(original_shape[0] / self.k)]) + list(original_shape)
        attention_input_shape[1] = self.k
        self_attention_input = inputs.reshape(attention_input_shape)
        attn_output = self.multihead_attn(self_attention_input, self_attention_input, self_attention_input)
        attn_output = attn_output.reshape(original_shape)
        # Apply attention mechanism
        return scatter(attn_output, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)


class Net(torch.nn.Module):
    def __init__(self, out_channels, graph_convolution_layers=2, T=1, k=4, spatio_temporal_factor=0.01, aggr='max'):
        super().__init__()
        self.stn = STN3d()
        self.graph_convolution_layers = graph_convolution_layers
        self.conv1 = GeneralizedTemporalSelfAttentionDynamicEdgeConv(nn=MLP([2 * 3, 64, 64, 64]),
                                                                     attention_in_features=64,
                                                                     head_num=8,
                                                                     k=k,
                                                                     spatio_temporal_factor=spatio_temporal_factor,
                                                                     T=T).jittable()
        self.conv2 = GeneralizedTemporalSelfAttentionDynamicEdgeConv(nn=MLP([2 * 64, 128]),
                                                                     attention_in_features=128,
                                                                     head_num=8,
                                                                     k=k,
                                                                     spatio_temporal_factor=spatio_temporal_factor,
                                                                     aggr=aggr,
                                                                     T=T).jittable()
        self.conv3 = GeneralizedTemporalSelfAttentionDynamicEdgeConv(nn=MLP([2 * 128, 256]),
                                                                     attention_in_features=256,
                                                                     head_num=8,
                                                                     k=k,
                                                                     spatio_temporal_factor=spatio_temporal_factor,
                                                                     aggr=aggr,
                                                                     T=T).jittable()
        assert 1 <= graph_convolution_layers <= 3
        if graph_convolution_layers == 3:
            self.lin1 = MLP([256 + 128 + 64, 1024])
        elif graph_convolution_layers == 2:
            self.lin1 = MLP([128 + 64, 1024])
        elif graph_convolution_layers == 1:
            self.lin1 = MLP([64, 1024])
        self.lin1 = MLP([64, 1024])
        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, sequence_numbers, pos, batch):
        # sequence_numbers, pos, batch = data.x[:, 0].float(), data.pos.float(), data.batch
        pos = pos.reshape(len(torch.unique(batch)), -1, 3).transpose(2, 1)
        trans = self.stn(pos)
        pos = pos.transpose(2, 1)
        pos = torch.bmm(pos, trans)
        pos = pos.reshape(-1, 3)
        if self.graph_convolution_layers == 3:
            x1 = self.conv1(pos, sequence_numbers, batch)
            x2 = self.conv2(x1, sequence_numbers, batch)
            x3 = self.conv3(x2, sequence_numbers, batch)
            out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        elif self.graph_convolution_layers == 2:
            x1 = self.conv1(pos, sequence_numbers, batch)
            x2 = self.conv2(x1, sequence_numbers, batch)
            out = self.lin1(torch.cat([x1, x2], dim=1))
        else:
            x1 = self.conv1(pos, sequence_numbers, batch)
            out = self.lin1(x1)
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)
