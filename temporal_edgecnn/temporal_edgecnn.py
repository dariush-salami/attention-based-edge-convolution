from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
import numpy as np
import torch
from torch import Tensor
from torch_multi_head_attention import MultiHeadAttention
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax

try:
    from torch_cluster import knn
except ImportError:
    knn = None


def scatted_concat(source, index):
    # TODO: There should be an optimized way to do this
    unique_indices, _ = torch.sort(torch.unique(index), dim=0)
    result_shape = list([unique_indices[-1].item() + 1]) + list(source.size())
    result_shape[1] = int(result_shape[1] / result_shape[0])
    result = torch.empty(result_shape)
    for u_index in unique_indices:
        result[u_index] = source[(index == u_index).nonzero().reshape(-1)]
    del source
    torch.cuda.empty_cache()
    return result.to(source.get_device())


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


class EdgeConv(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),
    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.
    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class TemporalDynamicEdgeConv(MessagePassing):
    def __init__(self, nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super(TemporalDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`TemporalDynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            sequence_number: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        num_frames = len(np.unique(sequence_number.cpu().round().numpy()))
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `TemporalDynamicEdgeConv`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            # b = (batch, batch)
            b_list = [(batch * num_frames + sequence_number - 1).long(),
                      (batch * num_frames + sequence_number - 2).long()]
            b_list[1] = torch.where((sequence_number == 1) | (sequence_number == num_frames), b_list[0], b_list[1])
            b = (b_list[0], b_list[1])
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1],
                         num_workers=self.num_workers)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)


class TemporalAttentionDynamicEdgeConv(MessagePassing):
    def __init__(self, nn: Callable, gate_nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super(TemporalAttentionDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`TemporalAttentionDynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.gate_nn = gate_nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            sequence_number: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        num_frames = len(np.unique(sequence_number.cpu().round().numpy()))
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `TemporalAttentionDynamicEdgeConv`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            # b = (batch, batch)
            b_list = [(batch * num_frames + sequence_number - 1).long(),
                      (batch * num_frames + sequence_number - 2).long()]
            b_list[1] = torch.where((sequence_number == 1) | (sequence_number == num_frames), b_list[0], b_list[1])
            b = (b_list[0], b_list[1])
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1],
                         num_workers=self.num_workers)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None, batch=batch)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  batch: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        gate = self.gate_nn(inputs).view(-1, 1)
        gate = softmax(gate, index)
        # Apply attention mechanism
        return scatter(gate * inputs, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)


class TemporalSelfAttentionDynamicEdgeConv(MessagePassing):
    def __init__(self, nn: Callable, in_features: int, head_num: int, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super(TemporalSelfAttentionDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`TemporalSelfAttentionDynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.multihead_attn = MultiHeadAttention(in_features, head_num)
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.multihead_attn)
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            sequence_number: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        num_frames = len(np.unique(sequence_number.cpu().round().numpy()))
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        assert x[0].dim() == 2, \
            'Static graphs not supported in `TemporalSelfAttentionDynamicEdgeConv`.'

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            # b = (batch, batch)
            b_list = [(batch * num_frames + sequence_number - 1).long(),
                      (batch * num_frames + sequence_number - 2).long()]
            b_list[1] = torch.where((sequence_number == 1) | (sequence_number == num_frames), b_list[0], b_list[1])
            b = (b_list[0], b_list[1])
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        edge_index = knn(x[0], x[1], self.k, b[0], b[1],
                         num_workers=self.num_workers)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None, batch=batch)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  batch: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        self_attention_input = scatted_concat(inputs, index)
        attn_output = self.multihead_attn(self_attention_input, self_attention_input, self_attention_input)
        attn_output = attn_output.reshape(inputs.shape)
        # Apply attention mechanism
        return scatter(attn_output, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)
