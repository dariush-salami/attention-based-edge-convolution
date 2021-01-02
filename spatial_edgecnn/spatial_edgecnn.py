from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj
import numpy as np
import torch
from torch import Tensor
from torch_multi_head_attention import MultiHeadAttention
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax

from automated_graph_creation.attention_for_graph_creation import SelfAttentionEdgeIndexCreatorLayer

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


class AutomatedGraphDynamicEdgeConv(MessagePassing):
    def __init__(self, nn_before_graph_creation: Union[Callable, None], nn: Callable, graph_creation_in_features: int,
                 in_features: int, head_num: int,
                 k: int, aggr: str = 'max', **kwargs):
        super(AutomatedGraphDynamicEdgeConv,
              self).__init__(aggr=aggr, flow='target_to_source', **kwargs)

        if knn is None:
            raise ImportError('`AutomatedGraphDynamicEdgeConv` requires `torch-cluster`.')
        self.k = k
        self.graph_creator = SelfAttentionEdgeIndexCreatorLayer(graph_creation_in_features, head_num, k)
        self.nn_before_graph_creation = nn_before_graph_creation
        self.nn = nn
        self.multihead_attn = MultiHeadAttention(in_features, head_num)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.multihead_attn)
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None, ) -> Tensor:
        batch_size = len(np.unique(batch.cpu().numpy()))
        num_point = len(x) // batch_size
        if self.nn_before_graph_creation:
            x = self.nn_before_graph_creation(x)
        graph_creator_input = x.reshape(batch_size, -1, x.shape[-1])
        edge_index = self.graph_creator(graph_creator_input, graph_creator_input)
        point_index_corrector = torch.tensor([i * num_point for i in range(batch_size)]).to(x.device)
        point_index_corrector = point_index_corrector\
            .reshape(-1, 1).repeat(1, num_point * self.k)\
            .reshape(batch_size, num_point * self.k, 1).repeat(1, 1, 2)\
            .permute(0, 2, 1)
        edge_index = (edge_index + point_index_corrector).permute(1, 0, 2).reshape(2, -1)

        # propagate_type: (x: PairTensor)
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
