import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import global_max_pool, BatchNorm
from custom_graph_convolution import CGCNConv
from temporal_edgecnn.temporal_edgecnn import TemporalSelfAttentionDynamicEdgeConv, TemporalDynamicEdgeConv, \
    AutomatedGraphDynamicEdgeConv, GeneralizedTemporalSelfAttentionDynamicEdgeConv, \
    GeneralizedTemporalSelfAttentionDynamicEdgeConvWithoutMask
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, out_channels, graph_convolution_layers=2, T=1, k=4, spatio_temporal_factor=0.01, aggr='max'):
        super().__init__()
        self.graph_convolution_layers = graph_convolution_layers

        self.conv1 = TemporalDynamicEdgeConv(nn=MLP([2 * 3, 64, 64, 64]),
                                             k=k)

        self.lin1 = MLP([64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        sequence_numbers, pos, batch = data.x[:, 0].float(), data.pos.float(), data.batch
        x1 = self.conv1(pos, sequence_numbers, batch)
        out = self.lin1(x1)
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)
