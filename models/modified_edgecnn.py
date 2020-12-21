import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import global_max_pool, BatchNorm
from custom_graph_convolution import CGCNConv
from temporal_edgecnn.temporal_edgecnn import TemporalAttentionDynamicEdgeConv


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=10, aggr='max'):
        super().__init__()

        self.conv1 = TemporalAttentionDynamicEdgeConv(MLP([2 * 3, 64, 64, 64]),
                                                      MLP([64, 1]), k, aggr)
        self.conv2 = TemporalAttentionDynamicEdgeConv(MLP([2 * 64, 128]),
                                                      MLP([128, 1]), k, aggr)
        self.conv3 = TemporalAttentionDynamicEdgeConv(MLP([2 * 128, 256]),
                                                      MLP([256, 1]), k, aggr)
        self.lin1 = MLP([256 + 128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        sequence_numbers, pos, batch = data.x[:, 0].float(), data.pos.float(), data.batch
        x1 = self.conv1(pos, sequence_numbers, batch)
        x2 = self.conv2(x1, sequence_numbers, batch)
        x3 = self.conv3(x2, sequence_numbers, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)