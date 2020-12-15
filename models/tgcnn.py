import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, BatchNorm
from custom_graph_convolution import CGCNConv


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, out_channels):
        super(Net, self).__init__()
        self.temporal_conv_1 = CGCNConv(2 * 2, 16)
        self.temporal_conv_2 = CGCNConv(16 * 2, 64)
        self.temporal_lin = MLP([64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        pos, batch, edge_index, edge_attr = data.pos.float(), data.batch, data.edge_index, data.edge_attr.float()

        # Temporal Feature Extraction
        _, edge_attr = self.temporal_conv_1(pos, edge_index, edge_attr)
        _, edge_attr = self.temporal_conv_2(pos, edge_index, edge_attr)
        temporal_latent_features = self.temporal_lin(edge_attr)
        temporal_latent_features = global_max_pool(temporal_latent_features, batch[edge_index[0]])

        # Concatenating two set of features
        out = self.mlp(temporal_latent_features)
        return F.log_softmax(out, dim=1)
