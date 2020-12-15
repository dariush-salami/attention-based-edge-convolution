import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, BatchNorm
from custom_graph_convolution import CGCNConv


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super(Net, self).__init__()
        self.spatial_conv_1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.spatial_conv_2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.spatial_lin = MLP([128 + 64, 1024])

        self.temporal_conv_1 = CGCNConv(2 * 2, 16)
        self.temporal_conv_2 = CGCNConv(16 * 2, 64)
        self.temporal_lin = MLP([64, 1024])

        self.batch_norm = BatchNorm(2048)

        self.mlp = Seq(
            MLP([2048, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        pos, batch, edge_index, edge_attr = data.pos.float(), data.batch, data.edge_index, data.edge_attr.float()
        # Spatial Feature Extraction
        x1 = self.spatial_conv_1(pos, batch)
        x2 = self.spatial_conv_2(x1, batch)
        spatial_latent_features = self.spatial_lin(torch.cat([x1, x2], dim=1))
        spatial_latent_features = global_max_pool(spatial_latent_features, batch)

        # Temporal Feature Extraction
        _, edge_attr = self.temporal_conv_1(pos, edge_index, edge_attr)
        _, edge_attr = self.temporal_conv_2(pos, edge_index, edge_attr)
        temporal_latent_features = self.temporal_lin(edge_attr)
        temporal_latent_features = global_max_pool(temporal_latent_features, batch[edge_index[0]])

        # Concatenating two set of features
        combined_latent_features = torch.cat([spatial_latent_features, temporal_latent_features], dim=1)
        out = self.batch_norm(combined_latent_features)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)
