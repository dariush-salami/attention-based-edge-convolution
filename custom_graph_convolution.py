import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_mean


class CGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CGCNConv, self).__init__(aggr='mean')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform edge feature matrix.
        src, dst = edge_index
        mean_incoming_edge_attr = scatter_mean(edge_attr, dst, dim=0)
        edge_attr = torch.cat([edge_attr, mean_incoming_edge_attr[src]], dim=1)
        edge_attr = self.lin(edge_attr)

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr), edge_attr

