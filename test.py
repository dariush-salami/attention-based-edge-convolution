import torch
from torch_geometric.data import Data
from torch_scatter import scatter_mean

edge_index = torch.tensor([[1, 1, 2, 3, 3],
                           [2, 3, 4, 0, 4]], dtype=torch.long)

edge_attr = torch.tensor([[1, 2], [5, 6],
                          [3, 4], [0, 1],
                          [3, 2]], dtype=torch.float)

x = torch.tensor([[0], [1], [2], [3], [4]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

src, dst = edge_index
mean_incoming_edge_attr = scatter_mean(edge_attr, dst, dim=0)

print(mean_incoming_edge_attr)

print( mean_incoming_edge_attr[src])

print(torch.cat([edge_attr, mean_incoming_edge_attr[src]], dim=1))

