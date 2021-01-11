import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class TemporalSelfAttentionEdgeIndexCreatorLayer(nn.Module):
    def __init__(self, embed_size, heads, number_of_edges, num_points, device):
        super(TemporalSelfAttentionEdgeIndexCreatorLayer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.number_of_edges = number_of_edges
        self.head_dim = embed_size // heads
        self.head_edges = number_of_edges // heads
        assert (embed_size % heads == 0), 'Embedding size needs to be divisible by heads'
        assert (number_of_edges % heads == 0), 'The number of edges needs to be divisible by heads'

        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)
        self.node_indices_template = torch.tensor(range(num_points), dtype=torch.float64, requires_grad=True).to(device)

    def forward(self, keys, query, mask=None):
        batch_size = query.shape[0]
        key_len, query_len = keys.shape[1], query.shape[1]

        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        query = query.reshape(batch_size, query_len, self.heads, self.head_dim)

        keys = self.keys(keys)  # (batch_size, key_len, heads, head_dim)
        queries = self.queries(query)  # (batch_size, query_len, heads, heads_dim)

        energy = torch.einsum('nqhd,nkhd->nhqk', queries, keys)

        normalized_energy = energy / (self.embed_size ** (1 / 2))
        normalized_energy = torch.mean(normalized_energy, dim=1)
        if mask is not None:
            normalized_energy = normalized_energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(normalized_energy, dim=2)
        attention = torch.topk(attention, self.head_edges * self.heads, 2)\
            .indices\
            .type(torch.float64)\
            .requires_grad_(True)
        edges_indices = torch.sort(attention, 2).values
        node_indices = self.node_indices_template \
            .repeat(batch_size).reshape(batch_size, key_len, 1) \
            .repeat(1, 1, edges_indices.shape[-1])

        edges_indices = edges_indices.reshape(batch_size, -1)
        node_indices = node_indices.reshape(batch_size, -1)

        edge_index = torch.stack([
            torch.cat((i[0].reshape(-1, 1), i[1].reshape(-1, 1)), dim=1).permute(1, 0)
            for i in zip(*(node_indices, edges_indices))
        ])
        return edge_index


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frame_num = 8
    point_num = 64
    t = 3
    key = torch.rand(5, 512, 512).to(device)
    query = torch.rand(5, 512, 512).to(device)
    mask = torch.zeros(512, 512).to(device)
    for frame in range(frame_num):
        start_row_index = (frame - t // 2) * point_num
        if start_row_index < 0:
            start_row_index = 0
        end_row_index = (frame + t // 2) * point_num
        start_col_index = frame * point_num
        end_col_index = (frame + 1) * point_num
        mask[start_row_index:end_row_index, start_col_index:end_col_index] = 1

    # plt.subplot(211)
    # plt.imshow(mask.numpy(), cmap='Greys', interpolation='nearest')
    #
    # plt.show()
    self_attention_layer = TemporalSelfAttentionEdgeIndexCreatorLayer(512, 4, 8, num_points=512, device=device)
    edge_index = self_attention_layer.forward(key, query, mask)
    edge_index.backward()
    print(edge_index.shape)
