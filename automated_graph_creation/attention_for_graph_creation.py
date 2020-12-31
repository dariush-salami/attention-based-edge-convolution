import torch
import torch.nn as nn


class SelfAttentionEdgeIndexCreatorLayer(nn.Module):
    def __init__(self, embed_size, heads, number_of_edges):
        super(SelfAttentionEdgeIndexCreatorLayer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.number_of_edges = number_of_edges
        self.head_dim = embed_size // heads
        self.head_edges = number_of_edges // heads
        assert (embed_size % heads == 0), 'Embedding size needs to be divisible by heads'
        assert (number_of_edges % heads == 0), 'The number of edges needs to be divisible by heads'

        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, keys, query):
        batch_size = query.shape[0]
        key_len, query_len = keys.shape[1], query.shape[1]

        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        query = query.reshape(batch_size, query_len, self.heads, self.head_dim)

        keys = self.keys(keys)  # (batch_size, key_len, heads, head_dim)
        queries = self.queries(query)  # (batch_size, query_len, heads, heads_dim)

        energy = torch.einsum('nqhd,nkhd->nhqk', queries, keys)

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        edges_indices = torch.sort(
            torch.topk(torch.mean(attention, dim=1), self.head_edges * self.heads, 2)[1].reshape(batch_size, key_len,
                                                                                                 -1).to(keys.device),
            2)[0]

        node_indices = torch.tensor(range(key_len)) \
            .repeat(batch_size).reshape(batch_size, key_len, 1) \
            .repeat(1, 1, edges_indices.shape[-1]) \
            .to(keys.device)

        edges_indices = edges_indices.reshape(batch_size, -1)
        node_indices = node_indices.reshape(batch_size, -1)

        edge_index = torch.stack([
            torch.cat((i[0].reshape(-1, 1), i[1].reshape(-1, 1)), dim=1).permute(1, 0)
            for i in zip(*(node_indices, edges_indices))
        ]).to(keys.device)
        return edge_index


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(5, 512, 512).to(device)
    seq_number = torch.tensor([
        [1, 1, 2],
        [1, 2, 3]
    ]).to(device)
    self_attention_layer = SelfAttentionEdgeIndexCreatorLayer(512, 4, 8)
    edge_index = self_attention_layer.forward(x, x)

    print(edge_index)
