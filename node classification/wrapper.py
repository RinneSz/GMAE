import torch

import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos


# GMAE_graph positional encoding
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        super(MyDataset, self).__init__()
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return preprocess_item(item)


def preprocess_item(item):
    x, y, adj = item[0], item[1], item[2].to_dense()
    N = x.size(0)

    # node adj matrix [N, N] bool
    adj = adj.bool()

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N, N], dtype=torch.float)

    in_degree = adj.long().sum(dim=1).view(-1)
    out_degree = adj.long().sum(dim=0).view(-1)
    return x, y, adj, attn_bias, spatial_pos, in_degree, out_degree
