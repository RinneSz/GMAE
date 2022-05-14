from collator import collator
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from functools import partial
import random
import torch
from wrapper import MyDataset
from torch_geometric.utils import to_undirected

from torch_geometric.datasets import Planetoid, WikiCS, Amazon
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T


dataset = None


def get_dataset(dataset_name='Cora'):
    global dataset
    path = 'dataset/' + dataset_name
    if dataset is not None:
        return dataset

    elif dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        return Planetoid(root=path, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())
    elif dataset_name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    elif dataset_name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    else:
        raise NotImplementedError


class GraphDataModule(LightningDataModule):
    name = "Cora"

    def __init__(
        self,
        dataset_name: str = 'Cora',
        num_workers: int = 8,
        batch_size: int = 64,
        seed: int = 42,
        l1: int = 2,
        l2: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(dataset_name)
        self.seed = seed
        self.n_val_sampler = 10
        self.l1 = l1
        self.l2 = l2

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...

    def setup(self, stage: str = None):
        num_nodes = self.dataset.data.num_nodes
        node_idx = list(range(num_nodes))
        random.shuffle(node_idx)
        self.shuffled_index = torch.LongTensor(node_idx)

    def process_samples(self, batch_size, n_id, adj):
        edge_index = adj[0].edge_index
        if edge_index.size(1) != 0:
            edge_index = to_undirected(edge_index)
        n_nodes = len(n_id)
        edge_sp_adj = torch.sparse.FloatTensor(edge_index,
                                               torch.ones(edge_index.shape[1]),
                                               [n_nodes, n_nodes])
        edge_adj = edge_sp_adj
        return [self.dataset.data.x[n_id], self.dataset.data.y[n_id[0]], edge_adj]

    def train_dataloader(self):
        sampler = NeighborSampler(self.dataset.data.edge_index, sizes=[self.l1, self.l2], batch_size=1,
                                  shuffle=False, node_idx=self.shuffled_index)
        items = []
        for s in sampler:
            items.append(self.process_samples(s[0], s[1], s[2]))
        self.dataset_train = MyDataset(items)
        loader = DataLoader(self.dataset_train, batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            collate_fn=partial(collator),
                            )
        return loader

    def val_dataloader(self):
        """for downstream evaluation"""
        samplers = []
        for i in range(self.n_val_sampler):
            samplers.append(NeighborSampler(self.dataset.data.edge_index, sizes=[self.l1, self.l2], batch_size=1,
                                  shuffle=False, node_idx=self.shuffled_index))
        items = []
        for samples in zip(*samplers):
            for s in samples:
                items.append(self.process_samples(s[0], s[1], s[2]))
        self.dataset_val = MyDataset(items)
        loader = DataLoader(self.dataset_val, batch_size=self.batch_size*self.n_val_sampler,
                            shuffle=False,
                            num_workers=self.num_workers,
                            collate_fn=partial(collator),
                            )
        return loader
