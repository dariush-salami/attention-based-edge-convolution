import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import gzip
import os.path as osp


class MMNISTDataset(InMemoryDataset):
    urls = {
        'train': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'test': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    }

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    @property
    def processed_file_names(self):
        return ['ply_data_train.pt', 'ply_data_test.pt']

    def download(self):
        download_url(self.urls['train'], osp.join(self.root, self.raw_dir))
        download_url(self.urls['test'], osp.join(self.root, self.raw_dir))

    def process(self):
        torch.save(self.process_set(self.raw_paths[0]), self.processed_paths[0])
        torch.save(self.process_set(self.raw_paths[1]), self.processed_paths[1])

    def process_set(self, dataset):
        print(dataset)
        with gzip.open(dataset, 'r') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

        data_list = []
        for digit_image in data:
            item = Data()
            item.x = torch.tensor(digit_image)
            data_list.append(item)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)
