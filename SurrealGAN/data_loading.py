import torch
import numpy as np

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

class PTIterator(object):
    def __init__(self, PT_data, random_seed, fraction, batch_size):
        super(PTIterator, self).__init__()
        np.random.seed(random_seed)
        indices = np.random.choice(PT_data.shape[0], int(fraction*PT_data.shape[0]), replace=False)
        self.data = PT_data.astype('float32')[indices]
        self.num_samples = self.data.shape[0]
        self.batch_size = batch_size
        self.n_batches = int(self.num_samples / self.batch_size)
        if self.num_samples % self.batch_size != 0:
            self.n_batches += 1
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.data_indices = np.random.permutation(self.num_samples)
        self.batch_idx = 0

    def __next__(self):
        if self.batch_idx == self.n_batches-1:
            self.reset()
            raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]

        self.batch_idx += 1

        return {'y': torch.from_numpy(self.data[chosen_indices])}

    def __len__(self):
        return self.num_samples

class CNIterator(object):
    def __init__(self, CN_data, random_seed, fraction, batch_size):
        super(CNIterator, self).__init__()
        np.random.seed(random_seed)
        indices = np.random.choice(CN_data.shape[0], int(fraction*CN_data.shape[0]), replace=False)
        self.data = CN_data.astype('float32')[indices]
        self.num_samples = self.data.shape[0]
        self.batch_size = batch_size
        self.n_batches = int(self.num_samples / self.batch_size)
        if self.num_samples % self.batch_size != 0:
            self.n_batches += 1
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.data_indices = np.random.permutation(self.num_samples)
        self.batch_idx = 0

    def next(self):
        if self.batch_idx == self.n_batches:
            self.reset()
            #raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]

        self.batch_idx += 1

        return {'x': torch.from_numpy(self.data[chosen_indices])}

    def __len__(self):
        return self.num_samples

class val_PT_construction(object):
    def __init__(self, PT_data, **kwargs):
        super(val_PT_construction, self).__init__()
        self.data = PT_data.astype('float32')

    def load(self):
        return torch.from_numpy(self.data)

    def __len__(self):
        return self.num_samples

class val_CN_construction(object):
    def __init__(self, CN_data, **kwargs):
        super(val_CN_construction, self).__init__()
        self.data = CN_data.astype('float32')

    def load(self):
        return torch.from_numpy(self.data)

    def __len__(self):
        return self.num_samples

