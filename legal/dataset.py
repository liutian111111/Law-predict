from torch.utils.data import Dataset
import config
import numpy as np


class LegalDataset(Dataset):
    def __init__(self, train=True):
        if train:
            base_dir = config.train_file_base
        else:
            base_dir = config.test_file_base
        self.single_fact = np.load(base_dir + 'singlefact.npy')
        self.fact = np.load(base_dir + 'fact.npy')
        self.term = np.load(base_dir + 'term.npy')
        self.law = np.load(base_dir + 'law.npy')
        self.accu = np.load(base_dir + 'accu.npy')
        print(self.single_fact.shape)

    def __getitem__(self, index):
        return self.single_fact[index], self.term[index], self.law[index], self.accu[index]

    def __len__(self):
        return self.fact.shape[0]


