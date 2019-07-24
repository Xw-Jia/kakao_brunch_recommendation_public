import random

import torch.utils.data as data

from config import article_count
from utils import load_preprocessed


class Dataset(data.Dataset):
    def __init__(self, preprocessed):
        self.csr = preprocessed.interactions_sparse
        self.csr.sort_indices()
        self.coo = self.csr.tocoo()
        self.test_users = preprocessed.test_users
        self.neg_pool = list(range(1, article_count+1))
        self.neg_pool_without_heavy_items = list(set(self.neg_pool) - set(preprocessed.heavy_items))

    def __getitem__(self, index):
        row = self.coo.row[index]
        col = self.coo.col[index]
        val = self.coo.data[index]

        if row in self.test_users:
            pool = self.neg_pool_without_heavy_items
        else:
            pool = self.neg_pool
        found = False
        while not found:
            # neg_col = random.randint(1, article_count)
            neg_col = pool[int(random.random()*len(pool))]
            if self.csr[row, neg_col] == 0.0:
                found = True

        return row, col, neg_col, val

    def __len__(self):
        return self.coo.nnz
