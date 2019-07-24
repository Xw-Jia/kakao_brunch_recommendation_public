import math

import torch.nn as nn

from config import mf_factors as factors
from config import reader_count
from config import article_count


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embedding = nn.Embedding(reader_count+1, factors, sparse=True)
        self.item_embedding = nn.Embedding(article_count+1, factors, sparse=True)
        self.user_bias = nn.Embedding(reader_count+1, 1, sparse=True)
        self.item_bias = nn.Embedding(article_count+1, 1, sparse=True)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(factors)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, user, item):
        user_factor = self.user_embedding(user)
        item_factor = self.item_embedding(item)
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)

        dot = (user_factor * item_factor).sum(1)
        bias = (user_bias + item_bias).squeeze()

        scores = dot + bias
        return scores
