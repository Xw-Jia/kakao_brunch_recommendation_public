import os

import torch

from config import mf_factors as factors
from config import mf_iterations as iterations
from config import learning_rate
from config import device
from config import device_idx
from config import models_root

from .model import Model
from .dataset import Dataset
from .trainer import Trainer


class MF:
    def __init__(self, preprocessed, dataset):
        self.preprocessed = preprocessed
        self.dataset = dataset
        self.model_save_path = models_root.joinpath('.'.join(map(str, [
            'mf', 'torch', 'model', factors, iterations, learning_rate, 'pkl'
        ])))
        self.model_log_path = models_root.joinpath('.'.join(map(str, [
            'mf', 'torch', 'model', factors, iterations, learning_rate, 'log', 'pkl'
        ])))
        self.best_model_path = models_root.joinpath('.'.join(map(str, [
            'mf', 'torch', 'model', factors, iterations, learning_rate, 'best', 'pkl'
        ])))
        self.model = Model().to(device)

    def load(self):
        if not self.model_save_path.is_file():
            raise FileNotFoundError('Run train.py to build model')
        state_dict = torch.load(self.model_save_path)
        self.model.load_state_dict(state_dict)

    def load_best_model(self):
        if not self.best_model_path.is_file():
            raise FileNotFoundError('Run train.py to build model')
        state_dict = torch.load(self.model_save_path)
        self.model.load_state_dict(state_dict)

    def train(self):
        train_dataset = Dataset(self.preprocessed)
        trainer = Trainer(model=self.model,
                          train_dataset=train_dataset,
                          checkpoint_path=self.model_save_path,
                          best_model_path=self.best_model_path,
                          log_path=self.model_log_path)
        trainer.train()

    def train_or_load_if_exists(self):
        if self.model_save_path.is_file():
            print('Loading instead of training because model file is found')
            self.load()
        else:
            print('Training because model file is not found')
            self.train()

    def recommend_topn(self, reader_num, topn, include_cond=None):
        user = self.model.user_embedding(torch.LongTensor([reader_num]).to(device))
        items = self.model.item_embedding.weight
        scores = user.matmul(items.transpose(1, 0)).squeeze()
        args = scores.argsort(descending=True).tolist()
        ret = []
        for a in args:
            if len(ret) == topn:
                break
            if include_cond is not None and not include_cond(a):
                continue
            ret.append(a)
        return ret

    def mf_scores(self, reader_num, item_nums):
        user = self.model.user_embedding(torch.LongTensor([reader_num]).to(device))
        items = self.model.item_embedding(torch.LongTensor(item_nums).to(device))
        scores = user.matmul(items.transpose(1, 0)).squeeze()
        return scores.tolist()
