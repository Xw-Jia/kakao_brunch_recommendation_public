import os

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

from config import mf_iterations
from config import batch_size
from config import num_workers
from config import device
from config import device_idx
from config import num_gpu
from config import optimizer
from config import learning_rate


class Trainer:
    def __init__(self, model, train_dataset,
                 checkpoint_path, best_model_path, log_path):
        os.environ['CUDA_VISIBLE_DEVICES'] = device_idx
        self.bare_model = model.to(device)
        self.model = nn.DataParallel(model) if num_gpu > 1 else model
        self.train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
        self.optim = optimizer(self.model.parameters(), lr=learning_rate)

        self.checkpoint_path = checkpoint_path
        self.best_model_path = best_model_path
        self.log_path = log_path

        self.best_avg_loss = 987654321

    def train(self):
        self.model.train()
        for epoch in range(1, mf_iterations+1):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        tqdm_dataloader = tqdm(self.train_dataloader)
        loss_sum = 0.0
        loss_cnt = 0.0
        sig = nn.Sigmoid()
        for batch_idx, data in enumerate(tqdm_dataloader):
            users, positive_items, negative_items, val = [x.to(device).long() for x in data]
            positive_logits = self.model(users, positive_items)
            negative_logits = self.model(users, negative_items)

            self.optim.zero_grad()
            loss = (1.0 - sig(positive_logits - negative_logits)).pow(2).sum()
            loss.backward()
            self.optim.step()

            loss_sum += loss.item()
            loss_cnt += batch_size
            tqdm_dataloader.set_description('Epoch {}, loss {:.3f}'.format(epoch, loss_sum / loss_cnt))
        avg_loss = loss_sum / loss_cnt
        self.save_recent()
        self.log('Train Epoch {}, loss {:.3f}'.format(epoch, avg_loss))
        if avg_loss < self.best_avg_loss:
            self.save_best()
            self.best_avg_loss = avg_loss

    def save_to(self, path):
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True)
        torch.save(self.bare_model.state_dict(), path)

    def save_recent(self):
        self.save_to(self.checkpoint_path)

    def save_best(self):
        self.save_to(self.best_model_path)

    def log(self, msg):
        with self.log_path.open('a') as f:
            f.write(msg + '\n')

