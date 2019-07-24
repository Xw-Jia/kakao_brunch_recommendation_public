import random

import torch
import numpy as np

from preprocess import Preprocessor
from dataset import Dataset
from mf import MF
from item2item import Item2Item
from candidate_generator import CandidateGenerator
from ranker import Ranker


def train():
    print('Preprocessing raw data')
    preprocessor = Preprocessor()
    preprocessor.preprocess()

    dataset = Dataset(preprocessor)

    print('Training MF')
    mf = MF(preprocessor, dataset)
    mf.train_or_load_if_exists()

    print('Building I2I')
    i2i = Item2Item(dataset)

    print('Generating candidates')
    candidate_generator = CandidateGenerator(preprocessor, dataset, mf, i2i)
    X_train, y_train, q_train, q_train_reader = candidate_generator.generate_train()
    X_val, y_val, q_val, q_val_reader = candidate_generator.generate_val()

    import pickle
    try:
        with open('puke.pkl', 'wb') as f:
            pickle.dump((X_train, y_train, q_train, q_train_reader,
                         X_val, y_val, q_val, q_val_reader), f)
    except:
        print("Couldn't save puke")

    print('Training ranker')
    ranker = Ranker()
    ranker.train(X_train, y_train, q_train, X_val, y_val, q_val)
    ranker.save()

    print('Validating ranker')
    rank_scores = ranker.rank(X_val)
    print('ndcg', dataset.validate_ndcg(y_val, q_val, q_val_reader, rank_scores))

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    train()
