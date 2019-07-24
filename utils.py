from itertools import chain
import pickle
import math

from config import dataset_root


def chainer(s):
    return list(chain.from_iterable(s.str.split(' ')))


def load_preprocessed(name):
    path = dataset_root.joinpath('preprocessed').joinpath(name+'.pkl')
    return pickle.load(path.open('rb'))


def entropy(items):
    sz = len(items)
    freq = {}
    for i in items:
        freq[i] = freq.get(i, 0) + 1
    ent = -sum([v / sz * math.log(v / sz) for v in freq.values()])
    return ent

