import random
import pickle

import torch
import numpy as np
from tqdm import tqdm

from preprocess import Preprocessor
from dataset import Dataset
from mf import MF
from item2item import Item2Item
from candidate_generator import CandidateGenerator
from ranker import Ranker
from config import result_path
from config import article_count
from utils import entropy


def inference():
    preprocessor = Preprocessor(first_time=False)
    preprocessor.preprocess()
    dataset = Dataset(preprocessor)
    mf = MF(preprocessor, dataset)
    mf.load()
    i2i = Item2Item(dataset)
    candidate_generator = CandidateGenerator(preprocessor, dataset, mf, i2i)
    ranker = Ranker()
    ranker.load()

    X_submit, X_article_nums, q_submit, q_reader = candidate_generator.generate_submit()
    try:
        with open('submit_puke.pkl', 'wb') as f:
            pickle.dump((X_submit, X_article_nums, q_submit, q_reader), f)
    except:
        print("Couldn't save submit_puke")

    # X_submit, X_article_nums, q_submit, q_reader = pickle.load(open('submit_puke.pkl', 'rb'))

    rank_scores = ranker.rank(X_submit)
    base = 0
    entire_articles = []
    not_heavy_items = set(range(1, article_count+1)) - set(preprocessor.heavy_items)
    not_heavy_items = sorted(not_heavy_items)
    cut = 50

    random.seed(0)
    with result_path.open('w') as fout:
        for group_size, reader in tqdm(zip(q_submit, q_reader), total=len(q_submit)):
            articles = X_article_nums[base:base+group_size]
            scores = rank_scores[base:base+group_size]

            articles = [a for _, a in sorted(zip(scores, articles), key=lambda x: x[0], reverse=True)]
            articles = articles[:cut]
            from_followable = candidate_generator.get_readers_followable_articles(reader)
            # from_keywords = candidate_generator.get_readers_keyword_articles(reader)
            for item in from_followable:
                if len(articles) >= cut + 15:
                    break
                if item in articles:
                    continue
                articles.append(item)
            while len(articles) < 100:
                item = random.choice(not_heavy_items)
                if item not in articles:
                    articles.append(item)
            entire_articles.extend(articles)

            reader_str = preprocessor.num2reader[reader]
            article_strs = map(preprocessor.num2article.get, articles)

            fout.write('%s %s\n' % (reader_str, ' '.join(article_strs)))

            base += group_size
    print('Entropy of candidates = ', entropy(entire_articles))


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    inference()
