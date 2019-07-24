import pickle

import lightgbm as lgb

from config import models_root
from config import gbm_n_estimators


class Ranker:
    def __init__(self):
        self.gbm = lgb.LGBMRanker(n_estimators=gbm_n_estimators, importance_type='gain')
        self.ranker_path = models_root.joinpath('ranker.pkl')

    def train(self, X_train, y_train, q_train, X_test, y_test, q_test):
        self.gbm.fit(X_train, y_train, group=q_train,
                     eval_set=[(X_test,y_test)], eval_group=[q_test],
                     eval_metric=['ndcg'], eval_at=[100],
                     early_stopping_rounds=100,
                     categorical_feature=['is_following', 'is_magazine'])

    def load(self):
        if not self.ranker_path.is_file():
            raise FileNotFoundError('Run train.py beforehand')
        self.gbm = pickle.load(self.ranker_path.open('rb'))

    def train_or_load_if_exists(self, X_train, y_train, q_train, X_test, y_test, q_test):
        if not self.ranker_path.is_file():
            print('Training because model file not found')
            self.train(X_train, y_train, q_train, X_test, y_test, q_test)
        else:
            print('Loading saved model instead of training')
            self.load()

    def save(self):
        if not self.ranker_path.parent.is_dir():
            self.ranker_path.parent.mkdir(parents=True)
        with self.ranker_path.open('wb') as f:
            pickle.dump(self.gbm, f)

    def rank(self, X):
        scores = self.gbm.predict(X)
        return scores
