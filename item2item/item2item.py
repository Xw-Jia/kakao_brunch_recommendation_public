import pickle

from tqdm import trange
from sklearn.preprocessing import normalize

from config import models_root
from config import save_memory


class Item2Item:
    def __init__(self, dataset):
        self.dataset = dataset
        self.whole_sparse_with_train_answers_hidden = dataset.whole_sparse_with_train_answers_hidden
        self.item2index = dataset.heavy_item2index
        self.index2item = {v: k for k, v in self.item2index.items()}
        self.scores = self._get_scores()
        if not save_memory:
            self.index_to_similar_indices_ordered_by_score = self._get_similar_indices_dict()

    def _get_scores(self):
        scores_path = models_root.joinpath('i2i_scores.pkl')
        if scores_path.is_file():
            scores = pickle.load(scores_path.open('rb'))
        else:
            scores = self._build_scores()
            self._save(scores, scores_path)
        return scores

    def _get_similar_indices_dict(self):
        index_to_similar_indices_ordered_by_score = self._build_similar_indices_dict(self.scores)
        return index_to_similar_indices_ordered_by_score

    def _build_scores(self):
        user_item = self.dataset.heavy_sparse
        item_user = user_item.T.tocsr()
        item_user_norm = normalize(item_user, norm='l2', axis=1)
        item_user_norm_T = item_user_norm.T.tocsr()
        scores = item_user_norm * item_user_norm_T
        return scores

    def _build_similar_indices_dict(self, scores):
        index_to_similar_indices_oredered_by_score = {}
        print('Building index to similar indices dictionary')
        for index in trange(scores.shape[0]):
            row_scores = scores[index].tocoo()
            indices_and_scores = sorted(zip(row_scores.col, row_scores.data), key=lambda p: p[1], reverse=True)
            # indices = [index for index, score in indices_and_scores]
            index_to_similar_indices_oredered_by_score[index] = indices_and_scores
        return index_to_similar_indices_oredered_by_score

    def _save(self, obj, path):
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True)
        with path.open('wb') as f:
            pickle.dump(obj, f)

    def get_similar_indices_and_scores(self, item_index):
        if save_memory:
            scores = self.scores[item_index].tocoo()
            indices_and_scores = sorted(zip(scores.col, scores.data), key=lambda p: p[1], reverse=True)
        else:
            indices_and_scores = self.index_to_similar_indices_ordered_by_score[item_index]
        return indices_and_scores

    def nearest_N_from_one_item_index(self, item_index, N, include_cond=None, already_read_items=set(), already_selected_items2rank={}):
        indices_and_scores = self.get_similar_indices_and_scores(item_index)
        for rank, (index, score) in enumerate(indices_and_scores[:N]):
            item = self.index2item[index]
            if include_cond is not None and not include_cond(item):
                continue
            if item in already_read_items:
                continue
            if item in already_selected_items2rank:
                already_selected_items2rank[item] = min(already_selected_items2rank[item], rank)
                continue
            already_selected_items2rank[item] = rank

    def nearest_N_from_many_items(self, item_nums, N, include_cond=None):
        item_indices = [self.item2index[i] for i in item_nums if i in self.item2index]
        if len(item_indices) == 0:
            return [], []
        already_read_items = set(item_nums)
        selected_items2rank = {}
        for item_index in item_indices:
            self.nearest_N_from_one_item_index(item_index, N=N, include_cond=include_cond,
                                               already_read_items=already_read_items,
                                               already_selected_items2rank=selected_items2rank)

        selected_items_and_ranks = sorted(selected_items2rank.items(), key=lambda p: p[1])
        items = [x[0] for x in selected_items_and_ranks]
        ranks = [x[1] for x in selected_items_and_ranks]
        return items, ranks

    def recommend_topn(self, reader_num, topn, include_cond=None):
        item_nums = self.whole_sparse_with_train_answers_hidden[reader_num].nonzero()[1]
        items, ranks = self.nearest_N_from_many_items(item_nums, N=topn, include_cond=include_cond)
        return items, ranks

    def similarity_scores(self, item_nums):
        item_indices = [self.item2index[i] for i in item_nums if i in self.item2index]
        if len(item_indices) == 0:
            return {}
        scores = self.scores[item_indices].max(0)
        return dict(zip(scores.col, scores.data))

    def i2i_scores(self, reader_num, candidates):
        item_nums = self.whole_sparse_with_train_answers_hidden[reader_num].nonzero()[1]
        scores = self.similarity_scores(item_nums)
        return [scores.get(self.item2index[item], 0.0)
                if item in self.item2index else 0.0
                for item in candidates]

