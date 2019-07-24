import random
import math

from config import dataset_root
from config import train_sample_size
from config import val_sample_size
from config import submit_user_set


class Dataset:
    def __init__(self, preprocessed):
        reader2num = preprocessed.reader2num
        if submit_user_set == 'dev':
            userlist_path = dataset_root.joinpath('predict').joinpath('dev.users')
            submit_users = [reader2num[u.strip()] for u in userlist_path.open()]
        elif submit_user_set == 'test':
            userlist_path = dataset_root.joinpath('predict').joinpath('test.users')
            submit_users = [reader2num[u.strip()] for u in userlist_path.open()]
        else:
            raise ValueError('submit either dev or test')

        heavy_sparse = preprocessed.interactions_sparse[preprocessed.heavy_users]
        heavy_sparse = heavy_sparse[:, preprocessed.heavy_items]
        heavy_user2index = {v: i for i, v in enumerate(preprocessed.heavy_users)}
        heavy_item2index = {v: i for i, v in enumerate(preprocessed.heavy_items)}

        train_readers = sorted(preprocessed.train_readers)
        random.shuffle(train_readers)

        train_users = train_readers[:train_sample_size]
        val_users = train_readers[train_sample_size:train_sample_size + val_sample_size]
        train_val_users = train_users + val_users
        train_val_answer = {}
        whole_sparse_with_train_answers_hidden = preprocessed.interactions_sparse.tolil()

        for user in train_val_users:
            seen = list(preprocessed.interactions_dict_after22[user])
            random.shuffle(seen)
            hide_seen = seen[:(len(seen)+1)//2]
            train_val_answer[user] = hide_seen
            for s in hide_seen:
                whole_sparse_with_train_answers_hidden[user, s] = 0.0
        avg_hidden_answer_len = sum(map(len, train_val_answer.values())) / len(train_val_answer)
        print('Average length of hidden answers = {}'.format(avg_hidden_answer_len))

        self.preprocessed = preprocessed
        self.heavy_sparse = heavy_sparse
        self.heavy_user2index = heavy_user2index
        self.heavy_item2index = heavy_item2index
        self.train_users = train_users
        self.val_users = val_users
        self.train_val_answer = train_val_answer
        self.submit_users = submit_users
        self.whole_sparse_with_train_answers_hidden = whole_sparse_with_train_answers_hidden

    def interacted(self, reader_num, article_num):
        if reader_num in self.preprocessed.interactions_dict:
            if article_num in self.preprocessed.interactions_dict[reader_num]:
                if reader_num in self.train_val_answer and article_num in self.train_val_answer[reader_num]:
                    return False
                return True
        return False

    def interacted_in_train_answer(self, reader_num, article_num):
        assert reader_num in self.train_val_answer
        return article_num in self.train_val_answer[reader_num]

    def interaction_info(self, reader_num, writer_num):
        interactions = self.preprocessed.reader_num2writer_num2article_num_and_date.get(reader_num, {}).get(writer_num, [])
        hide_items = self.train_val_answer.get(reader_num, [])
        interactions = [(i, d) for i, d in interactions if i not in hide_items]
        interaction_count = len(interactions)
        last_interaction_article_num = None
        last_interaction_date = None
        if len(interactions) > 0:
            last_interaction_article_num, last_interaction_date = interactions[-1]
        return last_interaction_article_num, last_interaction_date, interaction_count

    def magazine_interaction_info(self, reader_num, magazine_num):
        interactions = self.preprocessed.reader_num2magazine_num2article_num_and_date.get(reader_num, {}).get(magazine_num, [])
        hide_items = self.train_val_answer.get(reader_num, [])
        interactions = [(i, d) for i, d in interactions if i not in hide_items]
        interactions_count = len(interactions)
        last_interaction_article_num = None
        last_interaction_date = None
        if len(interactions) > 0:
            last_interaction_article_num, last_interaction_date = interactions[-1]
        return last_interaction_article_num, last_interaction_date, interactions_count

    def validate_ndcg(self, labels, groups, group_readers, scores):
        Q, S = 0, 0
        base = 0
        for group_size, reader in zip(groups, group_readers):
            assert reader in self.train_val_answer, 'We only validate for train/val users'
            seen = self.train_val_answer[reader]
            if group_size == 0 or len(seen) == 0:
                continue
            ls = labels[base:base+group_size]
            ss = scores[base:base+group_size]
            base += group_size
            sort = sorted(zip(ls, ss), key=lambda x: x[1], reverse=True)
            idcg = sum([1.0 / math.log(i + 2, 2) for i in range(min(group_size, len(seen)))])
            dcg = 0.0
            for i, (l, s) in enumerate(sort):
                rank = i+1
                dcg += l / math.log(rank+1, 2)
            ndcg = dcg / idcg
            S += ndcg
            Q += 1
        return S / Q
