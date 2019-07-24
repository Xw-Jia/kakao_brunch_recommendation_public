from glob import glob
from itertools import chain
from datetime import date, datetime
import os
import pickle
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

from config import dataset_root
from config import heavy_user_threshold
from config import heavy_item_threshold
from utils import chainer


class Preprocessor:
    def __init__(self, first_time=True):
        self.first_time = first_time
        preprocessed_root = dataset_root.joinpath('preprocessed')
        required_object_names = [
            'read',
            'read_raw',
            'reader2num',
            'num2reader',
            'writer2num',
            'article2num',
            'num2article',
            'magazine2num',
            'interactions_sparse',
            'interactions_dict_before22',
            'interactions_dict_after22',
            'interactions_dict',
            'metadata_bynum',
            'magazine_bynum',
            'users_bynum',
            'read_meta_bynum',

            'reader_num2following_nums',
            'writer_num2article_nums',
            'article_num2writer_num',
            'article_num2written_date',
            'article_num2magazine_num',
            'magazine_num2article_nums',
            'reader_num2read_count',
            'article_num2popularity',
            'reader_num2writer_num2article_num_and_date',
            'reader_num2magazine_num2article_num_and_date',

            'reader_num2keywords',
            'article_num2keywords',
            'keyword2article_nums',

            'heavy_users',
            'heavy_items',
            'train_readers',
            'test_users',
            'feb_items',
        ]
        required_object_names_to_path = {obj_name: preprocessed_root.joinpath(obj_name + '.pkl')
                                         for obj_name in required_object_names}
        self.dataset_root = dataset_root
        self.preprocessed_root = preprocessed_root
        self.required_objects = required_object_names
        self.required_object_names_to_path = required_object_names_to_path

    def _get_required_object_if_exists(self, obj_name):
        path = self.required_object_names_to_path[obj_name]
        if not path.is_file():
            return None
        return pickle.load(path.open('rb'))

    def _get_json_file_from_dataset_folder(self, file_name):
        path = self.dataset_root.joinpath(file_name)
        return pd.read_json(path, lines=True)

    def _get_read_read_raw(self):
        read = self._get_required_object_if_exists('read')
        if read is None:
            paths = self.dataset_root.joinpath('read').joinpath('*')
            read_file_lst = glob(str(paths))
            read_df_lst = []
            for f in read_file_lst:
                file_name = os.path.basename(f)
                df_temp = pd.read_csv(f, header=None, names=['raw'])
                df_temp['dt'] = file_name[:8]
                df_temp['hr'] = file_name[8:10]
                df_temp['user_id'] = df_temp['raw'].str.split(' ').str[0]
                df_temp['article_id'] = df_temp['raw'].str.split(' ').str[1:].str.join(' ').str.strip()
                read_df_lst.append(df_temp)
            read = pd.concat(read_df_lst)
            read = read[read['article_id'] != '']

        read_raw = self._get_required_object_if_exists('read_raw')
        if read_raw is None:
            read_cnt_by_user = read['article_id'].str.split(' ').map(len)
            read_raw = pd.DataFrame({'dt': np.repeat(read['dt'], read_cnt_by_user),
                                     'hr': np.repeat(read['hr'], read_cnt_by_user),
                                     'user_id': np.repeat(read['user_id'], read_cnt_by_user),
                                     'article_id': chainer(read['article_id'])})
        return read, read_raw

    def _load_raw_files(self):
        print('Loading metadata.json')
        metadata = self._get_json_file_from_dataset_folder('metadata.json')
        columns = list(metadata.columns)
        columns[0] = 'id'  # article_id -> id
        columns[2] = 'article_id'  # id -> article_id
        metadata.columns = columns
        self.metadata = metadata
        print('Loading magazine.json')
        self.magazine = self._get_json_file_from_dataset_folder('magazine.json')
        print('Loading users.json')
        self.users = self._get_json_file_from_dataset_folder('users.json')
        print('Getting read, read_raw')
        self.read, self.read_raw = self._get_read_read_raw()

    def _load_reader2num(self):
        print('Loading reader2num')
        reader2num = self._get_required_object_if_exists('reader2num')
        if reader2num is None:
            readers_from_json = set(self.users['id'])
            readers_from_read = set(self.read['user_id'])
            all_readers = readers_from_json.union(readers_from_read)
            reader2num = self._create_mapping(all_readers)
        self.reader2num = reader2num

    def _load_writer2num(self):
        print('Loading writer2num')
        writer2num = self._get_required_object_if_exists('writer2num')
        if writer2num is None:
            writers_from_json = set(chain(*self.users['following_list']))
            writers_from_read = set(s.split('_')[0] for s in self.read_raw['article_id'])
            writers_from_metadata = set(self.metadata['user_id'])
            all_writers = writers_from_json.union(writers_from_read).union(writers_from_metadata)
            writer2num = self._create_mapping(all_writers)
        self.writer2num = writer2num

    def _load_num2reader(self):
        print('Loading num2reader')
        num2reader = self._get_required_object_if_exists('num2reader')
        if num2reader is None:
            num2reader = {v: k for k, v in self.reader2num.items()}
        self.num2reader = num2reader

    def _load_article2num(self):
        print('Loading article2num')
        article2num = self._get_required_object_if_exists('article2num')
        if article2num is None:
            articles_from_json = set(self.metadata['article_id'])
            articles_from_read_raw = set(self.read_raw['article_id'])
            all_articles = articles_from_json.union(articles_from_read_raw)
            assert ('' not in all_articles)
            article2num = self._create_mapping(all_articles)
        self.article2num = article2num

    def _load_num2article(self):
        print('Loading num2article')
        num2article = self._get_required_object_if_exists('num2article')
        if num2article is None:
            num2article = {v: k for k, v in self.article2num.items()}
        self.num2article = num2article

    def _load_magazine2num(self):
        print('Loading magazine2num')
        magazine2num = self._get_required_object_if_exists('magazine2num')
        if magazine2num is None:
            magazines_from_metadata = set(self.metadata['magazine_id'])
            magazines_from_magazine = set(self.magazine['id'])
            all_magazines = magazines_from_metadata.union(magazines_from_magazine)
            magazine2num = self._create_mapping(all_magazines, start_from_zero=True)
        self.magazine2num = magazine2num

    def _load_interactions_sparse(self):
        print('Loading interactions_sparse')
        interactions_sparse = self._get_required_object_if_exists('interactions_sparse')
        if interactions_sparse is None:
            interactions = {}
            for reader, article in zip(self.read_raw['user_id'], self.read_raw['article_id']):
                reader_num = self.reader2num[reader]
                article_num = self.article2num[article]
                interactions.setdefault(reader_num, set()).add(article_num)
            readers = []
            articles = []
            for k, vs in interactions.items():
                for v in vs:
                    readers.append(k)
                    articles.append(v)
            data = [1.0] * len(readers)

            sparse_matrix_shape = (len(self.reader2num) + 1, len(self.article2num) + 1)
            interactions_sparse = csr_matrix((data, (readers, articles)), shape=sparse_matrix_shape)
        self.interactions_sparse = interactions_sparse

    def _load_metadata_bynum(self):
        print('Loading metadata_bynum')
        df = self._get_required_object_if_exists('metadata_bynum')
        if df is None:
            df = pd.DataFrame()
            df['article_num'] = self.metadata['article_id'].map(self.article2num).astype('int32')
            df['writer_num'] = self.metadata['user_id'].map(self.writer2num).astype('int32')
            df['magazine_num'] = self.metadata['magazine_id'].map(self.magazine2num).astype('int32')

            df['keywords'] = self.metadata['keyword_list']
            df['title'] = self.metadata['title'].astype('str')
            df['subtitle'] = self.metadata['sub_title'].astype('str')
            df['write_datetime'] = self.metadata['reg_ts'].apply(lambda x: datetime.fromtimestamp(x / 1000.0))
            df.loc[df['write_datetime'] == df['write_datetime'].min(), 'write_datetime'] = np.nan  # 1970-01-01 is NaT
            df['write_datetime'] = df['write_datetime']
            df['write_date'] = df['write_datetime'].dt.date
            df['write_hour'] = df['write_datetime'].dt.hour.astype('float32')
        self.metadata_bynum = df
        # self.article_num2writer_num = dict(zip(df['article_num'], df['writer_num']))
        # self.article_num2date = dict(zip(df['article_num'], df['date']))

    def _load_magazine_bynum(self):
        print('Loading magazine_bynum')
        df = self._get_required_object_if_exists('magazine_bynum')
        if df is None:
            df = pd.DataFrame()
            df['magazine_num'] = self.magazine['id'].map(self.magazine2num).astype('int32')
            df['tag_list'] = self.magazine['magazine_tag_list']
        self.magazine_bynum = df

    def _load_users_bynum(self):
        print('Loading users_bynum')
        df = self._get_required_object_if_exists('users_bynum')
        if df is None:
            df = pd.DataFrame()
            df['reader_num'] = self.users['id'].map(self.reader2num).astype('int32')
            df['following_nums'] = self.users['following_list'].map(lambda lst: list(map(self.writer2num.get, lst)))
            df['keywords'] = self.users['keyword_list']
        self.users_bynum = df

    def _load_read_meta_bynum(self):
        print('Loading read_meta_bynum')
        df = self._get_required_object_if_exists('read_meta_bynum')
        if df is None:
            df = pd.DataFrame()
            df['reader_num'] = self.read_raw['user_id'].map(self.reader2num).astype('int32')
            df['article_num'] = self.read_raw['article_id'].map(self.article2num).astype('int32')
            df['read_date'] = self.read_raw['dt'].map(lambda x: datetime.strptime(x, '%Y%m%d').date())
            df['read_hour'] = self.read_raw['hr'].astype('int32')
            df = pd.merge(df, self.metadata_bynum, on='article_num', how='left')
            df['writer_num'] = df['writer_num'].astype('float32')
            df['magazine_num'] = df['magazine_num'].astype('float32')
            df['date_offset'] = df['read_date'] - df['write_date']
            df['date_offset'] = df['date_offset'].map(lambda x: x.days).astype('float32')
            df['hour_offset'] = (df['date_offset'] * 24) + (df['read_hour'] - df['write_hour'])
        self.read_meta_bynum = df

    def _load_interactions_dict(self):
        print('Loading interactions_dict')
        interactions_dict_before22 = self._get_required_object_if_exists('interactions_dict_before22')
        interactions_dict_after22 = self._get_required_object_if_exists('interactions_dict_after22')
        interactions_dict = self._get_required_object_if_exists('interactions_dict')
        twotwo = date(2019, 2, 22)
        if any(x is None for x in [interactions_dict, interactions_dict_before22, interactions_dict_after22]):
            interactions_dict_before22 = {}
            interactions_dict_after22 = {}
            interactions_dict = {}
            df = self.read_meta_bynum
            for reader_num, article_num, read_date in zip(df['reader_num'], df['article_num'], df['read_date']):
                if read_date < twotwo:
                    interactions_dict_before22.setdefault(reader_num, set()).add(article_num)
                else:
                    interactions_dict_after22.setdefault(reader_num, set()).add(article_num)
                interactions_dict.setdefault(reader_num, set()).add(article_num)
        self.interactions_dict_before22 = interactions_dict_before22
        self.interactions_dict_after22 = interactions_dict_after22
        self.interactions_dict = interactions_dict

    def _load_various_dictionaries(self):
        print('Loading various dictionaries')
        def dict_from_df(df, key_col, val_col):
            keys = df[key_col]
            values = df[val_col]
            return dict(zip(keys, values))
        # 'reader_num2following_nums',
        reader_num2following_nums = self._get_required_object_if_exists('reader_num2following_nums')
        if reader_num2following_nums is None:
            reader_num2following_nums = dict_from_df(self.users_bynum, 'reader_num', 'following_nums')
        self.reader_num2following_nums = reader_num2following_nums
        # 'writer_num2article_nums',
        # 'article_num2writer_num',
        # 'article_num2written_date',
        # 'article_num2magazine_num',
        # 'magazine_num2article_count',
        writer_num2article_nums = self._get_required_object_if_exists('writer_num2article_nums')
        article_num2writer_num = self._get_required_object_if_exists('article_num2writer_num')
        article_num2written_date = self._get_required_object_if_exists('article_num2written_date')
        article_num2magazine_num = self._get_required_object_if_exists('article_num2magazine_num')
        magazine_num2article_nums = self._get_required_object_if_exists('magazine_num2article_nums')
        if any(x is None for x in [writer_num2article_nums, article_num2writer_num, article_num2written_date,
                                   article_num2magazine_num, magazine_num2article_nums]):
            writer_num2article_nums = {}
            article_num2writer_num = {}
            article_num2written_date = {}
            article_num2magazine_num = {}
            magazine_num2article_nums = {}
            print('Building various dicts from metadata')
            for writer, article, magazine, written_date in \
                    tqdm(zip(self.metadata_bynum['writer_num'],
                             self.metadata_bynum['article_num'],
                             self.metadata_bynum['magazine_num'],
                             self.metadata_bynum['write_date']), total=len(self.metadata_bynum)):
                writer_num2article_nums.setdefault(writer, []).append(article)
                article_num2writer_num[article] = writer
                article_num2written_date[article] = written_date
                article_num2magazine_num[article] = magazine
                magazine_num2article_nums.setdefault(magazine, []).append(article)
        self.writer_num2article_nums = writer_num2article_nums
        self.article_num2writer_num = article_num2writer_num
        self.article_num2written_date = article_num2written_date
        self.article_num2magazine_num = article_num2magazine_num
        self.magazine_num2article_nums = magazine_num2article_nums
        # 'reader_num2read_count',
        reader_num2read_count = self._get_required_object_if_exists('reader_num2read_count')
        if reader_num2read_count is None:
            counts = self.read_meta_bynum['reader_num'].value_counts()
            reader_num2read_count = dict(counts)
        self.reader_num2read_count = reader_num2read_count
        # 'article_num2popularity',
        article_num2popularity = self._get_required_object_if_exists('article_num2popularity')
        if article_num2popularity is None:
            counts = self.read_meta_bynum['article_num'].value_counts()
            article_num2popularity = dict(counts)
        self.article_num2popularity = article_num2popularity
        # 'reader_num2writer_num2article_num_and_date',
        reader_num2writer_num2article_num_and_date = self._get_required_object_if_exists('reader_num2writer_num2article_num_and_date')
        reader_num2magazine_num2article_num_and_date = self._get_required_object_if_exists('reader_num2magazine_num2article_num_and_date')
        if any(x is None for x in [reader_num2writer_num2article_num_and_date, reader_num2magazine_num2article_num_and_date]):
            d = {}
            dm = {}
            df = self.read_meta_bynum
            print('Building various dicts from read_meta')
            for reader_num, writer_num, article_num, read_date, magazine_num in \
                tqdm(zip(df['reader_num'],
                         df['writer_num'],
                         df['article_num'],
                         df['read_date'],
                         df['magazine_num']), total=len(df)):
                if pd.isnull(read_date):
                    read_date = None
                d.setdefault(reader_num, {}).setdefault(writer_num, []).append((article_num, read_date))
                dm.setdefault(reader_num, {}).setdefault(magazine_num, []).append((article_num, read_date))
            for r in d:
                for w in d[r]:
                    d[r][w].sort(key=lambda x: x[1].toordinal() if x[1] is not None else 0)
            for r in dm:
                for m in dm[r]:
                    dm[r][m].sort(key=lambda x: x[1].toordinal() if x[1] is not None else 0)
            reader_num2writer_num2article_num_and_date = d
            reader_num2magazine_num2article_num_and_date = dm
        self.reader_num2writer_num2article_num_and_date = reader_num2writer_num2article_num_and_date
        self.reader_num2magazine_num2article_num_and_date = reader_num2magazine_num2article_num_and_date

    def _load_keywords(self):
        reader_num2keywords = self._get_required_object_if_exists('reader_num2keywords')
        article_num2keywords = self._get_required_object_if_exists('article_num2keywords')
        keyword2article_nums = self._get_required_object_if_exists('keyword2article_nums')
        if any(x is None for x in [reader_num2keywords, article_num2keywords, keyword2article_nums]):
            reader_num2keywords = {}
            article_num2keywords = {}
            keyword2article_nums = {}
            for article_num, keywords in zip(self.metadata_bynum['article_num'], self.metadata_bynum['keywords']):
                article_num2keywords[article_num] = keywords
                for keyword in keywords:
                    keyword2article_nums.setdefault(keyword, []).append(article_num)
            for reader_num, keywords in zip(self.users_bynum['reader_num'], self.users_bynum['keywords']):
                keywords = [d['keyword'] for d in keywords]
                reader_num2keywords[reader_num] = keywords

        self.reader_num2keywords = reader_num2keywords
        self.article_num2keywords = article_num2keywords
        self.keyword2article_nums = keyword2article_nums

    def _sample_and_split(self):
        print('Sample and split')
        heavy_users = self._get_required_object_if_exists('heavy_users')
        heavy_items = self._get_required_object_if_exists('heavy_items')
        if any(x is None for x in [heavy_users, heavy_items]):
            sparse = self.interactions_sparse

            user_item, item_user = sparse, sparse.T.tocsr()
            user_cnt, item_cnt = user_item.shape
            user_nnzs, item_nnzs = user_item.indptr, item_user.indptr
            user_row_nnzs = [user_nnzs[i+1]-user_nnzs[i] for i in range(user_cnt)]
            item_row_nnzs = [item_nnzs[i+1]-item_nnzs[i] for i in range(item_cnt)]
            heavy_users = [u for u in range(user_cnt) if user_row_nnzs[u] >= heavy_user_threshold]
            heavy_items = [i for i in range(item_cnt) if item_row_nnzs[i] >= heavy_item_threshold]
        self.heavy_users = heavy_users
        self.heavy_items = heavy_items
        print('{} heavy users {} heavy items'.format(len(self.heavy_users), len(self.heavy_items)))

        train_readers = self._get_required_object_if_exists('train_readers')
        test_users = self._get_required_object_if_exists('test_users')
        if any(x is None for x in [train_readers, test_users]):
            read_meta = self.read_meta_bynum
            reader2num = self.reader2num

            split_date = date(2019, 2, 22)
            read_after_split = read_meta[read_meta['read_date'] >= split_date]
            readers_after_split = set(read_after_split['reader_num'])

            test_users_path = dataset_root.joinpath('predict').joinpath('test.users')
            test_users = set(reader2num[u.strip()] for u in test_users_path.open())

            train_readers = readers_after_split - test_users
        self.train_readers = train_readers
        self.test_users = test_users

        feb_items = self._get_required_object_if_exists('feb_items')
        if feb_items is None:
            df = self.metadata_bynum

            feb = df[df['write_date'] >= date(2019, 2, 1)]
            feb_items = list(feb['article_num'])
        self.feb_items = feb_items

    def _release_useless_memory(self):
        # Does this code work??
        print('Garbage collecting')
        del self.metadata
        del self.magazine
        del self.users
        del self.read
        del self.read_raw
        del self.metadata_bynum
        del self.magazine_bynum
        del self.users_bynum
        del self.read_meta_bynum
        gc.collect()

    def preprocess(self):
        if not self.preprocessed_root.is_dir():
            self.preprocessed_root.mkdir(parents=True)
        if self.first_time:
            self._load_raw_files()

        self._load_reader2num()
        self._load_num2reader()
        self._load_writer2num()
        self._load_article2num()
        self._load_num2article()
        self._load_magazine2num()
        self._load_interactions_sparse()

        if self.first_time:
            self._load_metadata_bynum()
            self._load_magazine_bynum()
            self._load_users_bynum()
            self._load_read_meta_bynum()
        self._load_interactions_dict()

        self._load_various_dictionaries()
        self._load_keywords()

        self._sample_and_split()

        self._save_all_required_objects_newly_created()
        if self.first_time:
            self._release_useless_memory()

    def _save_all_required_objects_newly_created(self):
        for obj_name, file_path in self.required_object_names_to_path.items():
            # if obj_name not in self.__dict__:
            #     raise AssertionError('{} not preprocessed'.format(obj_name))
            if file_path.is_file():
                continue
            print('Saving', obj_name)
            obj = self.__dict__[obj_name]
            with file_path.open('wb') as f:
                pickle.dump(obj, f)

    @staticmethod
    def _create_mapping(vs, start_from_zero=False):
        vs = sorted(vs)
        offset = 0 if start_from_zero else 1
        d = {v: (i + offset) for i, v in enumerate(vs)}
        display_window = 3
        for i in range(display_window):
            print('Mapped {} -> {}'.format(vs[i], d[vs[i]]))
        print('...')
        for i in range(-display_window, 0):
            print('Mapped {} -> {}'.format(vs[i], d[vs[i]]))
        print()
        return d


if __name__ == '__main__':
    p = Preprocessor()
    p.preprocess()
