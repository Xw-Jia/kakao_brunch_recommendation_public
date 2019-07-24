from datetime import date

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import candidate_size
from utils import entropy


class CandidateGenerator:
    def __init__(self, preprocessed, dataset, mf, i2i):
        self.pre = preprocessed
        self.dataset = dataset
        self.mf = mf
        self.i2i = i2i

        heavy_items = self.pre.heavy_items
        self.heavy_items = set(heavy_items)
        self.heavy_items_by_pop = sorted(heavy_items,
                                         key=lambda i: self.pre.article_num2popularity.get(i, 0),
                                         reverse=True)
        feb_items = self.pre.feb_items
        self.feb_items = set(feb_items)
        self.feb_items_by_pop = sorted(feb_items,
                                       key=lambda i: self.pre.article_num2popularity.get(i, 0),
                                       reverse=True)

    def get_following_writers(self, reader_num):
        return self.pre.reader_num2following_nums.get(reader_num, [])

    def get_followable_writers(self, reader_num, already_following):
        reader_dict = self.pre.reader_num2writer_num2article_num_and_date.get(reader_num, {})
        writer_and_read_size = map(lambda p: (p[0], len(p[1])), reader_dict.items())
        writers = [w for w, rsz in sorted(writer_and_read_size, key=lambda p: p[1], reverse=True)]
        return [w for w in writers if w not in already_following]

    def get_writers_articles(self, writer_num):
        return self.pre.writer_num2article_nums.get(writer_num, [])

    def get_writer_of_article(self, article_num):
        return self.pre.article_num2writer_num.get(article_num, None)

    def get_articles_written_date(self, article_num):
        ret = self.pre.article_num2written_date.get(article_num, None)
        if pd.isnull(ret):
            return None
        return ret

    def get_articles_written_date_ordinal(self, article_num):
        written_date = self.get_articles_written_date(article_num)
        if written_date is None:
            return 0
        return written_date.toordinal()

    def get_readers_following_articles(self, reader_num, written_after=date(2019, 2, 22)):
        following = self.get_following_writers(reader_num)
        ret = []
        for writer_num in following:
            _, last_date, _ = self.dataset.interaction_info(reader_num, writer_num)
            written_date_after = written_after if last_date is None else min(last_date, written_after)
            for article_num in self.get_writers_articles(writer_num):
                written_date = self.get_articles_written_date(article_num)
                if written_date is not None and written_date_after <= written_date:
                    ret.append(article_num)
        return ret

    def get_readers_followable_articles(self, reader_num):
        following = self.get_following_writers(reader_num)
        followable = self.get_followable_writers(reader_num, following)
        ret = []
        for writer_num in followable:
            # _, last_date, _ = self.dataset.interaction_info(reader_num, writer_num)
            # written_date_after = written_after if last_date is None else min(last_date, written_after)
            articles = []
            for article_num in self.get_writers_articles(writer_num):
                # written_date = self.get_articles_written_date(article_num)
                if self.dataset.interacted(reader_num, article_num):
                    continue
                if article_num in self.heavy_items:
                    continue
                if article_num in self.feb_items:
                    continue
                # if written_date is not None and written_date_after <= written_date:
                #     ret.append(article_num)
                articles.append(article_num)
            articles.sort(key=lambda a: self.pre.article_num2popularity.get(a, 0), reverse=True)
            # articles = articles[:1]
            ret.extend(articles)
        # ret.sort(key=lambda a: self.pre.article_num2popularity.get(a, 0), reverse=True)
        return ret

    def get_readers_keyword_articles(self, reader_num):
        keywords = self.pre.reader_num2keywords.get(reader_num, [])
        ret = set()
        for keyword in keywords:
            ret.update(self.pre.keyword2article_nums.get(keyword, []))
        ret = [a for a in ret if not self.dataset.interacted(reader_num, a)]
        ret.sort(key=lambda a: self.pre.article_num2popularity.get(a, 0), reverse=True)
        return ret


    def generate_candidates(self, reader_num, N):
        candidates = []
        ret_i2i_ranks = []
        def include_cond(x):
            if x == 0:
                return False
            if x in candidates:
                return False
            if self.dataset.interacted(reader_num, x):
                return False
            return True

        from_following = self.get_readers_following_articles(reader_num, date(2019, 2, 18))
        # from_following.sort(key=lambda a: self.get_articles_written_date_ordinal(a))
        from_following.sort(key=lambda a: self.pre.article_num2popularity.get(a, 0), reverse=True)
        from_following = list(filter(include_cond, from_following))
        candidates.extend(from_following)
        ret_i2i_ranks.extend([np.nan] * len(from_following))

        from_i2i, i2i_ranks = self.i2i.recommend_topn(reader_num, N, include_cond=include_cond)
        candidates.extend(from_i2i)
        ret_i2i_ranks.extend(i2i_ranks)

        # from_keywords = self.get_readers_keyword_articles(reader_num)
        # from_keywords.sort(key=lambda a: self.get_articles_written_date_ordinal(a))
        # candidates.extend(filter(include_cond, from_keywords))

        from_popular = self.feb_items_by_pop[:3*N]
        candidates.extend(filter(include_cond, from_popular))

        # from_follwable = self.get_readers_followable_articles(reader_num)
        # from_follwable.sort(key=lambda a: self.get_articles_written_date_ordinal(a))
        # candidates.extend(filter(include_cond, from_follwable))

        # from_mf = self.mf.recommend_topn(reader_num, N, include_cond=include_cond)
        # candidates.extend(from_mf)

        # from_random = [random.randint(1, article_count) for _ in range(20)]
        # candidates.extend(filter(include_cond, from_random))
        # candidates.extend(from_random)
        #
        # from_popular = []
        # for i in self.heavy_items_by_pop:
        #     if len(from_popular) == 20:
        #         break
        #     if include_cond(i):
        #         from_popular.append(i)
        # candidates.extend(from_popular)

        # cold start
        # if len(candidates) < 100:
        #     from_pop = self.feb_items_by_pop[:5*N]
        #     candidates.extend(filter(include_cond, from_pop))

        return candidates, ret_i2i_ranks

    def extract_features(self, reader, candidates):
        features = pd.DataFrame()
        is_following = []
        following_size = []
        read_size = []
        date_offset = []
        written_ordinal = []
        date_offset_from_last_interaction = []
        reader_read_how_many_of_writers_articles = []
        reader_read_how_many_of_writers_articles_portion = []
        writer_article_count = []
        popularity = []

        is_magazine = []
        date_offset_from_last_magazine_interaction = []
        reader_read_how_many_of_magazine_articles = []
        reader_read_how_many_of_magazine_articles_portion = []
        magazine_article_count = []

        keyword_overlap_count = []

        mf_scores = self.mf.mf_scores(reader, candidates)
        i2i_scores = self.i2i.i2i_scores(reader, candidates)

        following = self.get_following_writers(reader)
        read_count_total = self.pre.reader_num2read_count.get(reader, 0)
        read_date = date(2019, 2, 22)
        reader_keywords = set(self.pre.reader_num2keywords.get(reader, []))
        for article_position, article in enumerate(candidates):
            following_size.append(len(following))
            read_size.append(read_count_total)
            popularity.append(self.pre.article_num2popularity.get(article, 0))

            writer = self.get_writer_of_article(article)
            if writer is not None:
                is_following.append(int(writer in following))

                last_article, last_date, interaction_count = self.dataset.interaction_info(reader, writer)
                reader_read_how_many_of_writers_articles.append(interaction_count)
                writer_article_size = len(self.get_writers_articles(writer))
                if writer_article_size == 0:
                    reader_read_how_many_of_writers_articles_portion.append(np.nan)
                    writer_article_count.append(np.nan)
                else:
                    reader_read_how_many_of_writers_articles_portion.append(interaction_count / writer_article_size)
                    writer_article_count.append(writer_article_size)

                if last_date is not None:
                    offset = (read_date - last_date).days
                    offset = max(0, offset)
                    date_offset_from_last_interaction.append(offset)
                else:
                    date_offset_from_last_interaction.append(np.nan)
            else:
                is_following.append(np.nan)
                reader_read_how_many_of_writers_articles.append(np.nan)
                reader_read_how_many_of_writers_articles_portion.append(np.nan)
                writer_article_count.append(np.nan)
                date_offset_from_last_interaction.append(np.nan)

            written_date = self.get_articles_written_date(article)
            if written_date is None:
                date_offset.append(np.nan)
                written_ordinal.append(np.nan)
            else:
                offset = (read_date - written_date).days
                offset = max(0, offset)
                date_offset.append(offset)
                written_ordinal.append(written_date.toordinal())

            magazine = self.pre.article_num2magazine_num.get(article, None)
            if magazine is None or magazine == 0:
                is_magazine.append(0 if magazine == 0 else np.nan)
                date_offset_from_last_magazine_interaction.append(np.nan)
                reader_read_how_many_of_magazine_articles.append(np.nan)
                reader_read_how_many_of_magazine_articles_portion.append(np.nan)
                magazine_article_count.append(np.nan)
            else:
                is_magazine.append(1)
                last_article, last_date, interaction_count = self.dataset.magazine_interaction_info(reader, magazine)
                magazine_size = len(self.pre.magazine_num2article_nums.get(magazine, []))
                reader_read_how_many_of_magazine_articles.append(interaction_count)
                reader_read_how_many_of_magazine_articles_portion.append(interaction_count / magazine_size)
                magazine_article_count.append(magazine_size)

                if last_date is None:
                    date_offset_from_last_magazine_interaction.append(np.nan)
                else:
                    offset = (read_date - last_date).days
                    offset = max(0, offset)
                    date_offset_from_last_magazine_interaction.append(offset)

            article_keywords = set(self.pre.article_num2keywords.get(article, []))
            keyword_overlap_count.append(len(reader_keywords.intersection(article_keywords)))

        features['is_following'] = is_following
        features['following_size'] = following_size
        features['read_size'] = read_size
        features['date_offset'] = date_offset
        features['written_ordinal'] = written_ordinal
        features['date_offset_from_last_interaction'] = date_offset_from_last_interaction
        features['popularity'] = popularity
        features['reader_read_how_many_of_writers_articles'] = reader_read_how_many_of_writers_articles
        features['reader_read_how_many_of_writers_articles_portion'] = reader_read_how_many_of_writers_articles_portion
        features['writer_article_count'] = writer_article_count
        features['is_magazine'] = is_magazine
        features['date_offset_from_last_magazine_interaction'] = date_offset_from_last_magazine_interaction
        features['reader_read_how_many_of_magazine_articles'] = reader_read_how_many_of_magazine_articles
        features['reader_read_how_many_of_magazine_articles_portion'] = reader_read_how_many_of_magazine_articles_portion
        features['magazine_article_count'] = magazine_article_count
        features['keyword_overlap_count'] = keyword_overlap_count
        features['mf_scores'] = mf_scores
        features['i2i_scores'] = i2i_scores

        return features

    def generate_dataset(self, readers, for_submission=False):
        group_sizes = []
        group_reader = []
        features = []
        labels = []
        entire_candidates = []
        tqdm_readers = tqdm(readers)
        for reader in tqdm_readers:
            candidates, i2i_ranks = self.generate_candidates(reader, candidate_size)
            entire_candidates.extend(candidates)
            group_size = len(candidates)
            group_sizes.append(group_size)
            group_reader.append(reader)
            if not for_submission:
                for article in candidates:
                    labels.append(self.dataset.interacted_in_train_answer(reader, article))
            feat = self.extract_features(reader, candidates)
            i2i_ranks.extend([np.nan] * (group_size - len(i2i_ranks)))
            feat['i2i_ranks'] = i2i_ranks
            features.append(feat)

            tqdm_readers.set_description('Avg cand size {:.2f}'.format(sum(group_sizes) / len(group_sizes)))

        X = pd.concat(features).reset_index().drop(['index'], axis=1)
        y = entire_candidates if for_submission else pd.Series(labels)
        q = np.array(group_sizes)
        q_reader = group_reader
        print('Entropy of candidates = ', entropy(entire_candidates))
        return X, y, q, q_reader

    def generate_train(self):
        return self.generate_dataset(self.dataset.train_users)

    def generate_val(self):
        return self.generate_dataset(self.dataset.val_users)

    def generate_submit(self):
        return self.generate_dataset(self.dataset.submit_users, for_submission=True)
