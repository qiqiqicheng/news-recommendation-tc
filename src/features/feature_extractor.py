import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from gensim.models import Word2Vec
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from src.data import ClickLogLoader, ArticleInfoLoader
from src.data import UserFeatureExtractor, ItemFeatureExtractor, InteractionFeatureExtractor
from src.utils.config import RecallConfig
from src.utils.persistence import PersistenceManager

class FeatureExtractor:
    """
    current features:
        user_id
        article_id -- from recall resultes
        score -- from recall results
        sim_{i}, i=1,...,N
        time_diff_{i}, i=1,...,N
        word_diff_{i}, i=1,...,N
        sim_max, sim_mean, sim_min, sim_std
        item_user_sim -- from youtubednn embedding
        user_click_count -- normalized user click frequency
        user_avg_time_gap -- normalized average time gap between user clicks
        article_popularity -- normalized article frequency
        device_group
        avg_click_time
        avg_word_count
        recall_in_user_cat -- whether recall item's category_id is in the set of the clicked article categories
    """
    def __init__(self, config: RecallConfig):
        self.config = config
        self.train_set_df: pd.DataFrame = pd.DataFrame()  # (user_id, item_id, features..., label)
        self.test_set_df: pd.DataFrame = pd.DataFrame()  # (user_id, item_id, features...)
        
    def load_data(self):
        """
        load train click data, test click data and article data
        """
        print("start loading data...")
        click_loader = ClickLogLoader(self.config)
        article_loader = ArticleInfoLoader(self.config)
        self.train_click_df, self.test_click_df = click_loader.load(debug=self.config.debug_mode, sample_size=self.config.debug_sample_size)
        self.article_info_df, article_emb_df = article_loader.load(debug=self.config.debug_mode)

        self.article_type_dict, self.article_words_dict, self.article_created_time_dict = ItemFeatureExtractor.get_item_info_dict(self.article_info_df)

        # create article content embedding dict from article_emb_df
        emb_cols = [f'emb_{i}' for i in range(250)]
        article_ids = article_emb_df['article_id'].values
        embedding_matrix = article_emb_df[emb_cols].values

        self.article_content_emb_dict = {
            article_id: embedding_matrix[idx] 
            for idx, article_id in enumerate(article_ids)
        }
        
        self.train_user_id = self.train_click_df['user_id'].unique().tolist()
        self.test_user_id = self.test_click_df['user_id'].unique().tolist()
        print("data loaded.")
        
    def load_recall(self, load_path: Optional[str]=None):
        """
        load recall results from fusion results created by recall pipeline
        you should run recall pipeline first and save to get the recall results
        """
        if load_path is None:
            load_path = self.config.recall_path
        print("start loading recall results...")
        recall_dict = PersistenceManager.load_pickle(load_path)
        df_row_list = [] # [user, item, score]
        for user, recall_list in recall_dict.items():
            for item, score in recall_list:
                df_row_list.append([user, item, score])
                
        self.recall_df = pd.DataFrame(df_row_list, columns=["user_id", "recall_id", "score"])
        self.train_set_df = self.recall_df[self.recall_df['user_id'].isin(self.train_user_id)].copy()
        self.test_set_df = self.recall_df[self.recall_df['user_id'].isin(self.test_user_id)].copy()
        
        print("recall results loaded.")
        
    def load_all(self):
        self.load_data()
        self.load_recall()
        
    @staticmethod
    def get_article_id_emb(click_df: pd.DataFrame, embed_size: int=64, save_path: Optional[str]=None):
        """
        get the article id embedding by word2vec
        using click log data
        
        Returns:
            {article_id: embedding_vector}
        """
        click_df = click_df.sort_values(by="click_timestamp")
        user_click_seq = click_df.groupby("user_id")["click_article_id"].apply(list).tolist()
        model = Word2Vec(sentences=user_click_seq, vector_size=embed_size, window=5, min_count=1, workers=4, sg=1, epochs=10)
        
        article_ids = list(model.wv.index_to_key)
        article_embs = model.wv[article_ids]
        article_id_emb_dict = dict(zip(article_ids, article_embs))
        
        if save_path:
            PersistenceManager.save_pickle(article_id_emb_dict, save_path)
            
        return article_id_emb_dict
    
    def _get_all_emb_dict(self):
        """
        get all embedding dicts:
        1. article id embedding dict: {article_id: embedding_vector}
        2. article content embedding dict: {article_id: embedding_vector}
        3. user youtubednn embedding dict: {user_id: embedding_vector}
        4. article youtubednn embedding dict: {article_id: embedding_vector}
        """
        if not hasattr(self, 'article_id_emb_dict'):
            self.article_id_emb_dict = self.get_article_id_emb(self.train_click_df, embed_size=self.config.embedding_dim, save_path=self.config.save_path + "article_id_emb.pkl")
    
        if os.path.exists(self.config.save_path + "user_youtubednn_emb.pkl"):
            self.user_youtubednn_emb_dict = PersistenceManager.load_pickle(self.config.save_path + "user_youtubednn_emb.pkl")

        if os.path.exists(self.config.save_path + "article_youtubednn_emb.pkl"):
            self.article_youtubednn_emb_dict = PersistenceManager.load_pickle(self.config.save_path + "article_youtubednn_emb.pkl")

    def _add_labels(self):
        """
        add labels to the recall results, that is, generating self.train_label_df
        1 for positive samples, 0 for negative samples
        here we set the label of test set to -1 (unknown)
        """        
        train_history, train_last_click = InteractionFeatureExtractor.get_hist_and_last_click(self.train_click_df)
        # convert train_history to dict for fast lookup
        self.train_history_dict = {}
        for user_id, group in train_history.groupby('user_id'):
            self.train_history_dict[user_id] = list(
                zip(group['click_article_id'], group['click_timestamp'])
            )
        
        self.test_history_dict = {}
        for user_id, group in self.test_click_df.groupby('user_id'):
            self.test_history_dict[user_id] = list(
                zip(group['click_article_id'], group['click_timestamp'])
            )

        train_last_click.rename(columns={"click_article_id": "recall_id"}, inplace=True)
        
        self.train_set_df = pd.merge(self.train_set_df, train_last_click, on=['user_id', 'recall_id'], how='left', indicator=True)
        self.train_set_df['label'] = (self.train_set_df['_merge'] == 'both').astype(int)

        self.test_set_df['label'] = -1  # unknown label for test set
    
    def _negative_sampling(self):
        """
        negative sampling for train set
        use it at the end of feature extraction
        """
        if self.config.neg_sample_rate <= 0:
            return
        
        pos_df = self.train_set_df[self.train_set_df['label'] == 1]
        neg_df = self.train_set_df[self.train_set_df['label'] == 0]
        
        print(f"positive samples: {len(pos_df)}, negative samples: {len(neg_df)}, pos/neg ratio: {len(pos_df)/len(neg_df):.4f}")

        def neg_sample_func(group_df):
            neg_num = len(group_df)
            sample_num = max(1, int(neg_num * self.config.neg_sample_rate))
            sample_num = min(self.config.min_sample_size, sample_num)
            return group_df.sample(n=sample_num, replace=True, random_state=self.config.random_seed)
        
        neg_user_sample = neg_df.groupby('user_id', group_keys=False).apply(neg_sample_func).reset_index(drop=True)
        neg_item_sample = neg_df.groupby('recall_id', group_keys=False).apply(neg_sample_func).reset_index(drop=True)
        
        # neg_new = neg_user_sample.append(neg_item_sample).drop_duplicates(['user_id', 'recall_id']).reset_index(drop=True)
        neg_new = pd.concat([neg_user_sample, neg_item_sample], ignore_index=True).drop_duplicates(['user_id', 'recall_id']).reset_index(drop=True)
        self.train_set_df = pd.concat([pos_df, neg_new], ignore_index=True)
        
        print(f"after negative sampling, train set size: {len(self.train_set_df)}, pos/neg ratio: {len(pos_df)/len(neg_new):.4f}")
        
    def _add_history_features(self):
        """
        add history features for train set and test set
        including:
        1. sim_{i}, i=1,...,N: the similarity between recall item and last N items in user behavior sequence
        2. time_diff_{i}, i=1,...,N: the time difference between recall item and last N items in user behavior sequence
        3. word_diff_{i}, i=1,...,N: the word difference between recall item and last N items in user behavior sequence
        4. sim_max, sim_mean, sim_min, sim_std
        5. item_user_sim: the similarity between recall item and user embedding

        you should run _get_all_emb_dict() before this function
        """
        if not hasattr(self, 'article_id_emb_dict'):
            self._get_all_emb_dict()
        
        N= self.config.last_N
        for i in range(1, N + 1):
            self.train_set_df[f'sim_{i}'] = np.nan
            self.train_set_df[f'time_diff_{i}'] = np.nan
            self.train_set_df[f'word_diff_{i}'] = np.nan
            self.test_set_df[f'sim_{i}'] = np.nan
            self.test_set_df[f'time_diff_{i}'] = np.nan
            self.test_set_df[f'word_diff_{i}'] = np.nan
            
        datasets = [
            {
                'name': 'train',
                'df': self.train_set_df,
                'history_dict': self.train_history_dict,
                'desc': 'Processing train users'
            },
            {
                'name': 'test',
                'df': self.test_set_df,
                'history_dict': self.test_history_dict,
                'desc': 'Processing test users'
            }
        ]
        
        for dataset in datasets:
            df = dataset['df']
            history_dict = dataset['history_dict']
            
            for user_id, user_group in tqdm(df.groupby('user_id'), desc=dataset['desc']):
                if user_id not in history_dict:
                    continue

                history = history_dict[user_id][-N:]  # get last N items
                user_indices = user_group.index
                recall_items = user_group['recall_id'].values
                user_emb = self.user_youtubednn_emb_dict.get(user_id) if hasattr(self, 'user_youtubednn_emb_dict') else None

                if user_emb is not None and hasattr(self, 'article_youtubednn_emb_dict'):
                    item_user_sim_array = np.array([
                        np.dot(user_emb, self.article_youtubednn_emb_dict[x]) 
                        if x in self.article_youtubednn_emb_dict else 0.0
                        for x in recall_items
                    ])
                else:
                    item_user_sim_array = np.zeros(len(recall_items))

                df.loc[user_indices, 'item_user_sim'] = item_user_sim_array
                
                for hist_idx, (hist_item, hist_time) in enumerate(history):
                    hist_idx_col = hist_idx + 1
                    
                    hist_emb_norm = self.article_id_emb_dict.get(hist_item)
                    hist_content_emb = self.article_content_emb_dict.get(hist_item)
                    
                    sim_array = np.zeros(len(recall_items))
                    time_diff_array = np.full(len(recall_items), np.nan)
                    word_diff_array = np.full(len(recall_items), np.nan)
                    
                    for i, recall_item in enumerate(recall_items):
                        # 1. article id embedding similarity
                        if hist_emb_norm is not None and recall_item in self.article_id_emb_dict:
                            recall_emb_norm = self.article_id_emb_dict[recall_item]
                            sim_array[i] = np.dot(hist_emb_norm, recall_emb_norm)
                        
                        # 2. time difference
                        if hist_item in self.article_created_time_dict and recall_item in self.article_created_time_dict:
                            time_diff_array[i] = abs(
                                self.article_created_time_dict[recall_item] - 
                                self.article_created_time_dict[hist_item]
                            )
                        
                        # 3. content embedding distance
                        if hist_content_emb is not None and recall_item in self.article_content_emb_dict:
                            recall_content_emb = self.article_content_emb_dict[recall_item]
                            word_diff_array[i] = np.linalg.norm(hist_content_emb - recall_content_emb)
                        
                    
                    df.loc[user_indices, f'sim_{hist_idx_col}'] = sim_array
                    df.loc[user_indices, f'time_diff_{hist_idx_col}'] = time_diff_array
                    df.loc[user_indices, f'word_diff_{hist_idx_col}'] = word_diff_array
            
            # adding statistics features
            sim_cols = [f'sim_{i}' for i in range(1, N + 1)]
            df['sim_max'] = df[sim_cols].max(axis=1)
            df['sim_mean'] = df[sim_cols].mean(axis=1)
            df['sim_min'] = df[sim_cols].min(axis=1)
            df['sim_std'] = df[sim_cols].std(axis=1)

    def _add_user_activate_degree(self):
        """
        add user activate degree feature
        calculate 1 / freq and average time gap for every user, then normalize them as the feature
        
        Features:
        1. user_click_count: normalized user click frequency
        2. user_avg_time_gap: normalized average time gap between consecutive clicks
        """
        train_user_stats = self.train_click_df.groupby('user_id').agg({
            'click_article_id': 'count',
            'click_timestamp': lambda x: (x.max() - x.min()) / (len(x) - 1) if len(x) > 1 else 0
        }).reset_index()
        train_user_stats.columns = ['user_id', 'click_count', 'avg_time_gap']

        
        scaler_count = MinMaxScaler()
        scaler_time_gap = MinMaxScaler()
        
        train_user_stats['user_click_count'] = scaler_count.fit_transform(
            train_user_stats[['click_count']]
        )
        train_user_stats['user_avg_time_gap'] = scaler_time_gap.fit_transform(
            train_user_stats[['avg_time_gap']]
        )
        
        self.train_set_df = pd.merge(
            self.train_set_df,
            train_user_stats[['user_id', 'user_click_count', 'user_avg_time_gap']],
            on='user_id',
            how='left'
        )
        
        test_user_stats = self.test_click_df.groupby('user_id').agg({
            'click_article_id': 'count',
            'click_timestamp': lambda x: (x.max() - x.min()) / (len(x) - 1) if len(x) > 1 else 0
        }).reset_index()
        test_user_stats.columns = ['user_id', 'click_count', 'avg_time_gap']

        test_user_stats['user_click_count'] = scaler_count.fit_transform(
            test_user_stats[['click_count']]
        )
        
        test_user_stats['user_avg_time_gap'] = scaler_time_gap.transform(
            test_user_stats[['avg_time_gap']]
        )
        
        self.test_set_df = pd.merge(
            self.test_set_df,
            test_user_stats[['user_id', 'user_click_count', 'user_avg_time_gap']],
            on='user_id',
            how='left'
        )
        
        self.train_set_df['user_click_count'].fillna(0.5, inplace=True)
        self.train_set_df['user_avg_time_gap'].fillna(0.5, inplace=True)
        self.test_set_df['user_click_count'].fillna(0.5, inplace=True)
        self.test_set_df['user_avg_time_gap'].fillna(0.5, inplace=True)

    def _add_article_popularity(self):
        """
        add article popularity feature
        calculate the popularity of each article in the training click data
        then normalize it as the feature
        
        Feature:
        article_popularity: normalized article popularity
        """
        article_popularity = self.train_click_df['click_article_id'].value_counts().reset_index()
        article_popularity.columns = ['article_id', 'popularity']
        
        scaler = MinMaxScaler()
        article_popularity['article_popularity'] = scaler.fit_transform(
            article_popularity[['popularity']]
        )
        
        self.train_set_df = pd.merge(
            self.train_set_df,
            article_popularity[['article_id', 'article_popularity']],
            left_on='recall_id',
            right_on='article_id',
            how='left'
        ).drop(columns=['article_id'])
        
        self.test_set_df = pd.merge(
            self.test_set_df,
            article_popularity[['article_id', 'article_popularity']],
            left_on='recall_id',
            right_on='article_id',
            how='left'
        ).drop(columns=['article_id'])
        
        self.train_set_df['article_popularity'].fillna(0, inplace=True)
        self.test_set_df['article_popularity'].fillna(0, inplace=True)
        
    def _add_user_habits(self):
        """
        add user habits features
        Features:
        1. the mode of click_deviceGroup
        2. the mean of click_timestamp after minmaxscaler
        3. the mean of created_at_ts after minmaxscaler
        4. the mean of word_count
        5. whether recall item's category_id is in the set of the clicked article categories
        """
        user_device = self.train_click_df.groupby('user_id')['click_deviceGroup'].agg(lambda x: x.mode()[0] if len(x) > 0 else 'unknown').reset_index()
        user_device.columns = ['user_id', 'device_group']
        
        user_click_time = self.train_click_df.groupby('user_id')['click_timestamp'].mean().reset_index()
        user_click_time.columns = ['user_id', 'avg_click_time']
        
        user_article_types = self.train_click_df.merge(
            self.article_info_df[['article_id', 'category_id']],
            left_on='click_article_id',
            right_on='article_id',
            how='left'
        ).groupby('user_id')['category_id'].agg(lambda x: set(x.dropna().unique())).reset_index()
        user_article_types.columns = ['user_id', 'article_category_ids']
        
        # calculate avg word count
        if not hasattr(self, 'user_avg_word_count_dict'):
            self.user_avg_word_count_dict = {}
        for user_id, group in self.train_click_df.groupby('user_id'):
            article_ids = group['click_article_id'].unique()
            word_counts = [self.article_words_dict.get(article_id, 0) for article_id in article_ids]
            avg_word_count = np.mean(word_counts) if word_counts else 0

            self.user_avg_word_count_dict[user_id] = avg_word_count
        
        scaler_time = MinMaxScaler()
        user_click_time['avg_click_time'] = scaler_time.fit_transform(user_click_time[['avg_click_time']])
        
        self.train_set_df = pd.merge(self.train_set_df, user_device, on='user_id', how='left')
        self.train_set_df = pd.merge(self.train_set_df, user_click_time, on='user_id', how='left')
        self.train_set_df = pd.merge(self.train_set_df, user_article_types, on='user_id', how='left')
        
        self.test_set_df = pd.merge(self.test_set_df, user_device, on='user_id', how='left')
        self.test_set_df = pd.merge(self.test_set_df, user_click_time, on='user_id', how='left')
        self.test_set_df = pd.merge(self.test_set_df, user_article_types, on='user_id', how='left')
        
        self.train_set_df['device_group'].fillna('unknown', inplace=True)
        self.train_set_df['avg_click_time'].fillna(0.5, inplace=True)
        self.test_set_df['device_group'].fillna('unknown', inplace=True)
        self.test_set_df['avg_click_time'].fillna(0.5, inplace=True)
        
        # whether recall item's category_id is in the set of the clicked article categories
        self.train_set_df['recall_in_user_cat'] = self.train_set_df.apply(
            lambda row: 1 if self.article_type_dict.get(row['recall_id']) in row['article_category_ids'] else 0,
            axis=1
        )
        self.test_set_df['recall_in_user_cat'] = self.test_set_df.apply(
            lambda row: 1 if self.article_type_dict.get(row['recall_id']) in row['article_category_ids'] else 0,
            axis=1
        )
        
        self.train_set_df['avg_word_count'] = self.train_set_df['user_id'].map(self.user_avg_word_count_dict).fillna(0)
        self.test_set_df['avg_word_count'] = self.test_set_df['user_id'].map(self.user_avg_word_count_dict).fillna(0)
        
    def extract_features(self, save: bool=True):
        """
        extract all features step by step
        """
        print("start extracting features...")
        self._add_labels()
        print("labels added.")
        
        self._get_all_emb_dict()
        print("all embedding dicts ready.")
        
        self._add_history_features()
        print("history features added.")
        
        self._add_user_activate_degree()
        print("user activate degree features added.")
        
        self._add_article_popularity()
        print("article popularity feature added.")
        
        self._add_user_habits()
        print("user habits features added.")
        
        self._negative_sampling()
        print("negative sampling done.")
        
        if save:
            self.train_set_df.to_csv(self.config.save_path + "train_features.csv", index=False)
            self.test_set_df.to_csv(self.config.save_path + "test_features.csv", index=False)
            print(f"features saved to {self.config.save_path}")
    
    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.train_set_df, self.test_set_df