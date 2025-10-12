import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from gensim.models import Word2Vec
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer

from src.data import ClickLogLoader, ArticleInfoLoader
from src.data import (
    UserFeatureExtractor,
    ItemFeatureExtractor,
    InteractionFeatureExtractor,
)
from src.utils.config import RecallConfig
from src.utils.persistence import PersistenceManager


class FeatureExtractor:
    """
    Feature extractor for news recommendation ranking stage.

    Basic features include:
        - user_id: user identifier
        - recall_id (article_id): article identifier from recall results
        - score: score from recall results
        - sim_{i}, i=1,...,N: similarity between recall item and last N items in user history
        - time_diff_{i}, i=1,...,N: time difference between recall item and last N items
        - word_diff_{i}, i=1,...,N: word difference between recall item and last N items
        - sim_max, sim_mean, sim_min, sim_std: statistics of similarity features
        - item_user_sim: similarity between item and user from youtubednn embedding
        - user_click_count: normalized user click frequency
        - user_avg_time_gap: normalized average time gap between user clicks
        - article_popularity: normalized article frequency
        - device_group: user's device group (categorical)
        - avg_click_time: average click time for user
        - avg_word_count: average word count of articles clicked by user
        - recall_in_user_cat: whether recall item's category is in user's clicked categories

    For DIN model, we have:
        - user profile:
            - user_id
            - user_click_count
            - user_avg_time_gap
            - device_group
            - avg_click_time
            - avg_word_count
        - item (both for recall item and history items):
            - item_id
            - category_id
            - article_popularity
            - created_at_ts
            - words_count
        - context:
            - score
            - sim_{i}, i=1,...,N
            - time_diff_{i}, i=1,...,N
            - word_diff_{i}, i=1,...,N
            - sim_max, sim_mean, sim_min, sim_std
            - item_user_sim
            - recall_in_user_cat


    All continuous features are converted to discrete features through binning strategy
    for embedding operations in downstream ranking models.
    """

    def __init__(self, config: RecallConfig):
        self.config = config
        self.train_set_df: pd.DataFrame = (
            pd.DataFrame()
        )  # (user_id, item_id, features..., label)
        self.test_set_df: pd.DataFrame = (
            pd.DataFrame()
        )  # (user_id, item_id, features...)
        self.continuous_features: List[str] = []  # list of continuous feature names
        self.discrete_features: List[str] = []  # list of discrete feature names
        self.all_features: List[str] = (
            []
        )  # list of all feature names for downstream use
        self.binning_discretizers: Dict[str, KBinsDiscretizer] = (
            {}
        )  # store discretizers for each continuous feature

        # DIN-specific feature groups
        self.user_profile_features: List[str] = [
            "user_id",
            "user_click_count",
            "user_avg_time_gap",
            "device_group",
            "avg_click_time",
            "avg_word_count",
        ]
        self.item_features: List[str] = [
            "recall_id",
            "category_id",
            "article_popularity",
            "created_at_ts",
            "words_count",
        ]
        self.context_features: List[str] = []  # will be populated dynamically

    def load_data(self):
        """
        load train click data, test click data and article data
        """
        print("start loading data...")
        click_loader = ClickLogLoader(self.config)
        article_loader = ArticleInfoLoader(self.config)
        self.train_click_df, self.test_click_df = click_loader.load(
            debug=self.config.debug_mode, sample_size=self.config.debug_sample_size
        )
        self.article_info_df, article_emb_df = article_loader.load(
            debug=self.config.debug_mode
        )

        (
            self.article_type_dict,
            self.article_words_dict,
            self.article_created_time_dict,
        ) = ItemFeatureExtractor.get_item_info_dict(self.article_info_df)

        # create article content embedding dict from article_emb_df
        emb_cols = [f"emb_{i}" for i in range(250)]
        article_ids = article_emb_df["article_id"].values
        embedding_matrix = article_emb_df[emb_cols].values

        self.article_content_emb_dict = {
            article_id: embedding_matrix[idx]
            for idx, article_id in enumerate(article_ids)
        }

        self.train_user_id = self.train_click_df["user_id"].unique().tolist()
        self.test_user_id = self.test_click_df["user_id"].unique().tolist()
        print("data loaded.")

    def load_recall(self, load_path: Optional[str] = None):
        """
        load recall results from fusion results created by recall pipeline
        you should run recall pipeline first and save to get the recall results
        """
        if load_path is None:
            load_path = self.config.recall_path
        print("start loading recall results...")
        recall_dict = PersistenceManager.load_pickle(load_path)
        df_row_list = []  # [user, item, score]
        for user, recall_list in recall_dict.items():
            for item, score in recall_list:
                df_row_list.append([user, item, score])

        self.recall_df = pd.DataFrame(
            df_row_list, columns=["user_id", "recall_id", "score"]
        )
        self.train_set_df = self.recall_df[
            self.recall_df["user_id"].isin(self.train_user_id)
        ].copy()
        self.test_set_df = self.recall_df[
            self.recall_df["user_id"].isin(self.test_user_id)
        ].copy()

        print("recall results loaded.")

    def load_all(self):
        self.load_data()
        self.load_recall()

    @staticmethod
    def get_article_id_emb(
        click_df: pd.DataFrame, embed_size: int = 64, save_path: Optional[str] = None
    ):
        """
        get the article id embedding by word2vec
        using click log data

        Returns:
            {article_id: embedding_vector}
        """
        click_df = click_df.sort_values(by="click_timestamp").copy()
        click_df["click_article_id"] = click_df["click_article_id"].astype(
            str
        )  # ensure input is str
        user_click_seq = (
            click_df.groupby("user_id")["click_article_id"].apply(list).tolist()
        )
        model = Word2Vec(
            sentences=user_click_seq,
            vector_size=embed_size,
            window=5,
            min_count=1,
            workers=4,
            sg=1,
            epochs=10,
        )

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
        if not hasattr(self, "article_id_emb_dict"):
            self.article_id_emb_dict = self.get_article_id_emb(
                self.train_click_df,
                embed_size=self.config.embedding_dim,
                save_path=self.config.save_path + "article_id_emb.pkl",
            )

        if os.path.exists(self.config.save_path + "user_youtubednn_emb.pkl"):
            self.user_youtubednn_emb_dict = PersistenceManager.load_pickle(
                self.config.save_path + "user_youtubednn_emb.pkl"
            )

        if os.path.exists(self.config.save_path + "article_youtubednn_emb.pkl"):
            self.article_youtubednn_emb_dict = PersistenceManager.load_pickle(
                self.config.save_path + "article_youtubednn_emb.pkl"
            )

    def _add_labels(self):
        """
        add labels to the recall results, that is, generating self.train_label_df
        1 for positive samples, 0 for negative samples
        here we set the label of test set to -1 (unknown)
        """
        train_history, train_last_click = (
            InteractionFeatureExtractor.get_hist_and_last_click(self.train_click_df)
        )
        # convert train_history to dict for fast lookup
        self.train_history_dict = {}
        for user_id, group in train_history.groupby("user_id"):
            self.train_history_dict[user_id] = list(
                zip(group["click_article_id"], group["click_timestamp"])
            )

        self.test_history_dict = {}
        for user_id, group in self.test_click_df.groupby("user_id"):
            self.test_history_dict[user_id] = list(
                zip(group["click_article_id"], group["click_timestamp"])
            )

        train_last_click.rename(columns={"click_article_id": "recall_id"}, inplace=True)

        self.train_set_df = pd.merge(
            self.train_set_df,
            train_last_click,
            on=["user_id", "recall_id"],
            how="left",
            indicator=True,
        )
        self.train_set_df["label"] = (self.train_set_df["_merge"] == "both").astype(int)

        self.test_set_df["label"] = -1  # unknown label for test set

    def _negative_sampling(self):
        """
        negative sampling for train set
        use it at the end of feature extraction
        """
        if self.config.neg_sample_rate <= 0:
            return

        pos_df = self.train_set_df[self.train_set_df["label"] == 1]
        neg_df = self.train_set_df[self.train_set_df["label"] == 0]

        print(
            f"positive samples: {len(pos_df)}, negative samples: {len(neg_df)}, pos/neg ratio: {len(pos_df)/len(neg_df):.4f}"
        )

        def neg_sample_func(group_df):
            neg_num = len(group_df)
            sample_num = max(1, int(neg_num * self.config.neg_sample_rate))
            sample_num = min(self.config.min_sample_size, sample_num)
            return group_df.sample(
                n=sample_num, replace=True, random_state=self.config.random_seed
            )

        neg_user_sample = (
            neg_df.groupby("user_id", group_keys=False)
            .apply(neg_sample_func)
            .reset_index(drop=True)
        )
        neg_item_sample = (
            neg_df.groupby("recall_id", group_keys=False)
            .apply(neg_sample_func)
            .reset_index(drop=True)
        )

        # neg_new = neg_user_sample.append(neg_item_sample).drop_duplicates(['user_id', 'recall_id']).reset_index(drop=True)
        neg_new = (
            pd.concat([neg_user_sample, neg_item_sample], ignore_index=True)
            .drop_duplicates(["user_id", "recall_id"])  # type: ignore
            .reset_index(drop=True)
        )
        self.train_set_df = pd.concat([pos_df, neg_new], ignore_index=True)

        print(
            f"after negative sampling, train set size: {len(self.train_set_df)}, pos/neg ratio: {len(pos_df)/len(neg_new):.4f}"
        )

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
        if not hasattr(self, "article_id_emb_dict"):
            raise ValueError("you should run _get_all_emb_dict() before this function")

        N = self.config.last_N
        for i in range(1, N + 1):
            self.train_set_df[f"sim_{i}"] = np.nan
            self.train_set_df[f"time_diff_{i}"] = np.nan
            self.train_set_df[f"word_diff_{i}"] = np.nan
            self.test_set_df[f"sim_{i}"] = np.nan
            self.test_set_df[f"time_diff_{i}"] = np.nan
            self.test_set_df[f"word_diff_{i}"] = np.nan

        datasets = [
            {
                "name": "train",
                "df": self.train_set_df,
                "history_dict": self.train_history_dict,
                "desc": "Processing train users",
            },
            {
                "name": "test",
                "df": self.test_set_df,
                "history_dict": self.test_history_dict,
                "desc": "Processing test users",
            },
        ]

        for dataset in datasets:
            df = dataset["df"]
            history_dict = dataset["history_dict"]

            for user_id, user_group in tqdm(
                df.groupby("user_id"), desc=dataset["desc"]
            ):
                if user_id not in history_dict:
                    continue

                history = history_dict[user_id][-N:]  # get last N items
                user_indices = user_group.index
                recall_items = user_group["recall_id"].values
                user_emb = (
                    self.user_youtubednn_emb_dict.get(user_id)
                    if hasattr(self, "user_youtubednn_emb_dict")
                    else None
                )

                if user_emb is not None and hasattr(
                    self, "article_youtubednn_emb_dict"
                ):
                    item_user_sim_array = np.array(
                        [
                            (
                                np.dot(user_emb, self.article_youtubednn_emb_dict[x])
                                if x in self.article_youtubednn_emb_dict
                                else 0.0
                            )
                            for x in recall_items
                        ]
                    )
                else:
                    item_user_sim_array = np.zeros(len(recall_items))

                df.loc[user_indices, "item_user_sim"] = item_user_sim_array

                for hist_idx, (hist_item, hist_time) in enumerate(history):
                    hist_idx_col = hist_idx + 1

                    hist_emb_norm = self.article_id_emb_dict.get(hist_item)
                    hist_content_emb = self.article_content_emb_dict.get(hist_item)

                    sim_array = np.zeros(len(recall_items))
                    time_diff_array = np.full(len(recall_items), np.nan)
                    word_diff_array = np.full(len(recall_items), np.nan)

                    for i, recall_item in enumerate(recall_items):
                        # 1. article id embedding similarity
                        if (
                            hist_emb_norm is not None
                            and recall_item in self.article_id_emb_dict
                        ):
                            recall_emb_norm = self.article_id_emb_dict[recall_item]
                            sim_array[i] = np.dot(hist_emb_norm, recall_emb_norm)

                        # 2. time difference
                        if (
                            hist_item in self.article_created_time_dict
                            and recall_item in self.article_created_time_dict
                        ):
                            time_diff_array[i] = abs(
                                self.article_created_time_dict[recall_item]
                                - self.article_created_time_dict[hist_item]
                            )

                        # 3. content embedding distance
                        if (
                            hist_content_emb is not None
                            and recall_item in self.article_content_emb_dict
                        ):
                            recall_content_emb = self.article_content_emb_dict[
                                recall_item
                            ]
                            word_diff_array[i] = np.linalg.norm(
                                hist_content_emb - recall_content_emb
                            )

                    df.loc[user_indices, f"sim_{hist_idx_col}"] = sim_array
                    df.loc[user_indices, f"time_diff_{hist_idx_col}"] = time_diff_array
                    df.loc[user_indices, f"word_diff_{hist_idx_col}"] = word_diff_array

            # adding statistics features
            sim_cols = [f"sim_{i}" for i in range(1, N + 1)]
            df["sim_max"] = df[sim_cols].max(axis=1)
            df["sim_mean"] = df[sim_cols].mean(axis=1)
            df["sim_min"] = df[sim_cols].min(axis=1)
            df["sim_std"] = df[sim_cols].std(axis=1)

    def _add_user_activate_degree(self):
        """
        add user activate degree feature
        calculate 1 / freq and average time gap for every user, then normalize them as the feature

        Features:
        1. user_click_count: normalized user click frequency
        2. user_avg_time_gap: normalized average time gap between consecutive clicks
        """
        train_user_stats = (
            self.train_click_df.groupby("user_id")
            .agg(
                {
                    "click_article_id": "count",
                    "click_timestamp": lambda x: (
                        (x.max() - x.min()) / (len(x) - 1) if len(x) > 1 else 0
                    ),
                }
            )
            .reset_index()
        )
        train_user_stats.columns = ["user_id", "click_count", "avg_time_gap"]

        scaler_count = MinMaxScaler()
        scaler_time_gap = MinMaxScaler()

        train_user_stats["user_click_count"] = scaler_count.fit_transform(
            train_user_stats[["click_count"]]
        )
        train_user_stats["user_avg_time_gap"] = scaler_time_gap.fit_transform(
            train_user_stats[["avg_time_gap"]]
        )

        self.train_set_df = pd.merge(
            self.train_set_df,
            train_user_stats[["user_id", "user_click_count", "user_avg_time_gap"]],
            on="user_id",
            how="left",
        )

        test_user_stats = (
            self.test_click_df.groupby("user_id")
            .agg(
                {
                    "click_article_id": "count",
                    "click_timestamp": lambda x: (
                        (x.max() - x.min()) / (len(x) - 1) if len(x) > 1 else 0
                    ),
                }
            )
            .reset_index()
        )
        test_user_stats.columns = ["user_id", "click_count", "avg_time_gap"]

        test_user_stats["user_click_count"] = scaler_count.fit_transform(
            test_user_stats[["click_count"]]
        )

        test_user_stats["user_avg_time_gap"] = scaler_time_gap.transform(
            test_user_stats[["avg_time_gap"]]
        )

        self.test_set_df = pd.merge(
            self.test_set_df,
            test_user_stats[["user_id", "user_click_count", "user_avg_time_gap"]],
            on="user_id",
            how="left",
        )

        self.train_set_df["user_click_count"].fillna(0.5, inplace=True)
        self.train_set_df["user_avg_time_gap"].fillna(0.5, inplace=True)
        self.test_set_df["user_click_count"].fillna(0.5, inplace=True)
        self.test_set_df["user_avg_time_gap"].fillna(0.5, inplace=True)

    def _add_article_popularity(self):
        """
        add article popularity feature
        calculate the popularity of each article in the training click data
        then normalize it as the feature

        Feature:
        article_popularity: normalized article popularity
        """
        article_popularity = (
            self.train_click_df["click_article_id"].value_counts().reset_index()
        )
        article_popularity.columns = ["article_id", "popularity"]

        scaler = MinMaxScaler()
        article_popularity["article_popularity"] = scaler.fit_transform(
            article_popularity[["popularity"]]
        )

        self.train_set_df = pd.merge(
            self.train_set_df,
            article_popularity[["article_id", "article_popularity"]],
            left_on="recall_id",
            right_on="article_id",
            how="left",
        ).drop(columns=["article_id"])

        self.test_set_df = pd.merge(
            self.test_set_df,
            article_popularity[["article_id", "article_popularity"]],
            left_on="recall_id",
            right_on="article_id",
            how="left",
        ).drop(columns=["article_id"])

        self.train_set_df["article_popularity"].fillna(0, inplace=True)
        self.test_set_df["article_popularity"].fillna(0, inplace=True)

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
        user_device = (
            self.train_click_df.groupby("user_id")["click_deviceGroup"]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else "unknown")
            .reset_index()
        )
        user_device.columns = ["user_id", "device_group"]

        user_click_time = (
            self.train_click_df.groupby("user_id")["click_timestamp"]
            .mean()
            .reset_index()
        )
        user_click_time.columns = ["user_id", "avg_click_time"]

        user_article_types = (
            self.train_click_df.merge(
                self.article_info_df[["article_id", "category_id"]],
                left_on="click_article_id",
                right_on="article_id",
                how="left",
            )
            .groupby("user_id")["category_id"]
            .agg(lambda x: set(x.dropna().unique()))
            .reset_index()
        )
        user_article_types.columns = ["user_id", "article_category_ids"]

        # calculate avg word count
        if not hasattr(self, "user_avg_word_count_dict"):
            self.user_avg_word_count_dict = {}
        for user_id, group in self.train_click_df.groupby("user_id"):
            article_ids = group["click_article_id"].unique()
            word_counts = [
                self.article_words_dict.get(article_id, 0) for article_id in article_ids
            ]
            avg_word_count = np.mean(word_counts) if word_counts else 0

            self.user_avg_word_count_dict[user_id] = avg_word_count

        scaler_time = MinMaxScaler()
        user_click_time["avg_click_time"] = scaler_time.fit_transform(
            user_click_time[["avg_click_time"]]
        )

        self.train_set_df = pd.merge(
            self.train_set_df, user_device, on="user_id", how="left"
        )
        self.train_set_df = pd.merge(
            self.train_set_df, user_click_time, on="user_id", how="left"
        )
        self.train_set_df = pd.merge(
            self.train_set_df, user_article_types, on="user_id", how="left"
        )

        self.test_set_df = pd.merge(
            self.test_set_df, user_device, on="user_id", how="left"
        )
        self.test_set_df = pd.merge(
            self.test_set_df, user_click_time, on="user_id", how="left"
        )
        self.test_set_df = pd.merge(
            self.test_set_df, user_article_types, on="user_id", how="left"
        )

        self.train_set_df["device_group"].fillna("unknown", inplace=True)
        self.train_set_df["avg_click_time"].fillna(0.5, inplace=True)
        self.test_set_df["device_group"].fillna("unknown", inplace=True)
        self.test_set_df["avg_click_time"].fillna(0.5, inplace=True)

        # whether recall item's category_id is in the set of the clicked article categories
        self.train_set_df["recall_in_user_cat"] = self.train_set_df.apply(
            lambda row: (
                1
                if self.article_type_dict.get(row["recall_id"])
                in row["article_category_ids"]
                else 0
            ),
            axis=1,
        )
        self.test_set_df["recall_in_user_cat"] = self.test_set_df.apply(
            lambda row: (
                1
                if self.article_type_dict.get(row["recall_id"])
                in row["article_category_ids"]
                else 0
            ),
            axis=1,
        )

        self.train_set_df["avg_word_count"] = (
            self.train_set_df["user_id"].map(self.user_avg_word_count_dict).fillna(0)
        )
        self.test_set_df["avg_word_count"] = (
            self.test_set_df["user_id"].map(self.user_avg_word_count_dict).fillna(0)
        )

    def _add_item_features(self):
        """
        Add item-level features from article_info_df to both train and test sets.
        These features include:
        - category_id: article category
        - created_at_ts: article creation timestamp
        - words_count: number of words in article
        """
        # prepare item feature dataframe
        item_feature_df = self.article_info_df[
            ["article_id", "category_id", "created_at_ts", "words_count"]
        ].copy()

        # merge with train set
        self.train_set_df = pd.merge(
            self.train_set_df,
            item_feature_df,
            left_on="recall_id",
            right_on="article_id",
            how="left",
        ).drop(columns=["article_id"])

        # merge with test set
        self.test_set_df = pd.merge(
            self.test_set_df,
            item_feature_df,
            left_on="recall_id",
            right_on="article_id",
            how="left",
        ).drop(columns=["article_id"])

        # fill missing values
        self.train_set_df["category_id"].fillna(0, inplace=True)
        self.train_set_df["created_at_ts"].fillna(0, inplace=True)
        self.train_set_df["words_count"].fillna(0, inplace=True)

        self.test_set_df["category_id"].fillna(0, inplace=True)
        self.test_set_df["created_at_ts"].fillna(0, inplace=True)
        self.test_set_df["words_count"].fillna(0, inplace=True)

        print("Item features added.")

    def _identify_feature_types(self):
        """
        Identify continuous and discrete features from the feature set.
        Exclude ID columns, label, and temporary columns.
        """
        # columns to exclude from feature engineering
        exclude_cols = {
            "user_id",
            "recall_id",
            "article_id",
            "label",
            "click_timestamp",
            "article_category_ids",
            "_merge",
        }

        # get all columns from train_set_df
        all_cols = set(self.train_set_df.columns)
        feature_cols = all_cols - exclude_cols

        # identify continuous features (numeric types)
        self.continuous_features = []
        self.discrete_features = []

        for col in feature_cols:
            if col in self.train_set_df.columns:
                dtype = self.train_set_df[col].dtype
                # check if the feature is numeric and has more than a certain number of unique values
                if pd.api.types.is_numeric_dtype(dtype):
                    n_unique = self.train_set_df[col].nunique()
                    # if unique values > 20, treat as continuous, otherwise as discrete
                    if n_unique > 20:
                        self.continuous_features.append(col)
                    else:
                        self.discrete_features.append(col)
                else:
                    # non-numeric features are discrete
                    self.discrete_features.append(col)

        print(
            f"Identified {len(self.continuous_features)} continuous features: {self.continuous_features}"
        )
        print(
            f"Identified {len(self.discrete_features)} discrete features: {self.discrete_features}"
        )

    def _apply_binning(self):
        """
        Apply binning strategy to convert continuous features to discrete features.
        Use KBinsDiscretizer to bin continuous features.
        The binning strategy is determined by config.binning_strategy.
        """
        if not self.config.enable_binning or len(self.continuous_features) == 0:
            raise ValueError("No continuous features to bin or binning not enabled.")

        print(
            f"Applying binning strategy: {self.config.binning_strategy} with {self.config.default_n_bins} bins"
        )

        for feature in tqdm(
            self.continuous_features, desc="Binning continuous features"
        ):
            # skip if feature doesn't exist in dataframes
            if feature not in self.train_set_df.columns:
                continue

            # prepare data for binning
            train_data = self.train_set_df[[feature]].copy()
            test_data = self.test_set_df[[feature]].copy()

            # handle missing values before binning
            train_data[feature].fillna(train_data[feature].median(), inplace=True)
            test_data[feature].fillna(train_data[feature].median(), inplace=True)

            # determine number of bins based on unique values
            n_unique = train_data[feature].nunique()
            n_bins = min(self.config.default_n_bins, n_unique)

            if n_bins < 2:
                print(f"Skipping {feature}: insufficient unique values for binning")
                continue

            # create discretizer
            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy=self.config.binning_strategy,  # type: ignore
                subsample=None,
            )

            try:
                # fit on training data
                discretizer.fit(train_data[[feature]])

                # transform both train and test
                train_binned = discretizer.transform(train_data[[feature]])
                test_binned = discretizer.transform(test_data[[feature]])

                # convert to numpy array and then to int
                self.train_set_df[feature] = (
                    np.asarray(train_binned).astype(int).flatten()
                )
                self.test_set_df[feature] = (
                    np.asarray(test_binned).astype(int).flatten()
                )

                # store discretizer for future use
                self.binning_discretizers[feature] = discretizer

            except Exception as e:
                print(f"Error binning feature {feature}: {e}")
                continue

        print(f"Binning completed for {len(self.binning_discretizers)} features.")

    def _build_feature_list(self):
        """
        Build a comprehensive feature list for downstream use.
        After binning, all features should be discrete (categorical).
        """
        # exclude ID columns, label, and temporary columns
        exclude_cols = {
            # "user_id",
            # "recall_id",
            # "article_id",
            "label",
            "click_timestamp",
            "article_category_ids",
            "_merge",
        }

        # get all columns from train_set_df
        all_cols = set(self.train_set_df.columns)
        self.all_features = sorted(list(all_cols - exclude_cols))

        print(f"Total features for downstream use: {len(self.all_features)}")
        print(f"Feature list: {self.all_features}")

        # save feature list to config for downstream access
        self.config.features = self.all_features

    def _organize_din_features(self):
        """
        Organize features into groups for DIN model:
        - user_profile_features: user-level features
        - item_features: item-level features (for both recall and history items)
        - context_features: context-level features (interaction features)
        """
        # dynamically determine context features
        # context features are those not in user_profile or item features
        exclude_cols = {
            "user_id",
            "recall_id",
            "article_id",
            "label",
            "click_timestamp",
            "article_category_ids",
            "_merge",
        }

        all_feature_cols = set(self.train_set_df.columns) - exclude_cols
        user_profile_set = set(self.user_profile_features) - {
            "user_id"
        }  # keep user_id separate
        item_feature_set = set(self.item_features) - {
            "recall_id"
        }  # keep recall_id separate

        # context features = all features - user_profile - item_features
        self.context_features = sorted(
            list(
                all_feature_cols
                - user_profile_set
                - item_feature_set
                - {"user_id", "recall_id"}
            )
        )

        print(f"User profile features: {self.user_profile_features}")
        print(f"Item features: {self.item_features}")
        print(f"Context features: {self.context_features}")

    def _save_din_data(self, save_path: str):
        """
        Save DIN-specific data structures with discretized features:
        1. train_history_dict: {user_id: [(item_id, timestamp), ...]}
        2. test_history_dict: {user_id: [(item_id, timestamp), ...]}
        3. user_profile_features: list of user profile feature names
        4. item_features: list of item feature names
        5. context_features: list of context feature names
        6. article_info_dict: {article_id: {feature_name: discretized_value, ...}}

        Note: article_info_dict contains DISCRETIZED feature values from train/test sets
        to ensure all features used in DIN model are categorical.
        """
        # save history dicts (already created in _add_labels)
        PersistenceManager.save_pickle(
            self.train_history_dict, os.path.join(save_path, "train_history_dict.pkl")
        )
        PersistenceManager.save_pickle(
            self.test_history_dict, os.path.join(save_path, "test_history_dict.pkl")
        )

        # save feature group lists
        feature_groups = {
            "user_profile_features": self.user_profile_features,
            "item_features": self.item_features,
            "context_features": self.context_features,
        }
        PersistenceManager.save_pickle(
            feature_groups["user_profile_features"],
            os.path.join(save_path, "user_profile_features.pkl"),
        )
        PersistenceManager.save_pickle(
            feature_groups["item_features"],
            os.path.join(save_path, "item_features.pkl"),
        )
        PersistenceManager.save_pickle(
            feature_groups["context_features"],
            os.path.join(save_path, "context_features.pkl"),
        )

        # create article_info_dict with DISCRETIZED features from train/test sets
        # this ensures all features are categorical as required by DIN model
        article_info_dict = {}

        # combine train and test to get all articles with their discretized features
        all_data = pd.concat([self.train_set_df, self.test_set_df], ignore_index=True)

        # extract item features (excluding recall_id and article_popularity which is in dataframe)
        item_feature_cols = [feat for feat in self.item_features if feat != "recall_id"]

        # for each unique article, get its discretized feature values
        for article_id in all_data["recall_id"].unique():
            article_rows = all_data[all_data["recall_id"] == article_id]

            if len(article_rows) > 0:
                # use the first occurrence (all should be the same after discretization)
                first_row = article_rows.iloc[0]
                article_info_dict[article_id] = {}

                for feat in item_feature_cols:
                    if feat in first_row:
                        # use discretized value from dataframe
                        article_info_dict[article_id][feat] = int(first_row[feat])
                    else:
                        article_info_dict[article_id][feat] = 0

        # also add articles from article_info_df that might not appear in train/test
        # these will use default discretized values (0)
        for article_id in self.article_info_df["article_id"].unique():
            if article_id not in article_info_dict:
                article_info_dict[article_id] = {feat: 0 for feat in item_feature_cols}

        PersistenceManager.save_pickle(
            article_info_dict, os.path.join(save_path, "article_info_dict.pkl")
        )

        print(f"DIN-specific data saved to {save_path}")
        print(
            f"Article info dict contains {len(article_info_dict)} articles with discretized features"
        )

    def extract_features(self, save: bool = True):
        """
        Extract all features step by step, including:
        1. Add labels
        2. Load embedding dictionaries
        3. Extract history features
        4. Extract user activate degree features
        5. Extract article popularity features
        6. Extract user habits features
        7. Add item features (category, created_at_ts, words_count)
        8. Perform negative sampling
        9. Identify feature types (continuous vs discrete)
        10. Apply binning to convert continuous features to discrete
        11. Build feature list for downstream use
        12. Organize DIN-specific feature groups
        13. Save DIN-specific data structures
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

        self._add_item_features()
        print("item features added.")

        self._negative_sampling()
        print("negative sampling done.")

        # identify feature types before binning
        self._identify_feature_types()

        # apply binning to convert continuous features to discrete
        self._apply_binning()

        # build comprehensive feature list for downstream use
        self._build_feature_list()

        # organize DIN-specific feature groups
        self._organize_din_features()

        if save:
            # save train and test sets
            self.train_set_df.to_csv(
                os.path.join(self.config.save_path, "train_features.csv"), index=False
            )
            self.test_set_df.to_csv(
                os.path.join(self.config.save_path, "test_features.csv"), index=False
            )
            print(f"features saved to {self.config.save_path}")

            # save feature list and binning discretizers
            PersistenceManager.save_pickle(
                self.all_features,
                os.path.join(self.config.save_path, "feature_list.pkl"),
            )
            # PersistenceManager.save_pickle(
            #     self.binning_discretizers,
            #     os.path.join(self.config.save_path, "binning_discretizers.pkl"),
            # )

            # save DIN-specific data structures
            self._save_din_data(self.config.save_path)

            print(f"features saved to {self.config.save_path}")

    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.train_set_df, self.test_set_df

    def get_feature_list(self) -> List[str]:
        """
        Get the list of all features for downstream use.

        Returns:
            List of feature names
        """
        return self.all_features

    def get_binning_discretizers(self) -> Dict[str, KBinsDiscretizer]:
        """
        Get the discretizers used for binning continuous features.
        Useful for applying the same binning to new data.

        Returns:
            Dictionary mapping feature names to their discretizers
        """
        return self.binning_discretizers
