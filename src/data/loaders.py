import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from ..utils.config import RecallConfig


class BaseDataLoader(ABC):
    def __init__(self, config: RecallConfig):
        """
        Args:
            config
        """
        self.config = config
        self.data_path = config.data_path

    @abstractmethod
    def load(self, debug: bool = False, sample_size: Optional[int] = None):
        pass

    def _sample_data(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """
        sample the data

        Args:
            df: raw data
            sample_size
        """
        if len(df) <= sample_size:
            return df
        return df[:sample_size]


class ClickLogLoader(BaseDataLoader):
    def load(
        self, debug: bool = False, sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        load the click log data

        Args:
            debug
            sample_size

        Returns:
            (training click log, testing click log)
        """
        train_path = os.path.join(self.data_path, "train_click_log.csv")
        click_train_df = pd.read_csv(train_path)

        test_path = os.path.join(self.data_path, "testA_click_log.csv")
        click_test_df = pd.read_csv(test_path)

        if debug or self.config.debug_mode:
            if sample_size is None:
                sample_size = self.config.debug_sample_size
            click_train_df = self._sample_data(click_train_df, sample_size)
            click_test_df = self._sample_data(click_test_df, sample_size)

        click_train_df = click_train_df.drop_duplicates(
            ["user_id", "click_article_id", "click_timestamp"]
        )
        click_test_df = click_test_df.drop_duplicates(
            ["user_id", "click_article_id", "click_timestamp"]
        )

        return click_train_df, click_test_df

    def load_all(
        self,
        debug: bool = False,
        sample_size: Optional[int] = None,
        offline: bool = True,
    ) -> pd.DataFrame:
        """
        load and merge all click logs (train + test)

        Args:
            debug
            sample_size
            offline: True if only load training data

        Returns:
            merged click logs
        """
        click_train_df, click_test_df = self.load(debug, sample_size)

        if offline:
            return click_train_df

        all_click_df = pd.concat([click_train_df, click_test_df], ignore_index=True)
        all_click_df = all_click_df.drop_duplicates(
            ["user_id", "click_article_id", "click_timestamp"]
        )

        return all_click_df

    def load_with_user_sample(
        self, user_sample_size: int, data_sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        sample users and load data (for quick debugging)

        Args:
            user_sample_size
            data_sample_size
        """
        click_train_df, _ = self.load(debug=True, sample_size=data_sample_size)

        all_user_ids = click_train_df.user_id.unique()

        sample_user_ids = np.random.choice(
            all_user_ids, size=min(user_sample_size, len(all_user_ids)), replace=False
        )

        sampled_df = click_train_df[click_train_df["user_id"].isin(sample_user_ids)]

        return sampled_df


class ArticleInfoLoader(BaseDataLoader):
    def load(
        self, debug: bool = False, sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Args:
            debug: 是否为调试模式
            sample_size: 采样大小

        Returns:
            (article meta data, article embedding data)
        """
        articles_path = os.path.join(self.data_path, "articles.csv")
        articles_df = pd.read_csv(articles_path)

        articles_emb_path = os.path.join(self.data_path, "articles_emb.csv")
        articles_emb_df = pd.read_csv(articles_emb_path)

        if debug or self.config.debug_mode:
            if sample_size is None:
                sample_size = self.config.debug_sample_size
            articles_df = self._sample_data(articles_df, sample_size)
            articles_emb_df = self._sample_data(articles_emb_df, sample_size)

        articles_df = articles_df.rename(columns={"article_id": "click_article_id"})

        return articles_df, articles_emb_df

    def load_article_meta(
        self, debug: bool = False, sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        articles_df, _ = self.load(debug, sample_size)
        return articles_df

    def load_article_embedding(
        self, debug: bool = False, sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        _, articles_emb_df = self.load(debug, sample_size)
        return articles_emb_df
