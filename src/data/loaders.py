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
    ):
        """
        load the click log data

        Args:
            debug
            sample_size

        Returns:
            (training click log, testing click log)
            or (training click log, testing click log, all user ids) if debug is True
        """
        print("Loading click log data...")
        train_path = os.path.join(self.data_path, "train_click_log.csv")
        click_train_df = pd.read_csv(train_path)

        test_path = os.path.join(self.data_path, "testA_click_log.csv")
        click_test_df = pd.read_csv(test_path)
        
        all_user_ids = set(click_train_df["user_id"].unique()).union(
            set(click_test_df["user_id"].unique()))

        if debug or self.config.debug_mode:
            if sample_size is None:
                sample_size = self.config.debug_user_sample_size
            sample_user_ids = np.random.choice(
                list(all_user_ids), size=min(sample_size, len(all_user_ids)), replace=False
            )
            click_train_df = click_train_df[click_train_df["user_id"].isin(sample_user_ids)]
            click_test_df = click_test_df[click_test_df["user_id"].isin(sample_user_ids)]

        click_train_df = click_train_df.drop_duplicates(
            ["user_id", "click_article_id", "click_timestamp"]
        )
        click_test_df = click_test_df.drop_duplicates(
            ["user_id", "click_article_id", "click_timestamp"]
        )
        print(f"Loaded {len(click_train_df)} training click records")
        print(f"Loaded {len(click_test_df)} testing click records")
        
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
        print("Loading article data...")
        articles_path = os.path.join(self.data_path, "articles.csv")
        articles_df = pd.read_csv(articles_path)

        articles_emb_path = os.path.join(self.data_path, "articles_emb.csv")
        articles_emb_df = pd.read_csv(articles_emb_path)

        if debug:
            if sample_size is None:
                sample_size = self.config.debug_sample_size
            articles_df = self._sample_data(articles_df, sample_size)
            articles_emb_df = self._sample_data(articles_emb_df, sample_size)

        articles_df = articles_df.rename(columns={"article_id": "click_article_id"})
        # articles_emb_df = articles_emb_df.rename(
        #     columns={"article_id": "click_article_id"}
        # )
        print(f"Loaded {len(articles_df)} articles")

        return articles_df, articles_emb_df

    def load_with_user_sample(self, article_ids: list):
        """
        Sample users and load data (for quick debugging)

        Args:
            user_clicks_df: User click data
            article_ids: The given article id sample
            
        Returns:
            (sampled article meta data, sampled article embedding data)
        """
        if not self.config.debug_mode:
            raise ValueError("This method is only for debug mode.")
        
        articles_df, articles_emb_df = self.load(debug=False)

        sample_articles = articles_df[articles_df["click_article_id"].isin(article_ids)]
        sample_article_embs = articles_emb_df[articles_emb_df["article_id"].isin(article_ids)]

        return sample_articles, sample_article_embs

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
