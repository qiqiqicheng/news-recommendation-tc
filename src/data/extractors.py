import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler


class UserFeatureExtractor:
    @staticmethod
    def get_user_item_time_dict(
        click_df: pd.DataFrame,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Args:
            click_df

        Returns:
            {user_id: [(item_id, timestamp), ...]}
        """
        click_df = click_df.sort_values("click_timestamp")

        def make_item_time_pair(df):
            return list(zip(df["click_article_id"], df["click_timestamp"]))

        user_item_time_df = (
            click_df.groupby("user_id")[["click_article_id", "click_timestamp"]]
            .apply(lambda x: make_item_time_pair(x))
            .reset_index()
            .rename(columns={0: "item_time_list"})
        )

        user_item_time_dict = dict(
            zip(user_item_time_df["user_id"], user_item_time_df["item_time_list"])
        )

        return user_item_time_dict

    @staticmethod
    def get_user_activate_degree_dict(click_df: pd.DataFrame) -> Dict[int, float]:
        """
        get the user activate degree dict(click count normalized)

        Args:
            click_df

        Returns:
            {user_id: activate_degree}
        """
        user_click_count = (
            click_df.groupby("user_id")["click_article_id"].count().reset_index()
        )

        mm = MinMaxScaler()
        user_click_count["click_article_id"] = mm.fit_transform(
            user_click_count[["click_article_id"]]
        )

        user_activate_degree_dict = dict(
            zip(user_click_count["user_id"], user_click_count["click_article_id"])
        )

        return user_activate_degree_dict

    @staticmethod
    def get_user_hist_item_info_dict(
        click_df: pd.DataFrame,
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Args:
            click_df: merged click log data including item info

        Returns:
            (user_hist_item_typs_dict, user_hist_item_ids_dict,
             user_hist_item_words_dict, user_last_item_created_time_dict)
             
                user_hist_item_typs_dict: {user_id: set(category_id)}
                user_hist_item_ids_dict: {user_id: set(item_id)}
                user_hist_item_words_dict: {user_id: avg_words_count}
                user_last_item_created_time_dict: {user_id: last_created_time}
        """
        # 用户历史点击文章类型集合
        user_hist_item_typs = (
            click_df.groupby("user_id")["category_id"].agg(set).reset_index()
        )
        user_hist_item_typs_dict = dict(
            zip(user_hist_item_typs["user_id"], user_hist_item_typs["category_id"])
        )

        # 用户点击文章ID集合
        user_hist_item_ids = (
            click_df.groupby("user_id")["click_article_id"].agg(set).reset_index()
        )
        user_hist_item_ids_dict = dict(
            zip(user_hist_item_ids["user_id"], user_hist_item_ids["click_article_id"])
        )

        # 用户历史点击文章平均字数
        user_hist_item_words = (
            click_df.groupby("user_id")["words_count"].agg("mean").reset_index()
        )
        user_hist_item_words_dict = dict(
            zip(user_hist_item_words["user_id"], user_hist_item_words["words_count"])
        )

        # 用户最后一次点击文章的创建时间
        click_df_sorted = click_df.sort_values("click_timestamp")
        user_last_item_created_time = (
            click_df_sorted.groupby("user_id")["created_at_ts"]
            .apply(lambda x: x.iloc[-1])
            .reset_index()
        )

        # 归一化时间
        mm = MinMaxScaler()
        user_last_item_created_time["created_at_ts"] = mm.fit_transform(
            user_last_item_created_time[["created_at_ts"]]
        )

        user_last_item_created_time_dict = dict(
            zip(
                user_last_item_created_time["user_id"],
                user_last_item_created_time["created_at_ts"],
            )
        )

        return (
            user_hist_item_typs_dict,
            user_hist_item_ids_dict,
            user_hist_item_words_dict,
            user_last_item_created_time_dict,
        )


class ItemFeatureExtractor:
    @staticmethod
    def get_item_info_dict(item_info_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
        """
        Args:
            item_info_df

        Returns:
            (item_type_dict, item_words_dict, item_created_time_dict)
             
                item_type_dict: {item_id: category_id}
                item_words_dict: {item_id: words_count}
                item_created_time_dict: {item_id: created_time}
        """
        mm = MinMaxScaler()
        item_info_df = item_info_df.copy()
        item_info_df["created_at_ts"] = mm.fit_transform(
            item_info_df[["created_at_ts"]]
        )

        item_type_dict = dict(
            zip(item_info_df["click_article_id"], item_info_df["category_id"])
        )
        item_words_dict = dict(
            zip(item_info_df["click_article_id"], item_info_df["words_count"])
        )
        item_created_time_dict = dict(
            zip(item_info_df["click_article_id"], item_info_df["created_at_ts"])
        )

        return item_type_dict, item_words_dict, item_created_time_dict

    @staticmethod
    def get_item_topk_click(click_df: pd.DataFrame, k: int = 50) -> List[int]:
        """
        Args:
            click_df
            k

        Returns:
            list[int]
        """
        topk_click = click_df["click_article_id"].value_counts().index[:k].tolist()
        return topk_click

    @staticmethod
    def get_item_embedding_dict(
        item_emb_df: pd.DataFrame, save_path: str = ""
    ) -> Dict[int, np.ndarray]:
        """
        Args:
            item_emb_df
            save_path: use if you want to save the embedding dict

        Returns:
            {item_id: embedding_vector}
        """
        item_emb_cols = [col for col in item_emb_df.columns if "emb" in col]
        item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])

        item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

        item_emb_dict = dict(zip(item_emb_df["article_id"], item_emb_np))

        if save_path:
            from ..utils.persistence import PersistenceManager

            PersistenceManager.save_pickle(item_emb_dict, save_path)

        return item_emb_dict


class InteractionFeatureExtractor:
    @staticmethod
    def get_item_user_time_dict(
        click_df: pd.DataFrame,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        get item-user-time dict for UserCF

        Args:
            click_df

        Returns:
            {item_id: [(user_id, timestamp), ...]}
        """

        def make_user_time_pair(df):
            return list(zip(df["user_id"], df["click_timestamp"]))

        click_df = click_df.sort_values("click_timestamp")
        item_user_time_df = (
            click_df.groupby("click_article_id")[["user_id", "click_timestamp"]]
            .apply(lambda x: make_user_time_pair(x))
            .reset_index()
            .rename(columns={0: "user_time_list"})
        )

        item_user_time_dict = dict(
            zip(
                item_user_time_df["click_article_id"],
                item_user_time_df["user_time_list"],
            )
        )

        return item_user_time_dict

    @staticmethod
    def get_hist_and_last_click(
        click_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        split history and last click(for offline evaluation)

        Args:
            click_df

        Returns:
            (click_hist_df, click_last_df)
        """
        click_df = click_df.sort_values(by=["user_id", "click_timestamp"])

        # 获取每个用户的最后一次点击
        click_last_df = click_df.groupby("user_id").tail(1)

        # 获取历史点击(如果用户只有一次点击,也包含进去避免数据丢失)
        def hist_func(user_df):
            if len(user_df) == 1:
                return user_df
            else:
                return user_df[:-1]

        click_hist_df = (
            click_df.groupby("user_id").apply(hist_func).reset_index(drop=True)
        )

        return click_hist_df, click_last_df

    @staticmethod
    def normalize_timestamp(
        df: pd.DataFrame, timestamp_col: str = "click_timestamp"
    ) -> pd.DataFrame:
        df = df.copy()
        mm = MinMaxScaler()
        df[timestamp_col] = mm.fit_transform(df[[timestamp_col]])
        return df
