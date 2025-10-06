import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from .base import BaseRecaller
from ..utils.weights import WeightCalculator


class UserCFRecaller(BaseRecaller):
    def __init__(
        self,
        config,
        similarity_matrix: Dict,
        user_item_time_dict: Dict,
        item_created_time_dict: Dict,
        item_topk_click: List,
        emb_similarity_matrix: Dict = {},
    ):
        """
        Args:
            config: Configuration object
            similarity_matrix: User-to-user similarity matrix
            user_item_time_dict: Dictionary of user interaction history
            item_created_time_dict: Dictionary of item creation times
            item_topk_click: List of top-k popular items for cold start
            emb_similarity_matrix: Optional item embedding similarity matrix
        """
        super().__init__(config)
        self.u2u_sim = similarity_matrix
        self.user_item_time_dict = user_item_time_dict
        self.item_created_time_dict = item_created_time_dict
        self.item_topk_click = item_topk_click
        self.emb_i2i_sim = emb_similarity_matrix or {}
        self.weight_calc = WeightCalculator()

    def recall(self, user_id: int, topk: int = 10) -> List[Tuple[int, float]]:
        """
        Recall top-k items for a user using UserCF

        Args:
            user_id
            topk

        Returns:
            List of (item_id, score) tuples
        """
        # Cold start
        if user_id not in self.user_item_time_dict:
            return [(item, -i) for i, item in enumerate(self.item_topk_click[:topk])]

        if user_id not in self.u2u_sim:
            return [(item, -i) for i, item in enumerate(self.item_topk_click[:topk])]

        user_item_time_list = self.user_item_time_dict[user_id]
        user_hist_items = set([item_id for item_id, _ in user_item_time_list])

        # Get similar users
        sim_user_topk = self.config.usercf_sim_user_topk
        similar_users = sorted(
            self.u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True
        )[:sim_user_topk]

        # Aggregate scores from similar users
        items_rank = {}
        for sim_u, wuv in similar_users:
            if sim_u not in self.user_item_time_dict:
                continue

            for i, click_time in self.user_item_time_dict[sim_u]:
                if i in user_hist_items:
                    continue

                items_rank.setdefault(i, 0)

                # Calculate weights
                loc_weight = 1.0
                content_weight = 1.0
                created_time_weight = 1.0

                # Interaction with user's history
                for loc, (j, _) in enumerate(user_item_time_list):
                    # Position weight
                    loc_weight += self.weight_calc.position_weight(
                        len(user_item_time_list) - loc, beta=self.config.loc_beta
                    )

                    # Content similarity weight
                    if self.emb_i2i_sim:
                        if i in self.emb_i2i_sim and j in self.emb_i2i_sim[i]:
                            content_weight += self.emb_i2i_sim[i][j]
                        if j in self.emb_i2i_sim and i in self.emb_i2i_sim[j]:
                            content_weight += self.emb_i2i_sim[j][i]

                    # Created time weight
                    created_time_weight += self.weight_calc.time_decay_weight(
                        self.item_created_time_dict[i],
                        self.item_created_time_dict[j],
                        alpha=0.8,
                    )

                items_rank[i] += loc_weight * content_weight * created_time_weight * wuv

        # Fill with popular items if not enough
        if len(items_rank) < topk:
            for i, item in enumerate(self.item_topk_click):
                if item in items_rank or item in user_hist_items:
                    continue
                items_rank[item] = -i - 100
                if len(items_rank) == topk:
                    break

        # Sort and return top-k
        items_rank_sorted = sorted(
            items_rank.items(), key=lambda x: x[1], reverse=True
        )[:topk]

        return items_rank_sorted
