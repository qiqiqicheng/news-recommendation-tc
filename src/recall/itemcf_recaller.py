import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from .base import BaseRecaller
from ..utils.weights import WeightCalculator
from ..data.extractors import UserFeatureExtractor


class ItemCFRecaller(BaseRecaller):
    def __init__(
        self,
        config,
        similarity_matrix: Dict,
        item_created_time_dict: Dict,
        user_item_time_dict: Dict,
        item_topk_click: List,
        emb_similarity_matrix: Dict = {},
    ):
        """
        Args:
            config: Configuration object
            similarity_matrix: Item-to-item similarity matrix
            item_created_time_dict: Dictionary of item creation times
            user_item_time_dict: Dictionary of user interaction history
            item_topk_click: List of top-k popular items for cold start
            emb_similarity_matrix: Optional embedding-based similarity matrix
        """
        super().__init__(config)
        self.i2i_sim = similarity_matrix
        self.item_created_time_dict = item_created_time_dict
        self.user_item_time_dict = user_item_time_dict
        self.item_topk_click = item_topk_click
        self.emb_i2i_sim = emb_similarity_matrix or {}
        self.weight_calc = WeightCalculator()

    def recall(self, user_id: int, topk: int = 10) -> List[Tuple[int, float]]:
        """
        Recall top-k items for a user using ItemCF

        Args:
            user_id
            topk

        Returns:
            List of (item_id, score) tuples
        """
        # Get user history
        if user_id not in self.user_item_time_dict:
            # Cold start: return popular items
            return [(item, -i) for i, item in enumerate(self.item_topk_click[:topk])]

        user_hist_items = self.user_item_time_dict[user_id]
        user_hist_items_set = {item_id for item_id, _ in user_hist_items}

        # Aggregate scores from similar items
        item_rank = {}
        sim_item_topk = self.config.itemcf_sim_item_topk

        for loc, (i, click_time) in enumerate(user_hist_items):
            if i not in self.i2i_sim:
                continue

            # Get top-k similar items
            similar_items = sorted(
                self.i2i_sim[i].items(), key=lambda x: x[1], reverse=True
            )[:sim_item_topk]

            for j, wij in similar_items:
                if j in user_hist_items_set:
                    continue

                # Calculate weights
                # 1. Created time weight
                created_time_weight = self.weight_calc.time_decay_weight(
                    self.item_created_time_dict[i],
                    self.item_created_time_dict[j],
                    alpha=self.config.created_time_alpha,
                )

                # 2. Position weight (more recent interactions have higher weight)
                loc_weight = self.weight_calc.position_weight(
                    len(user_hist_items) - loc, beta=self.config.loc_beta
                )

                # 3. Content weight (from embedding similarity)
                content_weight = 1.0
                if self.emb_i2i_sim:
                    if i in self.emb_i2i_sim and j in self.emb_i2i_sim[i]:
                        content_weight += self.emb_i2i_sim[i][j]
                    if j in self.emb_i2i_sim and i in self.emb_i2i_sim[j]:
                        content_weight += self.emb_i2i_sim[j][i]

                # Aggregate score
                item_rank.setdefault(j, 0)
                item_rank[j] += created_time_weight * loc_weight * content_weight * wij

        # Fill with popular items if not enough
        if len(item_rank) < topk:
            for i, item in enumerate(self.item_topk_click):
                if item in item_rank or item in user_hist_items_set:
                    continue
                item_rank[item] = -i - 100
                if len(item_rank) == topk:
                    break

        # Sort and return top-k
        item_rank_sorted = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[
            :topk
        ]

        return item_rank_sorted
