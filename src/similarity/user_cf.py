import math
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from .base import BaseSimilarityCalculator
from ..utils.weights import WeightCalculator
from ..data.extractors import InteractionFeatureExtractor, UserFeatureExtractor


class UserCFSimilarity(BaseSimilarityCalculator):
    def __init__(self, config):
        super().__init__(config)
        self.weight_calc = WeightCalculator()

    def calculate(self, click_df, user_activate_degree_dict: Dict[int, float]) -> Dict:
        """
        based on the joint clicked users of items, we have:
        1. user activate weight
        2. item popularity penalty

        Args:
            click_df
            user_activate_degree_dict: {user_id: activate_degree, ...}

        Returns:
            {user_i: {user_j: similarity, ...}, ...}
        """
        print("Calculating UserCF similarity matrix...")

        item_user_time_dict = InteractionFeatureExtractor.get_item_user_time_dict(
            click_df
        )

        u2u_sim = {}
        user_cnt = defaultdict(int)

        for item, user_time_list in tqdm(
            item_user_time_dict.items(), desc="Computing user similarity"
        ):
            for u, click_time in user_time_list:
                user_cnt[u] += 1
                u2u_sim.setdefault(u, {})

                for v, click_time in user_time_list:
                    if u == v:
                        continue

                    u2u_sim[u].setdefault(v, 0)

                    activate_weight = self.weight_calc.activation_weight(
                        user_activate_degree_dict[u] + user_activate_degree_dict[v]
                    )

                    item_penalty = 1.0 / self.weight_calc.log_penalty(
                        len(user_time_list)
                    )

                    u2u_sim[u][v] += activate_weight * item_penalty

        u2u_sim_normalized = u2u_sim.copy()
        for u, related_users in u2u_sim.items():
            for v, wuv in related_users.items():
                u2u_sim_normalized[u][v] = wuv / math.sqrt(user_cnt[u] * user_cnt[v])

        self.similarity_matrix = u2u_sim_normalized

        print(f"UserCF similarity matrix calculated. Size: {len(u2u_sim_normalized)}")
        return u2u_sim_normalized

    def get_similar_users(
        self, user_id: int, topk: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Args:
            user_id
            topk

        Returns:
            [(user_id, similarity), ...]
        """
        if not self.is_calculated():
            raise ValueError(
                "Similarity matrix not calculated. Call calculate() first."
            )

        if user_id not in self.similarity_matrix:
            return []

        similar_users = sorted(
            self.similarity_matrix[user_id].items(), key=lambda x: x[1], reverse=True
        )[:topk]

        return similar_users
