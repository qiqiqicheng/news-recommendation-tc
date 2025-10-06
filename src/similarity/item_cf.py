import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from .base import BaseSimilarityCalculator
from ..utils.weights import WeightCalculator
from ..data.extractors import UserFeatureExtractor


class ItemCFSimilarity(BaseSimilarityCalculator):
    def __init__(self, config):
        super().__init__(config)
        self.weight_calc = WeightCalculator()

    def calculate(self, click_df, item_created_time_dict: Dict[int, float]) -> Dict:
        """
        based on ItemCF similarity calculation, we have:
        1. position weight (click order)
        2. time weight (click time difference)
        3. created time weight (item created time difference)
        4. popularity penalty (log smoothing)

        Args:
            click_df
            item_created_time_dict: {item_id: created_time, ...}

        Returns:
            {item_i: {item_j: similarity, ...}, ...}
        """
        print("Calculating ItemCF similarity matrix...")

        # 获取用户-物品-时间字典
        user_item_time_dict = UserFeatureExtractor.get_user_item_time_dict(click_df)

        # 初始化相似度矩阵和物品计数
        i2i_sim = {}
        item_cnt = defaultdict(int)

        # 遍历每个用户的点击序列
        for user, item_time_list in tqdm(
            user_item_time_dict.items(), desc="Computing item similarity"
        ):
            for loc1, (i, i_click_time) in enumerate(item_time_list):
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})

                # 与该用户点击的其他物品计算相似度
                for loc2, (j, j_click_time) in enumerate(item_time_list):
                    if i == j:
                        continue

                    # 1. 位置权重 (考虑正向和反向点击)
                    loc_alpha = (
                        self.config.loc_alpha
                        if loc2 > loc1
                        else self.config.loc_alpha_reverse
                    )
                    loc_weight = loc_alpha * (
                        self.config.loc_beta ** (np.abs(loc2 - loc1) - 1)
                    )

                    # 2. 点击时间权重
                    click_time_weight = self.weight_calc.time_decay_weight(
                        i_click_time, j_click_time, alpha=self.config.time_decay_alpha
                    )

                    # 3. 创建时间权重
                    created_time_weight = self.weight_calc.time_decay_weight(
                        item_created_time_dict[i],
                        item_created_time_dict[j],
                        alpha=self.config.created_time_alpha,
                    )

                    # 4. 用户活跃度惩罚 (防止活跃用户主导相似度)
                    user_penalty = 1.0 / self.weight_calc.log_penalty(
                        len(item_time_list)
                    )

                    # 累加相似度
                    i2i_sim[i].setdefault(j, 0)
                    i2i_sim[i][j] += (
                        loc_weight
                        * click_time_weight
                        * created_time_weight
                        * user_penalty
                    )

        # 归一化相似度 (使用余弦相似度的分母)
        i2i_sim_normalized = i2i_sim.copy()
        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                i2i_sim_normalized[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

        self.similarity_matrix = i2i_sim_normalized

        print(f"ItemCF similarity matrix calculated. Size: {len(i2i_sim_normalized)}")
        return i2i_sim_normalized

    def get_similar_items(
        self, item_id: int, topk: int = 20
    ) -> List[Tuple[int, float]]:
        """
        Args:
            item_id
            topk

        Returns:
            [(item_id, similarity), ...]
        """
        if not self.is_calculated():
            raise ValueError(
                "Similarity matrix not calculated. Call calculate() first."
            )

        if item_id not in self.similarity_matrix:
            return []

        similar_items = sorted(
            self.similarity_matrix[item_id].items(), key=lambda x: x[1], reverse=True
        )[:topk]

        return similar_items
