import numpy as np
import faiss
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from .base import BaseSimilarityCalculator


class EmbeddingSimilarity(BaseSimilarityCalculator):
    def __init__(self, config):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim

    def calculate(self, item_emb_df) -> Dict:
        """
        using Faiss to calculate item-item similarity based on embeddings

        Args:
            item_emb_df
            topk

        Returns:
            {item_i: {item_j: similarity, ...}, ...}
        """
        print("Calculating Embedding similarity matrix with Faiss...")

        topk = self.config.embedding_topk

        # 重置索引以确保是连续的0, 1, 2, ...
        item_emb_df_reset = item_emb_df.reset_index(drop=True)
        item_idx_2_rawid_dict = dict(
            zip(item_emb_df_reset.index, item_emb_df_reset["article_id"])
        )

        # 提取embedding列
        item_emb_cols = [col for col in item_emb_df_reset.columns if "emb" in col]
        item_emb_np = np.ascontiguousarray(
            item_emb_df_reset[item_emb_cols].values, dtype=np.float32
        )

        # 向量归一化
        item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

        # 自动检测embedding维度
        if self.embedding_dim is None:
            self.embedding_dim = item_emb_np.shape[1]
            print(f"Auto-detected embedding dimension: {self.embedding_dim}")

        # 建立Faiss索引 (使用内积,因为已归一化相当于余弦相似度)
        index = faiss.IndexFlatIP(item_emb_np.shape[1])
        index.add(item_emb_np)

        # 相似度查询
        print(f"Searching top-{topk} similar items for each item...")
        sim, idx = index.search(item_emb_np, topk + 1)  # +1因为会包含自己

        # 将结果转换为字典格式
        item_sim_dict = defaultdict(dict)
        for target_idx, sim_value_list, rele_idx_list in tqdm(
            zip(range(len(item_emb_np)), sim, idx),
            total=len(item_emb_np),
            desc="Building similarity dict",
        ):
            target_raw_id = item_idx_2_rawid_dict[target_idx]

            # 从1开始是为了去掉物品本身
            for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
                rele_raw_id = item_idx_2_rawid_dict[rele_idx]
                item_sim_dict[target_raw_id][rele_raw_id] = float(sim_value)

        self.similarity_matrix = dict(item_sim_dict)

        print(f"Embedding similarity matrix calculated. Size: {len(item_sim_dict)}")
        return self.similarity_matrix

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

    def get_embedding_dimension(self):
        return self.embedding_dim
