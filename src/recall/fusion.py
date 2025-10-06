"""
Multi-recall Fusion Module
Combines multiple recall strategies with weighted fusion
"""

import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from ..utils.persistence import PersistenceManager
from ..utils.config import RecallConfig


class RecallFusion:
    """Multi-recall fusion with normalization and weighted combination"""

    def __init__(self, config: RecallConfig):
        """
        Initialize recall fusion

        Args:
            config: Configuration object
        """
        self.config = config
        self.recall_results = {}
        self.weights = {}

    def add_recall_result(
        self, name: str, result: Dict[int, List[Tuple[int, float]]], weight: float = 1.0
    ):
        """
        Add a recall result with its weight

        Args:
            name: Name of the recall method
            result: Recall results dictionary {user_id: [(item_id, score), ...]}
            weight: Weight for this recall method
        """
        self.recall_results[name] = result
        self.weights[name] = weight
        print(f"Added recall result: {name} with weight {weight}")

    def _normalize_scores(
        self, sorted_item_list: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Normalize scores to [0, 1] range using min-max normalization

        Args:
            sorted_item_list: List of (item_id, score) tuples

        Returns:
            Normalized list of (item_id, score) tuples
        """
        # Handle edge cases
        if len(sorted_item_list) < 2:
            return sorted_item_list

        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]

        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = (
                    1.0 * (score - min_sim) / (max_sim - min_sim)
                    if max_sim > min_sim
                    else 1.0
                )
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))

        return norm_sorted_item_list

    def _weighted_merge(
        self,
        normalized_results: Dict[str, Dict[int, List[Tuple[int, float]]]],
        user_id: int,
    ) -> Dict[int, float]:
        """
        Merge multiple recall results with weighted fusion

        Args:
            normalized_results: Normalized recall results from all methods
            user_id: User ID to merge results for

        Returns:
            Merged item scores dictionary
        """
        merged_scores = {}

        for method, user_recall_items in normalized_results.items():
            if user_id not in user_recall_items:
                continue

            recall_weight = self.weights.get(method, 1.0)

            for item, score in user_recall_items[user_id]:
                merged_scores.setdefault(item, 0)
                merged_scores[item] += recall_weight * score

        return merged_scores

    def fuse(self, topk: int = 150) -> Dict[int, List[Tuple[int, float]]]:
        """
        Fuse multiple recall results

        Args:
            topk: Number of items to return per user

        Returns:
            Fused recall results {user_id: [(item_id, score), ...]}
        """
        if not self.recall_results:
            raise ValueError("No recall results added. Use add_recall_result() first.")

        print("Fusing multiple recall results...")

        # Step 1: Normalize each recall method's scores per user
        normalized_results = {}

        for method, user_recall_items in tqdm(
            self.recall_results.items(), desc="Normalizing recall results"
        ):
            print(f"Normalizing {method}...")
            normalized_results[method] = {}

            for user_id, sorted_item_list in user_recall_items.items():
                normalized_results[method][user_id] = self._normalize_scores(
                    sorted_item_list
                )

        # Step 2: Get all users
        all_users = set()
        for method_results in normalized_results.values():
            all_users.update(method_results.keys())

        # Step 3: Weighted fusion
        print("Performing weighted fusion...")
        fused_results = {}

        for user_id in tqdm(all_users, desc="Fusing results"):
            merged_scores = self._weighted_merge(normalized_results, user_id)

            # Sort and keep top-k
            sorted_items = sorted(
                merged_scores.items(), key=lambda x: x[1], reverse=True
            )[:topk]

            fused_results[user_id] = sorted_items

        print(f"Fusion completed! {len(fused_results)} users with fused results.")

        return fused_results

    def save(self, filename: str):
        """
        Save fused results to file

        Args:
            filename: Filename to save to
        """
        if not hasattr(self, "fused_results") or not self.fused_results:
            raise ValueError("No fused results to save. Call fuse() first.")

        filepath = os.path.join(self.config.save_path, filename)
        PersistenceManager.save_pickle(self.fused_results, filepath)
        print(f"Fused results saved to: {filepath}")

    def get_statistics(self) -> Dict:
        """
        Get fusion statistics

        Returns:
            Dictionary with statistics
        """
        stats = {
            "num_methods": len(self.recall_results),
            "methods": list(self.recall_results.keys()),
            "weights": self.weights,
        }

        # Per-method statistics
        for method, results in self.recall_results.items():
            stats[f"{method}_users"] = len(results)
            stats[f"{method}_avg_items"] = (
                sum(len(items) for items in results.values()) / len(results)
                if results
                else 0
            )

        return stats


class RecallEnsemble:
    """Ensemble multiple recall methods with advanced fusion strategies"""

    def __init__(self, config: RecallConfig):
        """
        Initialize recall ensemble

        Args:
            config: Configuration object
        """
        self.config = config
        self.recallers = {}

    def add_recaller(self, name: str, recaller, weight: float = 1.0):
        """
        Add a recaller to the ensemble

        Args:
            name: Name of the recaller
            recaller: Recaller instance
            weight: Weight for this recaller
        """
        self.recallers[name] = {"recaller": recaller, "weight": weight}
        print(f"Added recaller: {name} with weight {weight}")

    def recall(self, user_id: int, topk: int = 10) -> List[Tuple[int, float]]:
        """
        Ensemble recall for a single user

        Args:
            user_id: User ID
            topk: Number of items to recall

        Returns:
            List of (item_id, score) tuples
        """
        # Collect results from all recallers
        all_results = []

        for name, recaller_info in self.recallers.items():
            recaller = recaller_info["recaller"]
            weight = recaller_info["weight"]

            try:
                results = recaller.recall(user_id, topk=topk)
                all_results.append((name, results, weight))
            except Exception as e:
                print(f"Warning: {name} recall failed for user {user_id}: {e}")
                continue

        if not all_results:
            return []

        # Merge results
        item_scores = {}

        for name, results, weight in all_results:
            # Normalize scores
            if len(results) > 0:
                max_score = max(score for _, score in results)
                min_score = min(score for _, score in results)

                for item, score in results:
                    norm_score = (
                        (score - min_score) / (max_score - min_score)
                        if max_score > min_score
                        else 1.0
                    )
                    item_scores.setdefault(item, 0)
                    item_scores[item] += weight * norm_score

        # Sort and return top-k
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[
            :topk
        ]

        return sorted_items

    def batch_recall(
        self, user_ids: List[int], topk: int = 10
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Batch recall for multiple users

        Args:
            user_ids: List of user IDs
            topk: Number of items to recall

        Returns:
            Dictionary mapping user_id to list of (item_id, score) tuples
        """
        results = {}
        for user_id in tqdm(user_ids, desc="Ensemble batch recall"):
            results[user_id] = self.recall(user_id, topk)
        return results
