import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from tqdm import tqdm

from ..utils.persistence import PersistenceManager
from ..utils.config import RecallConfig


class RecallFusion:
    """
    Multi-recall fusion with multiple normalization and fusion strategies

    Supports:
    - Multiple normalization methods: local, global, z-score
    - Multiple fusion strategies: weighted_sum, weighted_avg, max_score, rrf, diversity_weighted
    - User history filtering to remove seen items
    """

    def __init__(
        self,
        config: RecallConfig,
        fusion_strategy: Literal[
            "weighted_sum",
            "weighted_avg",
            "max_score",
            "harmonic_mean",
            "diversity_weighted",
            "rrf",
        ] = "weighted_avg",
        normalize_method: Literal["local", "global", "z-score"] = "local",
    ):
        """
        Initialize recall fusion

        Args:
            config: Configuration object
            fusion_strategy: Strategy for merging scores
                - weighted_sum: Simple weighted sum (may favor items from multiple sources)
                - weighted_avg: Weighted average (more stable, recommended)
                - max_score: Take maximum weighted score (conservative)
                - harmonic_mean: Harmonic mean (penalizes extreme values)
                - diversity_weighted: Bonus for items from multiple sources
                - rrf: Reciprocal Rank Fusion (rank-based, robust)
            normalize_method: Method for score normalization
                - local: Per-user per-method normalization (default)
                - global: Global min-max across all scores
                - z-score: Standard score normalization
        """
        self.config = config
        self.recall_results = {}
        self.weights = {}
        self.fusion_strategy = fusion_strategy
        self.normalize_method = normalize_method

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
        self, item_list: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Normalize scores to [0, 1] range using min-max normalization

        Fixed version: does not assume input is sorted, handles edge cases properly

        Args:
            item_list: List of (item_id, score) tuples (can be unsorted)

        Returns:
            Normalized list of (item_id, score) tuples
        """
        # Handle edge cases
        if len(item_list) == 0:
            return []

        if len(item_list) == 1:
            return [(item_list[0][0], 1.0)]  # Single item gets max score

        # Extract scores and find min/max (no sorting assumption)
        scores = [score for _, score in item_list]
        min_sim = min(scores)
        max_sim = max(scores)

        norm_item_list = []
        for item, score in item_list:
            if max_sim > min_sim:
                norm_score = (score - min_sim) / (max_sim - min_sim)
            else:
                # All scores are the same, give equal weight
                norm_score = 1.0
            norm_item_list.append((item, norm_score))

        return norm_item_list

    def _global_normalize(self) -> Dict[str, Dict[int, List[Tuple[int, float]]]]:
        """
        Global normalization: find global min/max across all scores, then normalize

        This ensures all recall sources have the same score scale

        Returns:
            Normalized recall results for all methods
        """
        # Collect all scores
        all_scores = []
        for method, user_recall_items in self.recall_results.items():
            for user_id, item_list in user_recall_items.items():
                all_scores.extend([score for _, score in item_list])

        if not all_scores:
            return {}

        global_min = min(all_scores)
        global_max = max(all_scores)

        # Normalize using global min/max
        normalized_results = {}
        for method, user_recall_items in self.recall_results.items():
            normalized_results[method] = {}
            for user_id, item_list in user_recall_items.items():
                norm_list = []
                for item, score in item_list:
                    if global_max > global_min:
                        norm_score = (score - global_min) / (global_max - global_min)
                    else:
                        norm_score = 1.0
                    norm_list.append((item, norm_score))
                normalized_results[method][user_id] = norm_list

        return normalized_results

    def _zscore_normalize(self) -> Dict[str, Dict[int, List[Tuple[int, float]]]]:
        """
        Z-score normalization: (score - mean) / std

        Returns:
            Normalized recall results for all methods
        """
        normalized_results = {}

        for method, user_recall_items in self.recall_results.items():
            # Calculate global mean and std for this method
            all_scores = []
            for user_id, item_list in user_recall_items.items():
                all_scores.extend([score for _, score in item_list])

            if not all_scores:
                normalized_results[method] = {}
                continue

            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)

            normalized_results[method] = {}
            for user_id, item_list in user_recall_items.items():
                norm_list = []
                for item, score in item_list:
                    if std_score > 0:
                        # Z-score, then map to [0, 1] using sigmoid
                        z_score = (score - mean_score) / std_score
                        norm_score = 1.0 / (1.0 + np.exp(-z_score))
                    else:
                        norm_score = 0.5
                    norm_list.append((item, norm_score))
                normalized_results[method][user_id] = norm_list

        return normalized_results

    def _weighted_merge(
        self,
        normalized_results: Dict[str, Dict[int, List[Tuple[int, float]]]],
        user_id: int,
    ) -> Dict[int, float]:
        """
        Merge multiple recall results with configurable fusion strategy

        Args:
            normalized_results: Normalized recall results from all methods
            user_id: User ID to merge results for

        Returns:
            Merged item scores dictionary
        """
        merged_scores = {}
        item_sources = {}  # Track which sources each item comes from

        # Collect all items and their sources
        for method, user_recall_items in normalized_results.items():
            if user_id not in user_recall_items:
                continue

            recall_weight = self.weights.get(method, 1.0)

            for rank, (item, score) in enumerate(user_recall_items[user_id]):
                if item not in item_sources:
                    item_sources[item] = []
                item_sources[item].append(
                    {
                        "method": method,
                        "score": score,
                        "weight": recall_weight,
                        "rank": rank,
                    }
                )

        # Apply fusion strategy
        for item, sources in item_sources.items():
            if self.fusion_strategy == "weighted_sum":
                # Strategy 1: Weighted sum (may favor multi-source items)
                merged_scores[item] = sum(s["weight"] * s["score"] for s in sources)

            elif self.fusion_strategy == "weighted_avg":
                # Strategy 2: Weighted average (more stable, recommended)
                total_weight = sum(s["weight"] for s in sources)
                weighted_sum = sum(s["weight"] * s["score"] for s in sources)
                merged_scores[item] = (
                    weighted_sum / total_weight if total_weight > 0 else 0
                )

            elif self.fusion_strategy == "max_score":
                # Strategy 3: Maximum weighted score (conservative)
                merged_scores[item] = max(s["weight"] * s["score"] for s in sources)

            elif self.fusion_strategy == "harmonic_mean":
                # Strategy 4: Harmonic mean (penalizes extreme values)
                weighted_scores = [s["weight"] * s["score"] for s in sources]
                n = len(weighted_scores)
                harmonic = n / sum(1.0 / (s + 1e-8) for s in weighted_scores)
                merged_scores[item] = harmonic

            elif self.fusion_strategy == "diversity_weighted":
                # Strategy 5: Diversity bonus (items from multiple sources get bonus)
                base_score = sum(s["weight"] * s["score"] for s in sources)
                diversity_bonus = len(sources) * 0.1  # 10% bonus per additional source
                merged_scores[item] = base_score * (1 + diversity_bonus)

            elif self.fusion_strategy == "rrf":
                # Strategy 6: Reciprocal Rank Fusion (rank-based, robust)
                # RRF score = sum(weight / (k + rank)) for each source
                k = 60  # Standard RRF parameter
                rrf_score = sum(s["weight"] / (k + s["rank"]) for s in sources)
                merged_scores[item] = rrf_score

            else:
                # Default to weighted average
                total_weight = sum(s["weight"] for s in sources)
                weighted_sum = sum(s["weight"] * s["score"] for s in sources)
                merged_scores[item] = (
                    weighted_sum / total_weight if total_weight > 0 else 0
                )

        return merged_scores

    def fuse(
        self,
        topk: int = 30,
        user_history: Optional[Dict[int, set]] = None,
        remove_seen: bool = True,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Fuse multiple recall results with configurable normalization and fusion strategies

        Args:
            topk: Number of items to return per user
            user_history: Optional dictionary of {user_id: set(seen_item_ids)}
            remove_seen: Whether to filter out items in user_history

        Returns:
            Fused recall results {user_id: [(item_id, score), ...]}
        """
        if not self.recall_results:
            raise ValueError("No recall results added. Use add_recall_result() first.")

        print(f"Fusing multiple recall results with strategy: {self.fusion_strategy}")
        print(f"Normalization method: {self.normalize_method}")

        # Step 1: Normalize scores based on selected method
        if self.normalize_method == "global":
            print("Performing global normalization...")
            normalized_results = self._global_normalize()
        elif self.normalize_method == "z-score":
            print("Performing z-score normalization...")
            normalized_results = self._zscore_normalize()
        else:  # local normalization
            print("Performing local normalization...")
            normalized_results = {}
            for method, user_recall_items in tqdm(
                self.recall_results.items(), desc="Normalizing recall results"
            ):
                normalized_results[method] = {}
                for user_id, item_list in user_recall_items.items():
                    normalized_results[method][user_id] = self._normalize_scores(
                        item_list
                    )

        # Step 2: Get all users
        all_users = set()
        for method_results in normalized_results.values():
            all_users.update(method_results.keys())

        # Step 3: Weighted fusion with optional history filtering
        print(f"Performing weighted fusion for {len(all_users)} users...")
        fused_results = {}

        for user_id in tqdm(all_users, desc="Fusing results"):
            merged_scores = self._weighted_merge(normalized_results, user_id)

            # Filter out seen items if requested
            if remove_seen and user_history and user_id in user_history:
                seen_items = user_history[user_id]
                merged_scores = {
                    item: score
                    for item, score in merged_scores.items()
                    if item not in seen_items
                }

            # Sort and keep top-k
            sorted_items = sorted(
                merged_scores.items(), key=lambda x: x[1], reverse=True
            )[:topk]

            fused_results[user_id] = sorted_items

        print(f"Fusion completed! {len(fused_results)} users with fused results.")

        # Store for save method
        self.fused_results = fused_results

        return fused_results

    def save(
        self,
        filename: str,
        results: Optional[Dict[int, List[Tuple[int, float]]]] = None,
    ):
        """
        Save fused results to file

        Args:
            filename: Filename to save to
            results: Optional results to save. If None, uses self.fused_results from last fuse() call
        """
        results_to_save = (
            results if results is not None else getattr(self, "fused_results", None)
        )

        if not results_to_save:
            raise ValueError(
                "No results to save. Either pass results or call fuse() first."
            )

        filepath = os.path.join(self.config.save_path, filename)
        PersistenceManager.save_pickle(results_to_save, filepath)
        print(f"Fused results saved to: {filepath}")

    def get_statistics(self) -> Dict:
        """
        Get comprehensive fusion statistics

        Returns:
            Dictionary with detailed statistics
        """
        stats = {
            "num_methods": len(self.recall_results),
            "methods": list(self.recall_results.keys()),
            "weights": self.weights,
            "fusion_strategy": self.fusion_strategy,
            "normalize_method": self.normalize_method,
        }

        # Per-method statistics
        for method, results in self.recall_results.items():
            stats[f"{method}_users"] = len(results)
            stats[f"{method}_avg_items"] = (
                sum(len(items) for items in results.values()) / len(results)
                if results
                else 0
            )

            # Score statistics
            all_scores = [
                score for user_items in results.values() for _, score in user_items
            ]
            if all_scores:
                stats[f"{method}_score_mean"] = np.mean(all_scores)
                stats[f"{method}_score_std"] = np.std(all_scores)
                stats[f"{method}_score_min"] = np.min(all_scores)
                stats[f"{method}_score_max"] = np.max(all_scores)

        # Item coverage statistics
        all_items = set()
        for results in self.recall_results.values():
            for user_items in results.values():
                all_items.update(item for item, _ in user_items)
        stats["total_unique_items"] = len(all_items)

        # User coverage
        all_users = set()
        for results in self.recall_results.values():
            all_users.update(results.keys())
        stats["total_unique_users"] = len(all_users)

        return stats


class RecallEnsemble:
    """
    Ensemble multiple recall methods for online/real-time recall

    Difference from RecallFusion:
    - RecallFusion: For offline batch fusion with pre-computed results
    - RecallEnsemble: For online real-time recall with recaller instances
    """

    def __init__(
        self,
        config: RecallConfig,
        fusion_strategy: Literal[
            "weighted_avg", "weighted_sum", "max_score"
        ] = "weighted_avg",
    ):
        """
        Initialize recall ensemble

        Args:
            config: Configuration object
            fusion_strategy: Strategy for merging scores (simpler than RecallFusion for real-time)
        """
        self.config = config
        self.recallers = {}
        self.fusion_strategy = fusion_strategy

    def add_recaller(self, name: str, recaller, weight: float = 1.0):
        """
        Add a recaller instance to the ensemble

        Args:
            name: Name of the recaller
            recaller: Recaller instance with recall() method
            weight: Weight for this recaller
        """
        self.recallers[name] = {"recaller": recaller, "weight": weight}
        print(f"Added recaller: {name} with weight {weight}")

    def recall(self, user_id: int, topk: int = 10) -> List[Tuple[int, float]]:
        """
        Ensemble recall for a single user (real-time)

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
                results = recaller.recall(
                    user_id, topk=topk * 2
                )  # Get more for better fusion
                all_results.append((name, results, weight))
            except Exception as e:
                print(f"Warning: {name} recall failed for user {user_id}: {e}")
                continue

        if not all_results:
            return []

        # Merge results with normalization
        item_scores = {}
        item_sources = {}

        for name, results, weight in all_results:
            if len(results) == 0:
                continue

            # Normalize scores for this recaller
            scores = [score for _, score in results]
            max_score = max(scores)
            min_score = min(scores)

            for rank, (item, score) in enumerate(results):
                # Normalize score
                if max_score > min_score:
                    norm_score = (score - min_score) / (max_score - min_score)
                else:
                    norm_score = 1.0

                if item not in item_sources:
                    item_sources[item] = []
                    item_scores[item] = 0

                item_sources[item].append(
                    {
                        "method": name,
                        "score": norm_score,
                        "weight": weight,
                        "rank": rank,
                    }
                )

        # Apply fusion strategy
        for item, sources in item_sources.items():
            if self.fusion_strategy == "weighted_sum":
                item_scores[item] = sum(s["weight"] * s["score"] for s in sources)
            elif self.fusion_strategy == "weighted_avg":
                total_weight = sum(s["weight"] for s in sources)
                item_scores[item] = (
                    sum(s["weight"] * s["score"] for s in sources) / total_weight
                )
            elif self.fusion_strategy == "max_score":
                item_scores[item] = max(s["weight"] * s["score"] for s in sources)

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
