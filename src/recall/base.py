from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from ..utils.config import RecallConfig


class BaseRecaller(ABC):
    def __init__(self, config: RecallConfig):
        self.config = config

    @abstractmethod
    def recall(self, user_id: int, topk: int = 10) -> List[Tuple[int, float]]:
        """
        Recall top-k items for a single user

        Args:
            user_id
            topk

        Returns:
            List of (item_id, score) tuples
        """
        pass

    def batch_recall(
        self, user_ids: List[int], topk: int = 10
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Recall top-k items for multiple users

        Args:
            user_ids
            topk

        Returns:
            Dictionary mapping user_id to list of (item_id, score) tuples
        """
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recall(user_id, topk)
        return results

    def _filter_history(
        self, candidates: List[Tuple[int, float]], history_items: set
    ) -> List[Tuple[int, float]]:
        """
        Filter out items that user has already interacted with

        Args:
            candidates: List of (item_id, score) tuples
            history_items: Set of items user has already seen

        Returns:
            Filtered list of (item_id, score) tuples
        """
        return [
            (item, score) for item, score in candidates if item not in history_items
        ]
