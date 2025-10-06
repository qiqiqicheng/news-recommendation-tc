"""
Cold Start Recaller with rule-based filtering
"""

from datetime import datetime
from typing import Dict, List, Tuple, Set
from tqdm import tqdm

from .base import BaseRecaller


class ColdStartRecaller(BaseRecaller):
    """Cold start recaller using rule-based filtering"""

    def __init__(
        self,
        config,
        base_recall_results: Dict[int, List[Tuple[int, float]]],
        user_hist_item_types_dict: Dict[int, Set],
        user_hist_item_words_dict: Dict[int, float],
        user_last_item_created_time_dict: Dict[int, float],
        item_type_dict: Dict[int, int],
        item_words_dict: Dict[int, int],
        item_created_time_dict: Dict[int, float],
        click_article_ids_set: Set[int],
    ):
        """
        Initialize cold start recaller

        Args:
            config: Configuration object
            base_recall_results: Base recall results from other methods
            user_hist_item_types_dict: User's historical item types
            user_hist_item_words_dict: User's average item word count
            user_last_item_created_time_dict: User's last clicked item creation time
            item_type_dict: Item type mapping
            item_words_dict: Item word count mapping
            item_created_time_dict: Item creation time mapping
            click_article_ids_set: Set of all clicked items in the log
        """
        super().__init__(config)
        self.base_recall_results = base_recall_results
        self.user_hist_item_types_dict = user_hist_item_types_dict
        self.user_hist_item_words_dict = user_hist_item_words_dict
        self.user_last_item_created_time_dict = user_last_item_created_time_dict
        self.item_type_dict = item_type_dict
        self.item_words_dict = item_words_dict
        self.item_created_time_dict = item_created_time_dict
        self.click_article_ids_set = click_article_ids_set

        # Apply filtering
        self.filtered_results = self._apply_filtering()

    def _apply_filtering(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Apply rule-based filtering to base recall results

        Filtering rules:
        1. Item type must match user's historical preferences
        2. Item word count should be similar to user's historical average
        3. Item creation time should be recent (within 90 days)
        4. Item should not appear in click logs (truly cold start)

        Returns:
            Filtered recall results
        """
        print("Applying cold start filtering rules...")

        filtered_results = {}

        for user, item_list in tqdm(
            self.base_recall_results.items(), desc="Filtering cold start items"
        ):
            if user not in self.user_hist_item_types_dict:
                continue

            filtered_items = []

            # Get user profile
            hist_item_type_set = self.user_hist_item_types_dict[user]
            hist_mean_words = self.user_hist_item_words_dict[user]
            hist_last_item_created_time = self.user_last_item_created_time_dict[user]

            for item, score in item_list:
                # Check if item exists in dictionaries
                if item not in self.item_type_dict:
                    continue
                if item not in self.item_words_dict:
                    continue
                if item not in self.item_created_time_dict:
                    continue

                # Get item info
                curr_item_type = self.item_type_dict[item]
                curr_item_words = self.item_words_dict[item]
                curr_item_created_time = self.item_created_time_dict[item]

                # Apply filtering rules
                # Rule 1: Type matching
                if curr_item_type not in hist_item_type_set:
                    continue

                # Rule 2: Already clicked (not cold start)
                if item in self.click_article_ids_set:
                    continue

                # Rule 3: Word count similarity (within 200 words)
                if abs(curr_item_words - hist_mean_words) > 200:
                    continue

                # Rule 4: Time recency (within 90 days)
                # Note: timestamps are normalized, so we use a threshold
                time_diff = abs(curr_item_created_time - hist_last_item_created_time)
                if time_diff > 0.25:  # Adjusted threshold for normalized time
                    continue

                filtered_items.append((item, score))

            if filtered_items:
                filtered_results[user] = filtered_items

        print(
            f"Cold start filtering completed. {len(filtered_results)} users have cold start items."
        )

        return filtered_results

    def recall(self, user_id: int, topk: int = 10) -> List[Tuple[int, float]]:
        """
        Recall top-k cold start items for a user

        Args:
            user_id: User ID
            topk: Number of items to recall

        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.filtered_results:
            return []

        # Sort by score and return top-k
        items = sorted(
            self.filtered_results[user_id], key=lambda x: x[1], reverse=True
        )[:topk]

        return items

    def get_statistics(self) -> Dict:
        """
        Get cold start recall statistics

        Returns:
            Dictionary with statistics
        """
        total_users = len(self.base_recall_results)
        cold_start_users = len(self.filtered_results)

        total_items = sum(len(items) for items in self.base_recall_results.values())
        cold_start_items = sum(len(items) for items in self.filtered_results.values())

        return {
            "total_users": total_users,
            "cold_start_users": cold_start_users,
            "cold_start_user_ratio": (
                cold_start_users / total_users if total_users > 0 else 0
            ),
            "total_items_before_filtering": total_items,
            "total_items_after_filtering": cold_start_items,
            "filtering_ratio": cold_start_items / total_items if total_items > 0 else 0,
        }
