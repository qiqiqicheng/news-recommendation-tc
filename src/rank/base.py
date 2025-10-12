from abc import ABC, abstractmethod
import os
import pandas as pd
from typing import Optional, Dict, List

from src.utils.config import RecallConfig, RankConfig
from src.utils.persistence import PersistenceManager


class BaseRanker(ABC):
    def __init__(self, config: RankConfig) -> None:
        self.config = config

    def load(self, load_din_specific: bool = True):
        """
        Load training and testing data along with feature metadata.

        Args:
            load_din_specific: Whether to load DIN-specific feature groups and article info
        """
        # load train and test sets
        if os.path.exists(self.config.train_set_path):
            self.train_set = pd.read_csv(self.config.train_set_path)
            print(f"Loaded training set: {self.train_set.shape}")
        else:
            raise FileNotFoundError(
                f"Training set not found at {self.config.train_set_path}"
            )

        if os.path.exists(self.config.test_set_path):
            self.test_set = pd.read_csv(self.config.test_set_path)
            print(f"Loaded test set: {self.test_set.shape}")
        else:
            raise FileNotFoundError(
                f"Test set not found at {self.config.test_set_path}"
            )

        # load history dicts
        if os.path.exists(self.config.train_history_dict_path):
            self.train_history_dict = PersistenceManager.load_pickle(
                self.config.train_history_dict_path
            )
            print(f"Loaded train history dict: {len(self.train_history_dict)} users")
        else:
            raise FileNotFoundError(
                f"Train history dict not found at {self.config.train_history_dict_path}"
            )

        if os.path.exists(self.config.test_history_dict_path):
            self.test_history_dict = PersistenceManager.load_pickle(
                self.config.test_history_dict_path
            )
            print(f"Loaded test history dict: {len(self.test_history_dict)} users")
        else:
            raise FileNotFoundError(
                f"Test history dict not found at {self.config.test_history_dict_path}"
            )

        # load DIN-specific feature groups if requested
        if load_din_specific:
            self._load_din_specific_data()

    def _load_din_specific_data(self):
        """
        Load DIN-specific feature groups and article information.
        """
        # load feature group lists
        if os.path.exists(self.config.user_profile_features_path):
            self.user_profile_features = PersistenceManager.load_pickle(
                self.config.user_profile_features_path
            )
            print(f"Loaded user profile features: {self.user_profile_features}")
        else:
            print(
                f"Warning: User profile features not found at {self.config.user_profile_features_path}"
            )
            self.user_profile_features = []

        if os.path.exists(self.config.item_features_path):
            self.item_features = PersistenceManager.load_pickle(
                self.config.item_features_path
            )
            print(f"Loaded item features: {self.item_features}")
        else:
            print(
                f"Warning: Item features not found at {self.config.item_features_path}"
            )
            self.item_features = []

        if os.path.exists(self.config.context_features_path):
            self.context_features = PersistenceManager.load_pickle(
                self.config.context_features_path
            )
            print(f"Loaded context features: {self.context_features}")
        else:
            print(
                f"Warning: Context features not found at {self.config.context_features_path}"
            )
            self.context_features = []

        # load article info dict
        if os.path.exists(self.config.article_info_dict_path):
            self.article_info_dict = PersistenceManager.load_pickle(
                self.config.article_info_dict_path
            )
            print(f"Loaded article info dict: {len(self.article_info_dict)} articles")
        else:
            print(
                f"Warning: Article info dict not found at {self.config.article_info_dict_path}"
            )
            self.article_info_dict = {}

    def train(self):
        pass

    def predict(self):
        pass
