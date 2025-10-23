from dataclasses import dataclass, field
from typing import Optional, List
import os
from pathlib import Path


@dataclass
class RecallConfig:
    # path settings
    _project_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent.parent.resolve())
    )
    data_path: str = field(init=False)
    save_path: str = field(init=False)
    recall_path: str = field(init=False)

    # sampling and debug
    debug_mode: bool = False
    debug_sample_size: int = 10000
    debug_user_sample_size: int = 10000

    # ItemCF
    itemcf_sim_item_topk: int = 20
    itemcf_recall_num: int = 20
    itemcf_hot_topk: int = 20

    # UserCF
    usercf_sim_user_topk: int = 20
    usercf_recall_num: int = 10

    # Embedding
    embedding_topk: int = 20
    embedding_dim: int = 64

    # YoutubeDNN
    youtubednn_seq_max_len: int = 30
    youtubednn_embedding_dim: int = 16
    youtubednn_hidden_units: List[int] = field(default_factory=lambda: [64, 16])
    youtubednn_negsample: int = 4  # Number of negative samples per positive sample
    youtubednn_epochs: int = 1
    youtubednn_batch_size: int = 256
    youtubednn_learning_rate: float = 0.001
    youtubednn_topk: int = 20

    # fuse
    fuse_topk: int = 30

    # features
    neg_sample_rate: float = 0.001
    min_sample_size: int = (
        10  # minimum number of samples for each user in negative sampling
    )
    last_N: int = 3  # last N items in user behavior sequence as features
    features: List[str] = field(default_factory=list)

    enable_binning: bool = True
    binning_strategy: str = "quantile"  # "quantile" or "uniform"
    default_n_bins: int = 10  # default number of bins for continuous features

    def __post_init__(self):
        """Post initialization to set absolute paths and default values"""
        self.data_path = os.path.join(self._project_root, "data", "raw")
        self.save_path = os.path.join(self._project_root, "temp")
        self.all_recall_results_path = os.path.join(
            self.save_path, "all_recall_results.pkl"
        )
        self.train_set_path = os.path.join(self.data_path, "train_features.csv")
        self.test_set_path = os.path.join(self.save_path, "test_features.pkl")

        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

    # weights
    loc_alpha: float = 1.0
    loc_alpha_reverse: float = 0.7
    loc_beta: float = 0.9
    time_decay_alpha: float = 0.7
    created_time_alpha: float = 0.8

    # others
    offline: bool = True  # offline means using train data only
    random_seed: int = 23

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RecallConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class RankConfig:
    # general settings
    debug_mode: bool = False
    offline: bool = True
    random_seed: int = 23

    # path settings
    _project_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent.parent.resolve())
    )
    data_path: str = field(init=False)
    save_path: str = field(init=False)

    # Clean feature extractor paths
    main_features_path: str = field(init=False)
    user_profile_dict_path: str = field(init=False)
    item_features_dict_path: str = field(init=False)
    user_history_dict_path: str = field(init=False)
    feature_lists_path: str = field(init=False)
    discretizers_path: str = field(init=False)

    # DIN model hyperparameters
    din_embedding_dim: int = 32
    din_attention_hidden_units: List[int] = field(default_factory=lambda: [36])
    din_mlp_hidden_units: List[int] = field(default_factory=lambda: [200, 80])
    din_activation: str = "dice"  # "dice" or "prelu"
    din_seq_max_len: int = 30

    # training hyperparameters
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 5

    # DataLoader performance settings
    num_workers: int = (
        4  # Number of subprocesses for data loading (0 = main process only)
    )
    pin_memory: bool = True  # Faster data transfer to CUDA (set to True if using GPU)
    # Recommended: num_workers = 4-8 on GPU, 0 on CPU-only machines

    # negative sampling (to handle class imbalance)
    enable_negative_sampling: bool = True  # Whether to perform negative sampling
    negative_positive_ratio: float = (
        10.0  # Ratio of negative to positive samples (e.g., 10:1)
    )
    # Set to None to use all negative samples
    # Recommended range: 5-20 depending on original imbalance

    def __post_init__(self):
        """Post initialization to set absolute paths and default values"""
        self.data_path = os.path.join(self._project_root, "data", "raw")
        self.save_path = os.path.join(self._project_root, "temp")

        # Clean feature extractor paths
        self.main_features_path = os.path.join(self.save_path, "main_features.csv")
        self.user_profile_dict_path = os.path.join(
            self.save_path, "user_profile_dict.pkl"
        )
        self.item_features_dict_path = os.path.join(
            self.save_path, "item_features_dict.pkl"
        )
        self.user_history_dict_path = os.path.join(
            self.save_path, "user_history_dict.pkl"
        )
        self.feature_lists_path = os.path.join(self.save_path, "feature_lists.pkl")
        self.discretizers_path = os.path.join(self.save_path, "discretizers.pkl")

        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RankConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> dict:
        return self.__dict__
