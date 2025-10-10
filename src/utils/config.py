from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RecallConfig:
    # path settings
    data_path: str = "../data/raw/"
    save_path: str = "../temp/"

    # sampling and debug
    debug_mode: bool = False
    debug_sample_size: int = 10000
    debug_user_sample_size: int = 1000

    # ItemCF
    itemcf_sim_item_topk: int = 20
    itemcf_recall_num: int = 10
    itemcf_hot_topk: int = 50

    # UserCF
    usercf_sim_user_topk: int = 20
    usercf_recall_num: int = 10

    # Embedding
    embedding_topk: int = 50
    embedding_dim: Optional[int] = None

    # YoutubeDNN
    youtubednn_seq_max_len: int = 30
    youtubednn_embedding_dim: int = 16
    youtubednn_hidden_units: Optional[List[int]] = None # not decided yet
    youtubednn_negsample: int = 4
    youtubednn_epochs: int = 1
    youtubednn_batch_size: int = 256
    youtubednn_learning_rate: float = 0.001

    def __post_init__(self):
        """Post initialization to set default values for mutable types"""
        if self.youtubednn_hidden_units is None:
            self.youtubednn_hidden_units = [64, 16]

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
