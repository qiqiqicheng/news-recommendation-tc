from dataclasses import dataclass, field
from typing import Optional, List
import os
from pathlib import Path


@dataclass
class RecallConfig:
    # path settings (使用绝对路径,基于项目根目录)
    _project_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent.parent.resolve())
    )
    data_path: str = field(init=False)
    save_path: str = field(init=False)
    recall_path: str = field(init=False)

    # sampling and debug
    debug_mode: bool = True
    debug_sample_size: int = 10000
    debug_user_sample_size: int = 10000

    # ItemCF
    itemcf_sim_item_topk: int = 20
    itemcf_recall_num: int = 20
    itemcf_hot_topk: int = 20

    # UserCF
    usercf_sim_user_topk: int = 20
    usercf_recall_num: int = 30

    # Embedding
    embedding_topk: int = 20
    embedding_dim: int = 64

    # YoutubeDNN
    youtubednn_seq_max_len: int = 30
    youtubednn_embedding_dim: int = 16
    youtubednn_hidden_units: List[int] = field(default_factory=lambda: [64, 16])
    youtubednn_negsample: int = 4
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

    def __post_init__(self):
        """Post initialization to set absolute paths and default values"""
        self.data_path = os.path.join(self._project_root, "data", "raw")
        self.save_path = os.path.join(self._project_root, "temp")
        self.all_recall_results_path = os.path.join(self.save_path, "all_recall_results.pkl")

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
