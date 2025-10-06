from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RecallConfig:
    """召回配置"""

    # 数据路径配置
    data_path: str = "../data/raw/"
    save_path: str = "../temp/"

    # 采样配置
    debug_mode: bool = False
    debug_sample_size: int = 10000
    debug_user_sample_size: int = 1000

    # ItemCF配置
    itemcf_sim_item_topk: int = 20
    itemcf_recall_num: int = 10
    itemcf_hot_topk: int = 50

    # UserCF配置
    usercf_sim_user_topk: int = 20
    usercf_recall_num: int = 10

    # Embedding配置
    embedding_topk: int = 50
    embedding_dim: Optional[int] = None

    # YoutubeDNN配置
    youtubednn_seq_max_len: int = 30
    youtubednn_embedding_dim: int = 16
    youtubednn_hidden_units: Optional[List[int]] = None
    youtubednn_negsample: int = 4
    youtubednn_epochs: int = 1
    youtubednn_batch_size: int = 256
    youtubednn_learning_rate: float = 0.001

    def __post_init__(self):
        """Post initialization to set default values for mutable types"""
        if self.youtubednn_hidden_units is None:
            self.youtubednn_hidden_units = [64, 16]

    # 权重参数
    loc_alpha: float = 1.0
    loc_alpha_reverse: float = 0.7
    loc_beta: float = 0.9
    time_decay_alpha: float = 0.7
    created_time_alpha: float = 0.8

    # 其他配置
    offline: bool = True  # 是否离线模式(只用训练集)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RecallConfig":
        """从字典创建配置对象"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> dict:
        """转换为字典"""
        return self.__dict__
