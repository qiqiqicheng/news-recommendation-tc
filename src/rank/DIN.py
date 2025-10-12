import torch
import torch.nn as nn


from src.rank.base import BaseRanker
from src.utils.config import RecallConfig, RankConfig

class ActivationUnit(nn.Module):
    def __init__(self, config: RankConfig) -> None:
        super().__init__()
        

class DINModel(nn.Module):
    def __init__(self, config: RankConfig) -> None:
        super().__init__()
        self.config = config

        # DIN architecture
        # 1. embedding layer
        



class DINRanker(BaseRanker):
    def __init__(self, config: RankConfig):
        super().__init__(config)
        
    
