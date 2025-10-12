from abc import ABC, abstractmethod
import os
import pandas as pd

from src.utils.config import RecallConfig, RankConfig
from src.utils.persistence import PersistenceManager

class BaseRanker(ABC):
    def __init__(self, config: RankConfig) -> None:
        self.config = config
        
    def load(self):
        if os.path.exists(self.config.feature_list_path):
            self.feature_list= PersistenceManager.load_pickle(self.config.feature_list_path)
        else:
            raise FileNotFoundError(f"Feature list not found at {self.config.feature_list_path}")
            
        if os.path.exists(self.config.train_set_path):
            self.train_set = pd.read_csv(self.config.train_set_path)
        else:
            raise FileNotFoundError(f"Training set not found at {self.config.train_set_path}")
        
        if os.path.exists(self.config.test_set_path):
            self.test_set = pd.read_pickle(self.config.test_set_path)
        else:
            raise FileNotFoundError(f"Test set not found at {self.config.test_set_path}")
        
    def train(self):
        pass
    
    def predict(self):
        pass