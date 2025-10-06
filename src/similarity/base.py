import os
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..utils.persistence import PersistenceManager
from ..utils.config import RecallConfig


class BaseSimilarityCalculator(ABC):
    def __init__(self, config: RecallConfig):
        self.config = config
        self.save_path = config.save_path
        self.similarity_matrix = {}

    @abstractmethod
    def calculate(self, data: Any) -> Dict:
        pass

    def save(self, filename: str) -> None:
        if not self.similarity_matrix:
            raise ValueError("Similarity matrix is empty. Call calculate() first.")

        filepath = os.path.join(self.save_path, filename)
        PersistenceManager.save_pickle(self.similarity_matrix, filepath)
        print(f"Similarity matrix saved to: {filepath}")

    def load(self, filename: str) -> Dict:
        filepath = os.path.join(self.save_path, filename)

        if not PersistenceManager.exists(filepath):
            raise FileNotFoundError(f"Similarity matrix file not found: {filepath}")

        self.similarity_matrix = PersistenceManager.load_pickle(filepath)
        print(f"Similarity matrix loaded from: {filepath}")
        return self.similarity_matrix

    def get_similarity_matrix(self) -> Dict:
        return self.similarity_matrix

    def is_calculated(self) -> bool:
        return len(self.similarity_matrix) > 0
