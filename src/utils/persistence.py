import pickle
import os
from typing import Any


class PersistenceManager:
    @staticmethod
    def save_pickle(obj: Any, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
            
        print(f"Saved object to {path}")

    @staticmethod
    def load_pickle(path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)
        
        print(f"Loaded object from {path}")

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(path)
