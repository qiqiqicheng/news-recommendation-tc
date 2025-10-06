"""
持久化管理模块
统一管理pickle文件的保存和加载
"""

import pickle
import os
from typing import Any


class PersistenceManager:
    """持久化管理器,统一处理文件的保存和加载"""

    @staticmethod
    def save_pickle(obj: Any, path: str) -> None:
        """
        保存对象为pickle文件

        Args:
            obj: 要保存的对象
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path: str) -> Any:
        """
        从pickle文件加载对象

        Args:
            path: 文件路径

        Returns:
            加载的对象
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def exists(path: str) -> bool:
        """
        检查文件是否存在

        Args:
            path: 文件路径

        Returns:
            文件是否存在
        """
        return os.path.exists(path)
