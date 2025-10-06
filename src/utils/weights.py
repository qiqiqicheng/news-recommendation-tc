import numpy as np
import math


class WeightCalculator:
    @staticmethod
    def time_decay_weight(t1: float, t2: float, alpha: float = 0.7) -> float:
        """
        时间衰减权重

        Args:
            t1: 时间1
            t2: 时间2
            alpha: 衰减系数

        Returns:
            时间衰减权重
        """
        return np.exp(alpha ** np.abs(t1 - t2))

    @staticmethod
    def position_weight(distance: int, beta: float = 0.9) -> float:
        """
        位置权重(距离越远权重越小)

        Args:
            distance: 位置距离
            beta: 衰减系数

        Returns:
            位置权重
        """
        return beta**distance

    @staticmethod
    def activation_weight(count: float, scale: float = 100.0) -> float:
        """
        活跃度权重

        Args:
            count: 计数值
            scale: 缩放系数

        Returns:
            活跃度权重
        """
        return scale * 0.5 * count

    @staticmethod
    def log_penalty(count: int) -> float:
        """
        对数惩罚项(用于惩罚热门物品)

        Args:
            count: 计数值

        Returns:
            惩罚权重
        """
        return math.log(count + 1)

    @staticmethod
    def normalize_weight(weight: float, min_val: float, max_val: float) -> float:
        """
        归一化权重到[0, 1]

        Args:
            weight: 原始权重
            min_val: 最小值
            max_val: 最大值

        Returns:
            归一化后的权重
        """
        if max_val > min_val:
            return (weight - min_val) / (max_val - min_val)
        return 1.0
