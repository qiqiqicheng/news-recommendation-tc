"""
高级示例：在特征工程中使用 YoutubeDNN Embeddings
"""

import sys

sys.path.append(".")

import numpy as np
import pandas as pd
from src.utils.config import RecallConfig
from src.recall.youtubednn_recaller import YoutubeDNNRecaller


def calculate_user_item_similarity(user_emb_dict, item_emb_dict):
    """
    使用预训练的embeddings计算用户-物品相似度

    Args:
        user_emb_dict: {user_id: embedding}
        item_emb_dict: {item_id: embedding}

    Returns:
        函数用于计算相似度
    """

    def compute_sim(user_id, item_id):
        if user_id not in user_emb_dict or item_id not in item_emb_dict:
            return 0.0

        user_emb = user_emb_dict[user_id]
        item_emb = item_emb_dict[item_id]

        # 余弦相似度 (embeddings已经归一化)
        similarity = np.dot(user_emb, item_emb)
        return float(similarity)

    return compute_sim


def add_youtubednn_features(df, user_emb_dict, item_emb_dict):
    """
    为DataFrame添加基于YoutubeDNN的特征

    Args:
        df: DataFrame with columns ['user_id', 'item_id', ...]
        user_emb_dict: User embeddings
        item_emb_dict: Item embeddings

    Returns:
        DataFrame with new features
    """
    print("Adding YoutubeDNN features...")

    # 1. 用户-物品相似度
    compute_sim = calculate_user_item_similarity(user_emb_dict, item_emb_dict)
    df["youtubednn_user_item_sim"] = df.apply(
        lambda row: compute_sim(row["user_id"], row["item_id"]), axis=1
    )

    # 2. 用户embedding统计特征
    def get_user_emb_stats(user_id):
        if user_id not in user_emb_dict:
            return 0.0, 0.0, 0.0
        emb = user_emb_dict[user_id]
        return np.mean(emb), np.std(emb), np.max(emb)

    user_stats = df["user_id"].apply(get_user_emb_stats)
    df["user_emb_mean"] = [x[0] for x in user_stats]
    df["user_emb_std"] = [x[1] for x in user_stats]
    df["user_emb_max"] = [x[2] for x in user_stats]

    # 3. 物品embedding统计特征
    def get_item_emb_stats(item_id):
        if item_id not in item_emb_dict:
            return 0.0, 0.0, 0.0
        emb = item_emb_dict[item_id]
        return np.mean(emb), np.std(emb), np.max(emb)

    item_stats = df["item_id"].apply(get_item_emb_stats)
    df["item_emb_mean"] = [x[0] for x in item_stats]
    df["item_emb_std"] = [x[1] for x in item_stats]
    df["item_emb_max"] = [x[2] for x in item_stats]

    print(f"✅ Added {6} YoutubeDNN features")
    return df


def find_similar_items(item_id, item_emb_dict, topk=10):
    """
    查找相似物品

    Args:
        item_id: Target item ID
        item_emb_dict: Item embeddings
        topk: Number of similar items to return

    Returns:
        List of (item_id, similarity_score) tuples
    """
    if item_id not in item_emb_dict:
        return []

    target_emb = item_emb_dict[item_id]

    similarities = []
    for other_id, other_emb in item_emb_dict.items():
        if other_id == item_id:
            continue
        sim = np.dot(target_emb, other_emb)
        similarities.append((other_id, float(sim)))

    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:topk]


def main():
    config = RecallConfig()

    print("=" * 60)
    print("YoutubeDNN Embeddings - Advanced Usage Example")
    print("=" * 60)

    # 1. 加载预训练的embeddings
    print("\n[Step 1] Loading pre-trained embeddings...")
    try:
        user_emb_dict, item_emb_dict = YoutubeDNNRecaller.load_embedding_dicts(
            config.save_path
        )
    except FileNotFoundError:
        print("❌ Embedding files not found!")
        print("Please run 'youtubednn_embedding_builder.py' first.")
        return

    # 2. 示例：计算用户-物品相似度
    print("\n[Step 2] Computing user-item similarities...")
    sample_users = list(user_emb_dict.keys())[:5]
    sample_items = list(item_emb_dict.keys())[:5]

    compute_sim = calculate_user_item_similarity(user_emb_dict, item_emb_dict)

    print("\nUser-Item Similarity Matrix (sample):")
    print(f"{'User\\Item':<12}", end="")
    for item_id in sample_items:
        print(f"{item_id:<12}", end="")
    print()

    for user_id in sample_users:
        print(f"{user_id:<12}", end="")
        for item_id in sample_items:
            sim = compute_sim(user_id, item_id)
            print(f"{sim:<12.4f}", end="")
        print()

    # 3. 示例：查找相似物品
    print("\n[Step 3] Finding similar items...")
    target_item = sample_items[0]
    similar_items = find_similar_items(target_item, item_emb_dict, topk=5)

    print(f"\nTop 5 items similar to item {target_item}:")
    for rank, (item_id, sim) in enumerate(similar_items, 1):
        print(f"  {rank}. Item {item_id}: similarity = {sim:.4f}")

    # 4. 示例：在DataFrame中添加特征
    print("\n[Step 4] Adding features to DataFrame...")
    # 创建示例DataFrame
    sample_df = pd.DataFrame(
        {
            "user_id": sample_users * len(sample_items),
            "item_id": sample_items * len(sample_users),
        }
    )

    sample_df = add_youtubednn_features(sample_df, user_emb_dict, item_emb_dict)

    print("\nSample DataFrame with YoutubeDNN features:")
    print(sample_df.head(10))

    # 5. 统计信息
    print("\n" + "=" * 60)
    print("Embedding Statistics")
    print("=" * 60)

    # 用户embedding统计
    user_emb_array = np.array(list(user_emb_dict.values()))
    print(f"\nUser Embeddings:")
    print(f"  Shape: {user_emb_array.shape}")
    print(f"  Mean: {user_emb_array.mean():.4f}")
    print(f"  Std: {user_emb_array.std():.4f}")
    print(f"  Min: {user_emb_array.min():.4f}")
    print(f"  Max: {user_emb_array.max():.4f}")

    # 物品embedding统计
    item_emb_array = np.array(list(item_emb_dict.values()))
    print(f"\nItem Embeddings:")
    print(f"  Shape: {item_emb_array.shape}")
    print(f"  Mean: {item_emb_array.mean():.4f}")
    print(f"  Std: {item_emb_array.std():.4f}")
    print(f"  Min: {item_emb_array.min():.4f}")
    print(f"  Max: {item_emb_array.max():.4f}")

    print("\n" + "=" * 60)
    print("Advanced usage examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
