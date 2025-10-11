"""
示例：使用 YoutubeDNN 构建和保存用户/物品 Embedding 字典
"""

import sys

sys.path.append(".")

from src.utils.config import RecallConfig
from src.data.loaders import ClickLogLoader
from src.recall.youtubednn_recaller import YoutubeDNNRecaller
from src.data.extractors import InteractionFeatureExtractor


def main():
    # 1. 初始化配置
    config = RecallConfig()
    config.debug_mode = True  # 使用debug模式加快演示
    config.youtubednn_epochs = 2
    config.youtubednn_batch_size = 256

    print("=" * 60)
    print("YoutubeDNN Embedding Dictionary Builder")
    print("=" * 60)

    # 2. 加载数据
    print("\n[Step 1] Loading data...")
    click_loader = ClickLogLoader(config)
    all_clicks = click_loader.load_all(debug=config.debug_mode, offline=True)
    all_clicks["click_timestamp"] = InteractionFeatureExtractor.normalize_timestamp(
        all_clicks
    )
    print(f"✅ Loaded {len(all_clicks)} click records")

    # 3. 初始化并训练模型
    print("\n[Step 2] Training YoutubeDNN model...")
    recaller = YoutubeDNNRecaller(config)
    recaller.train(
        all_clicks,
        epochs=config.youtubednn_epochs,
        batch_size=config.youtubednn_batch_size,
        learning_rate=config.youtubednn_learning_rate,
    )
    print("✅ Model training completed")

    # 4. 构建并保存 Embedding 字典
    print("\n[Step 3] Building embedding dictionaries...")
    user_emb_dict, item_emb_dict = recaller.build_embedding_dicts(
        click_df=all_clicks, save_path=config.save_path
    )

    # 5. 展示结果
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"User embedding dictionary size: {len(user_emb_dict)}")
    print(f"Item embedding dictionary size: {len(item_emb_dict)}")

    # 显示示例
    sample_user_id = list(user_emb_dict.keys())[0]
    sample_item_id = list(item_emb_dict.keys())[0]

    print(f"\nSample user ID: {sample_user_id}")
    print(f"User embedding shape: {user_emb_dict[sample_user_id].shape}")
    print(f"User embedding (first 5 dims): {user_emb_dict[sample_user_id][:5]}")

    print(f"\nSample item ID: {sample_item_id}")
    print(f"Item embedding shape: {item_emb_dict[sample_item_id].shape}")
    print(f"Item embedding (first 5 dims): {item_emb_dict[sample_item_id][:5]}")

    # 6. 测试加载功能
    print("\n[Step 4] Testing loading functionality...")
    loaded_user_emb, loaded_item_emb = YoutubeDNNRecaller.load_embedding_dicts(
        config.save_path
    )

    # 验证
    assert len(loaded_user_emb) == len(user_emb_dict)
    assert len(loaded_item_emb) == len(item_emb_dict)
    print("✅ Loading test passed!")

    print("\n" + "=" * 60)
    print("All steps completed successfully!")
    print("=" * 60)
    print(f"\nEmbedding files saved to: {config.save_path}")
    print("  - user_youtubednn_emb.pkl")
    print("  - item_youtubednn_emb.pkl")


if __name__ == "__main__":
    main()
