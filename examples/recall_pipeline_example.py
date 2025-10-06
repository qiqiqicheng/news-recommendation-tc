import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import RecallConfig
from src.data.loaders import ClickLogLoader, ArticleInfoLoader
from src.data.extractors import (
    UserFeatureExtractor,
    ItemFeatureExtractor,
    InteractionFeatureExtractor,
)
from src.similarity import ItemCFSimilarity, UserCFSimilarity, EmbeddingSimilarity
from src.recall import ItemCFRecaller, UserCFRecaller, RecallFusion


def main():
    """Main function demonstrating the recall pipeline"""

    # =====================
    # 1. Configuration
    # =====================
    print("=" * 50)
    print("Step 1: Configuration")
    print("=" * 50)

    config = RecallConfig(
        data_path="../data/raw/",
        save_path="../temp/",
        debug_mode=True,  # Use debug mode for faster testing
        debug_sample_size=10000,
        itemcf_sim_item_topk=20,
        itemcf_recall_num=10,
        usercf_sim_user_topk=20,
        usercf_recall_num=10,
    )

    print(f"Config created: debug_mode={config.debug_mode}")

    # =====================
    # 2. Data Loading
    # =====================
    print("\n" + "=" * 50)
    print("Step 2: Data Loading")
    print("=" * 50)

    click_loader = ClickLogLoader(config)
    article_loader = ArticleInfoLoader(config)

    # Load data
    all_clicks = click_loader.load_all(debug=True, offline=True)
    articles_df, articles_emb_df = article_loader.load(debug=True)

    print(f"Loaded {len(all_clicks)} click records")
    print(f"Loaded {len(articles_df)} articles")

    # Normalize timestamp
    all_clicks = InteractionFeatureExtractor.normalize_timestamp(all_clicks)

    # =====================
    # 3. Feature Extraction
    # =====================
    print("\n" + "=" * 50)
    print("Step 3: Feature Extraction")
    print("=" * 50)

    # Extract item features
    item_type_dict, item_words_dict, item_created_time_dict = (
        ItemFeatureExtractor.get_item_info_dict(articles_df)
    )

    item_topk_click = ItemFeatureExtractor.get_item_topk_click(all_clicks, k=50)

    print(f"Extracted {len(item_type_dict)} item features")
    print(f"Top-50 popular items extracted")

    # Extract user features
    user_item_time_dict = UserFeatureExtractor.get_user_item_time_dict(all_clicks)
    user_activate_degree_dict = UserFeatureExtractor.get_user_activate_degree_dict(
        all_clicks
    )

    print(f"Extracted features for {len(user_item_time_dict)} users")

    # =====================
    # 4. Similarity Calculation
    # =====================
    print("\n" + "=" * 50)
    print("Step 4: Similarity Calculation")
    print("=" * 50)

    # ItemCF similarity
    print("\n4.1 Calculating ItemCF similarity...")
    itemcf_sim_calc = ItemCFSimilarity(config)
    i2i_sim = itemcf_sim_calc.calculate(all_clicks, item_created_time_dict)
    itemcf_sim_calc.save("itemcf_i2i_sim.pkl")

    # UserCF similarity
    print("\n4.2 Calculating UserCF similarity...")
    usercf_sim_calc = UserCFSimilarity(config)
    u2u_sim = usercf_sim_calc.calculate(all_clicks, user_activate_degree_dict)
    usercf_sim_calc.save("usercf_u2u_sim.pkl")

    # Embedding similarity
    print("\n4.3 Calculating Embedding similarity...")
    emb_sim_calc = EmbeddingSimilarity(config)
    emb_i2i_sim = emb_sim_calc.calculate(articles_emb_df, topk=50)
    emb_sim_calc.save("emb_i2i_sim.pkl")

    # =====================
    # 5. Recall
    # =====================
    print("\n" + "=" * 50)
    print("Step 5: Recall")
    print("=" * 50)

    # ItemCF recall
    print("\n5.1 ItemCF Recall...")
    itemcf_recaller = ItemCFRecaller(
        config=config,
        similarity_matrix=i2i_sim,
        item_created_time_dict=item_created_time_dict,
        user_item_time_dict=user_item_time_dict,
        item_topk_click=item_topk_click,
        emb_similarity_matrix=emb_i2i_sim,
    )

    # Recall for all users
    all_users = list(user_item_time_dict.keys())[:100]  # Test on 100 users
    itemcf_results = itemcf_recaller.batch_recall(all_users, topk=10)
    print(f"ItemCF recalled items for {len(itemcf_results)} users")

    # UserCF recall
    print("\n5.2 UserCF Recall...")
    usercf_recaller = UserCFRecaller(
        config=config,
        similarity_matrix=u2u_sim,
        user_item_time_dict=user_item_time_dict,
        item_created_time_dict=item_created_time_dict,
        item_topk_click=item_topk_click,
        emb_similarity_matrix=emb_i2i_sim,
    )

    usercf_results = usercf_recaller.batch_recall(all_users, topk=10)
    print(f"UserCF recalled items for {len(usercf_results)} users")

    # =====================
    # 6. Multi-Recall Fusion
    # =====================
    print("\n" + "=" * 50)
    print("Step 6: Multi-Recall Fusion")
    print("=" * 50)

    fusion = RecallFusion(config)

    # Add recall results with weights
    fusion.add_recall_result("itemcf", itemcf_results, weight=1.0)
    fusion.add_recall_result("usercf", usercf_results, weight=1.0)

    # Fuse results
    fused_results = fusion.fuse(topk=20)
    print(f"\nFusion completed: {len(fused_results)} users")

    # Get statistics
    stats = fusion.get_statistics()
    print("\nFusion Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # =====================
    # 7. Sample Results
    # =====================
    print("\n" + "=" * 50)
    print("Step 7: Sample Results")
    print("=" * 50)

    # Show results for a sample user
    sample_user = all_users[0]
    print(f"\nResults for user {sample_user}:")
    print(f"\nItemCF (Top 5):")
    for item, score in itemcf_results[sample_user][:5]:
        print(f"  Item {item}: {score:.4f}")

    print(f"\nUserCF (Top 5):")
    for item, score in usercf_results[sample_user][:5]:
        print(f"  Item {item}: {score:.4f}")

    print(f"\nFused (Top 5):")
    for item, score in fused_results[sample_user][:5]:
        print(f"  Item {item}: {score:.4f}")

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
