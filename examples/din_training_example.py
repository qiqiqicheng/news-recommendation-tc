"""
Example script for training DIN model with organized data structure.

This script demonstrates:
1. Extract features with DIN-specific organization
2. Train DIN model with DataLoader
3. Use the improved data organization system
"""

from src.features.feature_extractor import FeatureExtractor
from src.rank.DIN import DINRanker
from src.utils.config import RecallConfig, RankConfig

# Step 1: Extract features with DIN-specific organization
print("=" * 50)
print("Step 1: Extract Features")
print("=" * 50)

recall_config = RecallConfig(
    debug_mode=True,
    debug_sample_size=10000,
    enable_binning=True,
    binning_strategy="quantile",
    default_n_bins=10,
    last_N=3,
    neg_sample_rate=0.001,
    min_sample_size=10,
)

extractor = FeatureExtractor(recall_config)
extractor.load_all()
extractor.extract_features(save=True)

print("\nFeature extraction completed!")
print(f"User profile features: {extractor.user_profile_features}")
print(f"Item features: {extractor.item_features}")
print(f"Context features: {extractor.context_features}")

# Step 2: Train DIN model
print("\n" + "=" * 50)
print("Step 2: Train DIN Model")
print("=" * 50)

rank_config = RankConfig(
    debug_mode=True,
    # DIN model hyperparameters
    din_embedding_dim=32,
    din_attention_hidden_units=[36],
    din_mlp_hidden_units=[200, 80],
    din_activation="dice",
    din_seq_max_len=30,
    # training hyperparameters
    batch_size=256,
    learning_rate=0.001,
    epochs=3,
)

ranker = DINRanker(rank_config)
ranker.train()

print("\nTraining completed!")
