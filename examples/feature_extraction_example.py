"""
Example script demonstrating feature extraction with binning.

This script shows how to:
1. Initialize FeatureExtractor with custom configuration
2. Load data and recall results
3. Extract features with automatic binning
4. Access the feature list for downstream models
"""

from src.features.feature_extractor import FeatureExtractor
from src.utils.config import RecallConfig

# Create configuration
config = RecallConfig(
    debug_mode=True,
    debug_sample_size=10000,
    # binning configuration
    enable_binning=True,
    binning_strategy="quantile",  # "quantile", "uniform", or "kmeans"
    default_n_bins=10,
    # feature extraction configuration
    last_N=3,
    neg_sample_rate=0.001,
    min_sample_size=10,
)

# Initialize feature extractor
extractor = FeatureExtractor(config)

# Load data and recall results
extractor.load_all()

# Extract features (this will automatically apply binning)
extractor.extract_features(save=True)

# Get feature list for downstream use
feature_list = extractor.get_feature_list()
print(f"\nTotal features: {len(feature_list)}")
print(f"Feature list: {feature_list}")

# Get train and test data
train_df, test_df = extractor.get_train_test_data()
print(f"\nTrain set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Get binning discretizers (useful for applying same binning to new data)
discretizers = extractor.get_binning_discretizers()
print(f"\nNumber of binned features: {len(discretizers)}")
print(f"Binned features: {list(discretizers.keys())}")

# Show sample of the features
print("\nSample of train data:")
print(train_df[feature_list].head())

# Verify all features are discrete (categorical)
print("\nFeature data types:")
for feature in feature_list:
    print(
        f"{feature}: {train_df[feature].dtype}, unique values: {train_df[feature].nunique()}"
    )
