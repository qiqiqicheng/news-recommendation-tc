# Feature Engineering with Binning Strategy

## Overview

The feature extractor has been enhanced to automatically convert all continuous features into discrete (categorical) features using a configurable binning strategy. This makes the features suitable for embedding operations in downstream ranking models.

## Key Changes

### 1. Configuration Updates (`src/utils/config.py`)

Added three new configuration parameters:

```python
# binning strategy for continuous features
enable_binning: bool = True
binning_strategy: str = "quantile"  # "quantile", "uniform", or "kmeans"
default_n_bins: int = 10  # default number of bins for continuous features
```

- **enable_binning**: Toggle binning on/off
- **binning_strategy**: Choose from "quantile", "uniform", or "kmeans"
  - `quantile`: Equal-frequency binning (recommended for skewed distributions)
  - `uniform`: Equal-width binning (good for uniform distributions)
  - `kmeans`: K-means clustering-based binning (data-driven approach)
- **default_n_bins**: Number of bins to create for each continuous feature

### 2. Feature Extractor Enhancements (`src/features/feature_extractor.py`)

#### New Attributes

```python
self.continuous_features: List[str]  # list of continuous feature names
self.discrete_features: List[str]  # list of discrete feature names
self.all_features: List[str]  # list of all feature names for downstream use
self.binning_discretizers: Dict[str, KBinsDiscretizer]  # store discretizers
```

#### New Methods

1. **`_identify_feature_types()`**
   - Automatically identifies which features are continuous vs discrete
   - Uses heuristics: numeric features with >20 unique values are treated as continuous
   - Excludes ID columns and labels from feature engineering

2. **`_apply_binning()`**
   - Converts continuous features to discrete using KBinsDiscretizer
   - Fits on training data and transforms both train and test sets
   - Handles missing values by imputing with median
   - Stores discretizers for future use on new data

3. **`_build_feature_list()`**
   - Creates comprehensive feature list for downstream use
   - Updates config.features for easy access
   - Excludes non-feature columns (IDs, labels, temporary columns)

4. **`get_feature_list()`**
   - Public method to retrieve the feature list
   - Returns: List[str] of all feature names

5. **`get_binning_discretizers()`**
   - Public method to retrieve the fitted discretizers
   - Returns: Dict[str, KBinsDiscretizer] mapping feature names to discretizers
   - Useful for applying same binning to new/online data

#### Updated Workflow

The `extract_features()` method now follows this pipeline:

```
1. Add labels
2. Load embedding dictionaries
3. Extract history features
4. Extract user activate degree features
5. Extract article popularity features
6. Extract user habits features
7. Perform negative sampling
8. Identify feature types (NEW)
9. Apply binning (NEW)
10. Build feature list (NEW)
11. Save features
```

## Feature Types

### Continuous Features (Before Binning)

These features are automatically identified and binned:

- `score`: recall score
- `sim_1`, `sim_2`, `sim_3`: similarity features
- `time_diff_1`, `time_diff_2`, `time_diff_3`: time difference features
- `word_diff_1`, `word_diff_2`, `word_diff_3`: word difference features
- `sim_max`, `sim_mean`, `sim_min`, `sim_std`: similarity statistics
- `item_user_sim`: user-item similarity from embeddings
- `user_click_count`: normalized user click frequency
- `user_avg_time_gap`: normalized average time gap
- `article_popularity`: normalized article popularity
- `avg_click_time`: average click time
- `avg_word_count`: average word count

### Discrete Features (Already Categorical)

These features remain unchanged:

- `device_group`: user's device group
- `recall_in_user_cat`: whether recall item's category is in user's clicked categories

## Usage Example

```python
from src.features.feature_extractor import FeatureExtractor
from src.utils.config import RecallConfig

# Configure with binning settings
config = RecallConfig(
    enable_binning=True,
    binning_strategy="quantile",
    default_n_bins=10,
    last_N=3
)

# Initialize and extract features
extractor = FeatureExtractor(config)
extractor.load_all()
extractor.extract_features(save=True)

# Get feature list for downstream model
feature_list = extractor.get_feature_list()
print(f"Features: {feature_list}")

# Get train/test data
train_df, test_df = extractor.get_train_test_data()

# Use features in your ranking model
# Now all features are discrete and ready for embedding!
```

## Benefits

1. **Ready for Embeddings**: All features are discrete, perfect for embedding layers in deep learning models
2. **Reduced Dimensionality**: Binning reduces the cardinality of continuous features
3. **Better Generalization**: Discrete features are less prone to overfitting
4. **Interpretability**: Binned features are easier to interpret and analyze
5. **Consistency**: Same binning can be applied to new data using stored discretizers

## Applying to New Data

```python
# Get stored discretizers
discretizers = extractor.get_binning_discretizers()

# Apply to new data
for feature_name, discretizer in discretizers.items():
    if feature_name in new_df.columns:
        new_df[feature_name] = discretizer.transform(new_df[[feature_name]])
```

## Notes

- The binning is fitted only on training data to prevent data leakage
- Missing values are imputed with median before binning
- Features with <2 unique values are skipped
- The number of bins is automatically reduced if there are fewer unique values than requested bins
