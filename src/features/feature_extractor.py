import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from gensim.models import Word2Vec
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer

from src.data import ClickLogLoader, ArticleInfoLoader
from src.data import (
    UserFeatureExtractor,
    ItemFeatureExtractor,
    InteractionFeatureExtractor,
)
from src.utils.config import RecallConfig
from src.utils.persistence import PersistenceManager


class FeatureExtractor:
    """
    Clean feature extractor for news recommendation ranking stage.

    Data organization:
    1. DataFrame: (user_id, item_id, context_features...) - main interaction data
    2. user_profile_dict: {user_id: {feature: value}} - user-level features
    3. item_features_dict: {item_id: {feature: value}} - item-level features
    4. user_history_dict: {user_id: [item_id, ...]} - user behavior sequences

    Feature groups:
    - user_profile: user-level features (click_count, avg_time_gap, device_group, etc.)
    - item_features: item-level features (category_id, popularity, created_at_ts, words_count)
    - context: interaction features (score, similarity features, time_diff, word_diff, etc.)
    """

    def __init__(self, config: RecallConfig):
        self.config = config

        # Data containers
        self.main_df: pd.DataFrame = (
            pd.DataFrame()
        )  # (user_id, item_id, context_features...)
        self.user_profile_dict: Dict[str, Dict] = {}  # {user_id: {feature: value}}
        self.item_features_dict: Dict[str, Dict] = {}  # {item_id: {feature: value}}
        self.user_history_dict: Dict[str, List[str]] = {}  # {user_id: [item_id, ...]}

        # Feature lists
        self.user_profile_features: List[str] = []
        self.item_features: List[str] = []
        self.context_features: List[str] = []

        # Binning discretizers for continuous features
        self.discretizers: Dict[str, KBinsDiscretizer] = {}

    def load_data(self):
        """
        Load click logs and article information

        Important: Must load same data as recall pipeline for consistency
        - In debug mode: sample users then load corresponding articles
        - In normal mode: load all data
        - Always load ALL article metadata to avoid missing features
        """
        print("Loading data...")

        click_loader = ClickLogLoader(self.config)
        article_loader = ArticleInfoLoader(self.config)

        if self.config.debug_mode:
            # Load sampled user clicks
            self.train_click_df, self.test_click_df = click_loader.load(
                debug=True, sample_size=self.config.debug_sample_size
            )

            # Get all item IDs from sampled users' clicks
            all_item_ids = set(self.train_click_df["click_article_id"].unique()).union(
                set(self.test_click_df["click_article_id"].unique())
            )

            # CRITICAL FIX: Load articles matching recall pipeline's load strategy
            # In debug mode, recall pipeline uses load_with_user_sample, so we must too
            self.article_info_df, article_emb_df = article_loader.load_with_user_sample(
                list(all_item_ids)
            )

            print(
                f"Debug mode: loaded {len(all_item_ids)} unique articles for sampled users"
            )
        else:
            # Normal mode: load all data (same as recall pipeline)
            self.train_click_df, self.test_click_df = click_loader.load(debug=False)
            self.article_info_df, article_emb_df = article_loader.load(debug=False)

        print(
            f"Loaded {len(self.train_click_df)} train click records and {len(self.test_click_df)} test click records"
        )
        print(f"Loaded {len(self.article_info_df)} articles")

        # Create article info dictionaries
        (
            self.article_type_dict,
            self.article_words_dict,
            self.article_created_time_dict,
        ) = ItemFeatureExtractor.get_item_info_dict(self.article_info_df)

        # CRITICAL FIX: Convert all dictionary keys to string for consistent lookup
        # Main dataframe uses string IDs (e.g., "123456"), but dict keys might be integers
        self.article_type_dict = {
            str(int(k)): v for k, v in self.article_type_dict.items()
        }
        self.article_words_dict = {
            str(int(k)): v for k, v in self.article_words_dict.items()
        }
        self.article_created_time_dict = {
            str(int(k)): v for k, v in self.article_created_time_dict.items()
        }

        print(
            f"""
              CHECKPOINT:
              article_created_time_dict created with {len(self.article_created_time_dict)} entries
              sample (after string conversion): {list(self.article_created_time_dict.items())[:5]}
              """
        )

        # Create article content embedding dict
        emb_cols = [f"emb_{i}" for i in range(250)]
        article_ids = article_emb_df["article_id"].values
        embedding_matrix = article_emb_df[emb_cols].values

        # CRITICAL FIX: Use string keys to match main_df['item_id'] type
        self.article_content_emb_dict = {
            str(int(article_id)): embedding_matrix[idx]
            for idx, article_id in enumerate(article_ids)
        }
        

        if os.path.exists(os.path.join(self.config.save_path, "user_youtubednn_emb.pkl")):
            self.user_youtubednn_emb_dict = PersistenceManager.load_pickle(
                os.path.join(self.config.save_path, "user_youtubednn_emb.pkl")
            )
            print("successfully loaded user_youtubednn_emb_dict")

        if os.path.exists(os.path.join(self.config.save_path, "article_youtubednn_emb.pkl")):
            self.article_youtubednn_emb_dict = PersistenceManager.load_pickle(
                os.path.join(self.config.save_path, "article_youtubednn_emb.pkl")
            )
            print("successfully loaded article_youtubednn_emb_dict")

        print("Data loaded.")

    def load_recall(self, load_path: Optional[str] = None):
        """Load recall results and create main dataframe"""
        if load_path is None:
            load_path = self.config.all_recall_results_path

        print("Loading recall results...")
        recall_dict = PersistenceManager.load_pickle(load_path)

        # IMPORTANT: Train/Val/Test split strategy
        # - Train users (from train_click_log): Have ground truth → used for training
        #   We further split these into train_set and val_set
        # - Test users (from testA_click_log): NO ground truth → used for final prediction

        train_user_ids = set(
            str(int(uid)) for uid in self.train_click_df["user_id"].unique()
        )
        test_user_ids = set(
            str(int(uid)) for uid in self.test_click_df["user_id"].unique()
        )

        print(f"Train users (with labels): {len(train_user_ids)}")
        print(f"Test users (for prediction): {len(test_user_ids)}")

        # Split train users into train_set (80%) and val_set (20%)
        train_user_list = list(train_user_ids)
        np.random.seed(self.config.random_seed)
        np.random.shuffle(train_user_list)

        split_idx = int(len(train_user_list) * 0.8)
        train_set_users = set(train_user_list[:split_idx])
        val_set_users = set(train_user_list[split_idx:])

        print(f"  - Train set users: {len(train_set_users)} (80%)")
        print(f"  - Validation set users: {len(val_set_users)} (20%)")

        # Create main dataframe from recall results
        df_rows = []
        for user_id, recall_list in recall_dict.items():
            user_id = (
                str(int(float(user_id)))
                if isinstance(user_id, str)
                else str(int(user_id))
            )

            # Determine dataset membership
            is_train = user_id in train_set_users
            is_val = user_id in val_set_users
            is_test = user_id in test_user_ids

            for item_id, score in recall_list:
                item_id = (
                    str(int(float(item_id)))
                    if isinstance(item_id, str)
                    else str(int(item_id))
                )
                df_rows.append([user_id, item_id, score, is_train, is_val, is_test])

        self.main_df = pd.DataFrame(
            df_rows,
            columns=["user_id", "item_id", "score", "is_train", "is_val", "is_test"],
        )

        print(f"\nMain dataframe created with {len(self.main_df)} interactions")
        print(f"  - Train samples: {self.main_df['is_train'].sum()}")
        print(f"  - Validation samples: {self.main_df['is_val'].sum()}")
        print(f"  - Test samples (no labels): {self.main_df['is_test'].sum()}")
        print(
            f"user_id dtype: {self.main_df['user_id'].dtype}, example: {self.main_df['user_id'].iloc[0] if len(self.main_df) > 0 else 'empty'}"
        )
        print(
            f"item_id dtype: {self.main_df['item_id'].dtype}, example: {self.main_df['item_id'].iloc[0] if len(self.main_df) > 0 else 'empty'}"
        )

        # Verify data type consistency for dictionary lookups
        if len(self.main_df) > 0:
            sample_item = self.main_df["item_id"].iloc[0]
            print(f"\n=== Data Type Consistency Check ===")
            print(
                f"Sample item_id from main_df: {sample_item} (type: {type(sample_item)})"
            )
            print(
                f"Sample key from article_type_dict: {list(self.article_type_dict.keys())[0]} (type: {type(list(self.article_type_dict.keys())[0])})"
            )
            print(
                f"Sample key from article_created_time_dict: {list(self.article_created_time_dict.keys())[0]} (type: {type(list(self.article_created_time_dict.keys())[0])})"
            )
            print(
                f"Can lookup sample_item in article_created_time_dict: {sample_item in self.article_created_time_dict}"
            )
            if sample_item in self.article_created_time_dict:
                print(f"Lookup result: {self.article_created_time_dict[sample_item]}")
            print("=" * 50)

        # VALIDATION: Check for missing metadata
        self._validate_recall_data()

    def _validate_recall_data(self):
        """
        Validate that recalled items have corresponding metadata
        Reports missing articles to help diagnose data consistency issues
        """
        print("\n=== Data Validation ===")

        unique_items = set(self.main_df["item_id"].unique())
        print(f"Total unique recalled items: {len(unique_items)}")

        # Check metadata coverage
        items_in_type_dict = set(str(k) for k in self.article_type_dict.keys())
        items_in_time_dict = set(str(k) for k in self.article_created_time_dict.keys())
        items_in_words_dict = set(str(k) for k in self.article_words_dict.keys())
        items_in_emb_dict = set(str(k) for k in self.article_content_emb_dict.keys())

        missing_type = unique_items - items_in_type_dict
        missing_time = unique_items - items_in_time_dict
        missing_words = unique_items - items_in_words_dict
        missing_emb = unique_items - items_in_emb_dict

        print(
            f"Items missing category info: {len(missing_type)} ({len(missing_type)/len(unique_items)*100:.2f}%)"
        )
        print(
            f"Items missing time info: {len(missing_time)} ({len(missing_time)/len(unique_items)*100:.2f}%)"
        )
        print(
            f"Items missing words info: {len(missing_words)} ({len(missing_words)/len(unique_items)*100:.2f}%)"
        )
        print(
            f"Items missing embedding: {len(missing_emb)} ({len(missing_emb)/len(unique_items)*100:.2f}%)"
        )

        if missing_type or missing_time or missing_words or missing_emb:
            print("\nWARNING: Some recalled items are missing metadata!")

            if len(missing_type) > 0:
                print(f"\nSample missing item IDs: {list(missing_type)[:5]}")
        else:
            print("All recalled items have complete metadata")

        print("=" * 50 + "\n")

    def load_all(self):
        """Load all required data"""
        self.load_data()
        self.load_recall()

    def _extract_user_profile_features(self):
        """Extract user-level features"""
        print("Extracting user profile features...")

        # User click statistics
        user_stats = (
            self.train_click_df.groupby("user_id")
            .agg(
                {
                    "click_article_id": "count",
                    "click_timestamp": lambda x: (
                        (x.max() - x.min()) / (len(x) - 1) if len(x) > 1 else 0
                    ),
                }
            )
            .reset_index()
        )
        user_stats.columns = ["user_id", "click_count", "avg_time_gap"]

        # Normalize click count and time gap
        scaler_count = MinMaxScaler()
        scaler_time_gap = MinMaxScaler()

        user_stats["user_click_count"] = scaler_count.fit_transform(
            user_stats[["click_count"]]
        )
        user_stats["user_avg_time_gap"] = scaler_time_gap.fit_transform(
            user_stats[["avg_time_gap"]]
        )

        # User device group (mode)
        user_device = (
            self.train_click_df.groupby("user_id")["click_deviceGroup"]
            .agg(lambda x: x.mode()[0] if len(x) > 0 else "unknown")
            .reset_index()
        )
        user_device.columns = ["user_id", "device_group"]

        # User average click time
        user_click_time = (
            self.train_click_df.groupby("user_id")["click_timestamp"]
            .mean()
            .reset_index()
        )
        user_click_time.columns = ["user_id", "avg_click_time"]

        # Normalize click time
        scaler_time = MinMaxScaler()
        user_click_time["avg_click_time"] = scaler_time.fit_transform(
            user_click_time[["avg_click_time"]]
        )

        # User average word count
        user_avg_word_count = {}
        for user_id, group in self.train_click_df.groupby("user_id"):
            article_ids = group["click_article_id"].unique()
            word_counts = [
                self.article_words_dict.get(article_id, 0) for article_id in article_ids
            ]
            user_avg_word_count[user_id] = np.mean(word_counts) if word_counts else 0

        # Combine all user features
        user_features = user_stats.merge(user_device, on="user_id", how="left")
        user_features = user_features.merge(user_click_time, on="user_id", how="left")

        # Fill missing values (fixed: avoid FutureWarning by using proper assignment)
        user_features["device_group"] = user_features["device_group"].fillna("unknown")
        user_features["avg_click_time"] = user_features["avg_click_time"].fillna(0.5)

        # Create user profile dictionary
        for _, row in user_features.iterrows():
            user_id = str(int(row["user_id"]))
            self.user_profile_dict[user_id] = {
                "user_click_count": float(row["user_click_count"]),
                "user_avg_time_gap": float(row["user_avg_time_gap"]),
                "device_group": str(row["device_group"]),
                "avg_click_time": float(row["avg_click_time"]),
                "avg_word_count": float(
                    user_avg_word_count.get(int(row["user_id"]), 0)
                ),
            }

        # Define user profile feature names
        self.user_profile_features = [
            "user_click_count",
            "user_avg_time_gap",
            "device_group",
            "avg_click_time",
            "avg_word_count",
        ]

        print(
            f"User profile features extracted for {len(self.user_profile_dict)} users"
        )

    def _extract_item_features(self):
        """Extract item-level features"""
        print("Extracting item features...")

        # Article popularity from training data
        article_popularity = (
            self.train_click_df["click_article_id"].value_counts().reset_index()
        )
        article_popularity.columns = ["article_id", "popularity"]

        # Normalize popularity
        scaler = MinMaxScaler()
        article_popularity["article_popularity"] = scaler.fit_transform(
            article_popularity[["popularity"]]
        )

        # Create item features dictionary
        for _, row in tqdm(self.article_info_df.iterrows()):
            item_id = str(row["click_article_id"])
            popularity = (
                article_popularity[
                    article_popularity["article_id"] == row["click_article_id"]
                ]["article_popularity"].iloc[0]
                if len(
                    article_popularity[
                        article_popularity["article_id"] == row["click_article_id"]
                    ]
                )
                > 0
                else 0.0
            )

            self.item_features_dict[item_id] = {
                "category_id": int(row["category_id"]),
                "article_popularity": float(popularity),
                "created_at_ts": int(row["created_at_ts"]),
                "words_count": int(row["words_count"]),
            }

        # Define item feature names
        self.item_features = [
            "category_id",
            "article_popularity",
            "created_at_ts",
            "words_count",
        ]

        print(f"Item features extracted for {len(self.item_features_dict)} items")

    def _extract_context_features(self):
        """
        Extract context/interaction features with highly optimized numpy operations

        Major optimizations:
        1. Pre-allocate numpy arrays for all features (avoid repeated DataFrame.loc)
        2. Build dense embedding matrices with integer indexing
        3. Eliminate groupby loops with vectorized batch operations
        4. Single DataFrame assignment at the end (20-60x faster than per-user assignment)

        Performance: From 2-8 hours -> 5-15 minutes on large datasets
        """
        print("Extracting context features (optimized)...")

        # Get user history for similarity calculations (offline mode: exclude last click for recall)
        train_history, _ = InteractionFeatureExtractor.get_hist_and_last_click(
            self.train_click_df, offline=self.config.offline
        )

        # Create user history dictionary
        print("Building user history dictionary...")
        for user_id, group in train_history.groupby("user_id"):
            self.user_history_dict[str(user_id)] = (
                group["click_article_id"].astype(str).tolist()
            )

        # Get article ID embeddings for similarity calculation
        article_id_emb_dict = self._get_article_id_embeddings()

        # Initialize context feature column names
        context_cols = ["score"]  # start with recall score

        # Add similarity features for last N items
        N = self.config.last_N
        for i in range(1, N + 1):
            context_cols.extend([f"sim_{i}", f"time_diff_{i}", f"word_diff_{i}"])

        # Add similarity statistics
        context_cols.extend(["sim_max", "sim_mean", "sim_min", "sim_std"])

        # Add other context features
        context_cols.extend(["item_user_sim", "recall_in_user_cat"])

        # Pre-allocate numpy feature arrays (KEY OPTIMIZATION: avoid DataFrame.loc)
        num_rows = len(self.main_df)
        print(f"Pre-allocating feature arrays for {num_rows} rows...")

        # Use NaN for similarity (0 could be a valid similarity score)
        sim_features = np.full((num_rows, N), np.nan, dtype=np.float32)
        # Use 0 for time_diff and word_diff (missing data = no difference)
        time_diff_features = np.zeros((num_rows, N), dtype=np.float32)
        word_diff_features = np.zeros((num_rows, N), dtype=np.float32)

        sim_stats = np.full(
            (num_rows, 4), np.nan, dtype=np.float32
        )  # max, mean, min, std
        item_user_sim = np.zeros(num_rows, dtype=np.float32)
        recall_in_user_cat = np.zeros(num_rows, dtype=np.int8)

        # Pre-compute user category sets
        print("Pre-computing user categories...")
        user_categories_dict = {}
        for user_id_str, history in self.user_history_dict.items():
            user_cats = set()
            for hist_item in history:
                if hist_item in self.article_type_dict:
                    user_cats.add(self.article_type_dict[hist_item])
            user_categories_dict[user_id_str] = user_cats

        # Cache for embedding zeros (avoid repeated creation)
        zero_id_emb = np.zeros(self.config.embedding_dim, dtype=np.float32)
        zero_content_emb = np.zeros(250, dtype=np.float32)

        # Build user-to-row mapping for fast indexing
        print("Building user-to-row index...")
        user_row_map = {}
        for user_id, group in self.main_df.groupby("user_id"):
            user_id_str = str(user_id)
            # Store start and end indices for each user
            user_row_map[user_id_str] = (
                group.index.min(),
                group.index.max() + 1,
                group.index.values,
            )

        # Calculate context features (vectorized by user)
        print("Calculating context features with vectorized operations...")
        for user_i, (user_id_str, (start_idx, end_idx, row_indices)) in tqdm(
            enumerate(user_row_map.items()), desc="Processing users"
        ):
            if user_id_str not in self.user_history_dict:
                continue

            history = self.user_history_dict[user_id_str][-N:]  # last N items
            recall_items = self.main_df.loc[row_indices, "item_id"].values  # type: ignore
            num_recall = len(recall_items)

            # Calculate item-user similarity (if embeddings available)
            if hasattr(self, "user_youtubednn_emb_dict") and hasattr(
                self, "article_youtubednn_emb_dict"
            ):
                user_emb = self.user_youtubednn_emb_dict.get(user_id_str)
                if user_i < 3:
                    print(
                        f"""
                          CHECKPOINT:
                          user_emb for user_id {user_id_str} with shape {user_emb.shape if user_emb is not None else 'None'}
                          sample: {user_emb[:5] if user_emb is not None else 'None'}
                          """
                    )
                if user_emb is not None:
                    # Batch lookup: use list comprehension once, avoid repeated get()
                    item_embs = np.array(
                        [
                            self.article_youtubednn_emb_dict.get(item_id, zero_id_emb)
                            for item_id in recall_items
                        ],
                        dtype=np.float32,
                    )
                    # Vectorized dot product
                    item_user_sim[row_indices] = item_embs @ user_emb

                if user_i < 3:
                    print(
                        f"""
                          CHECKPOINT:
                          item_user_sim for user_id {user_id_str} with shape {item_user_sim[row_indices].shape}
                          sample: {item_user_sim[row_indices][:5]}
                          """
                    )

            # Batch lookup recall item embeddings and metadata (optimization: single pass)
            recall_id_embs = np.array(
                [
                    article_id_emb_dict.get(item_id, zero_id_emb)
                    for item_id in recall_items
                ],
                dtype=np.float32,
            )  # (num_recall, emb_dim)

            recall_content_embs = np.array(
                [
                    self.article_content_emb_dict.get(item_id, zero_content_emb)
                    for item_id in recall_items
                ],
                dtype=np.float32,
            )  # (num_recall, 250)

            recall_times = np.array(
                [
                    self.article_created_time_dict.get(item_id, np.nan)
                    for item_id in recall_items
                ],
                dtype=np.float32,
            )  # (num_recall,)

            if user_i < 3:
                print(
                    f"""
                      CHECKPOINT:
                      recall_times for user_id {user_id_str} with shape {recall_times.shape}
                      sample: {recall_times[:5]}
                      """
                )

            # Calculate similarity features for each history item
            for hist_idx, hist_item in enumerate(history):
                # Get history item embeddings and metadata
                hist_id_emb = article_id_emb_dict.get(hist_item)
                hist_content_emb = self.article_content_emb_dict.get(hist_item)
                hist_time = self.article_created_time_dict.get(hist_item, np.nan)

                if user_i < 3 and hist_idx < 3:
                    print(
                        f"""
                        CHECKPOINT:
                        hist_time for hist_idx {hist_idx}
                        sample: {hist_time}
                        """
                    )

                # Vectorized similarity calculation
                if hist_id_emb is not None:
                    sim_array = recall_id_embs @ hist_id_emb
                else:
                    sim_array = np.zeros(num_recall, dtype=np.float32)

                # Vectorized time difference (improved: use 0 for missing instead of NaN)
                if not np.isnan(hist_time):
                    time_diff_array = np.abs(recall_times - hist_time)
                    # Replace NaN in recall_times with a large value (e.g., max time diff)
                    time_diff_array = np.where(
                        np.isnan(time_diff_array),
                        0,  # Use 0 for missing time data instead of NaN
                        time_diff_array,
                    )
                else:
                    # If history item has no time, use 0 as default
                    time_diff_array = np.zeros(num_recall, dtype=np.float32)

                # Vectorized content embedding distance (improved: use 0 for missing)
                if hist_content_emb is not None:
                    word_diff_array = np.linalg.norm(
                        recall_content_embs - hist_content_emb[np.newaxis, :], axis=1
                    )
                    # Check if recall items have valid embeddings
                    # If recall item embedding is zero (missing), set distance to 0
                    recall_emb_valid = np.any(recall_content_embs != 0, axis=1)
                    word_diff_array = np.where(
                        recall_emb_valid,
                        word_diff_array,
                        0,  # Use 0 for missing embeddings instead of NaN
                    )
                else:
                    # If history item has no embedding, use 0 as default
                    word_diff_array = np.zeros(num_recall, dtype=np.float32)

                # Direct numpy array assignment (KEY: no DataFrame.loc!)
                sim_features[row_indices, hist_idx] = sim_array
                if user_i < 3 and hist_idx < 3:
                    print(
                        f"""
                          CHECKPOINT:
                          time_diff_array for hist_idx {hist_idx} with shape {time_diff_array.shape}
                          sample: {time_diff_array[:5]}
                          """
                    )
                time_diff_features[row_indices, hist_idx] = time_diff_array
                word_diff_features[row_indices, hist_idx] = word_diff_array

            # Calculate similarity statistics (vectorized)
            user_sims = sim_features[row_indices, :]
            sim_stats[row_indices, 0] = np.nanmax(user_sims, axis=1)  # max
            sim_stats[row_indices, 1] = np.nanmean(user_sims, axis=1)  # mean
            sim_stats[row_indices, 2] = np.nanmin(user_sims, axis=1)  # min
            sim_stats[row_indices, 3] = np.nanstd(user_sims, axis=1)  # std

            # Calculate recall_in_user_cat (vectorized with pre-computed categories)
            user_categories = user_categories_dict.get(user_id_str, set())
            if user_categories:
                recall_in_user_cat[row_indices] = np.array(
                    [
                        (
                            1
                            if self.article_type_dict.get(item_id) in user_categories
                            else 0
                        )
                        for item_id in recall_items
                    ],
                    dtype=np.int8,
                )

        print(
            f"""
              CHECKPOINT:
              time_diff_features with {time_diff_features.shape[0]} rows and {time_diff_features.shape[1]} cols
              sample: {time_diff_features[:5]}
              """
        )

        # Batch assign all features to DataFrame (SINGLE operation, not per-user!)
        print("Assigning features to DataFrame (batch operation)...")

        # Assign similarity features
        for i in range(N):
            self.main_df[f"sim_{i+1}"] = sim_features[:, i]
            self.main_df[f"time_diff_{i+1}"] = time_diff_features[:, i]
            self.main_df[f"word_diff_{i+1}"] = word_diff_features[:, i]

        # Assign statistics
        self.main_df["sim_max"] = sim_stats[:, 0]
        self.main_df["sim_mean"] = sim_stats[:, 1]
        self.main_df["sim_min"] = sim_stats[:, 2]
        self.main_df["sim_std"] = sim_stats[:, 3]

        # Assign other features
        self.main_df["item_user_sim"] = item_user_sim
        self.main_df["recall_in_user_cat"] = recall_in_user_cat

        # Define context feature names
        self.context_features = context_cols

        print(f"Context features extracted: {len(self.context_features)} features")
        print(f"Feature columns: {self.context_features}")

    def _get_article_id_embeddings(self):
        """Get article ID embeddings using Word2Vec"""
        print("Generating article ID embeddings...")

        click_df = self.train_click_df.sort_values(by="click_timestamp").copy()
        click_df["click_article_id"] = click_df["click_article_id"].astype(str)

        user_click_seq = (
            click_df.groupby("user_id")["click_article_id"].apply(list).tolist()
        )

        model = Word2Vec(
            sentences=user_click_seq,
            vector_size=self.config.embedding_dim,
            window=5,
            min_count=1,
            workers=4,
            sg=1,
            epochs=10,
        )

        article_ids = list(model.wv.index_to_key)
        article_embs = model.wv[article_ids]
        article_id_emb_dict = dict(zip(article_ids, article_embs))

        return article_id_emb_dict

    def _add_labels(self):
        """
        Add labels to main dataframe based on offline/online mode

        Offline mode (training): Match recalled items with user's last click (ground truth)
        Online mode (evaluation): No labels needed (all samples are for prediction)
        """
        print("Adding labels...")

        if not self.config.offline:
            # Online evaluation mode: no labels needed, all samples for prediction
            print("Online mode: skipping label generation (prediction mode)")
            self.main_df["label"] = -1  # placeholder for compatibility
            return

        # Offline training mode: generate labels by matching with ground truth
        # Get last click for each user as ground truth (excluded from recall history)
        _, train_last_click = InteractionFeatureExtractor.get_hist_and_last_click(
            self.train_click_df, offline=True
        )

        if train_last_click.empty:
            print("Warning: No ground truth clicks found")
            self.main_df["label"] = 0
            return

        # Extract ground truth item IDs
        train_last_click = train_last_click[["user_id", "click_article_id"]].copy()
        train_last_click.columns = ["user_id", "last_item_id"]

        # Ensure consistent formats for joining
        train_last_click["user_id"] = train_last_click["user_id"].astype(str)
        train_last_click["last_item_id"] = train_last_click["last_item_id"].astype(str)

        # Convert float strings to integer strings to ensure consistent format
        train_last_click["user_id"] = train_last_click["user_id"].apply(
            lambda x: (
                str(int(float(x)))
                if isinstance(x, str) and "." in x
                else str(int(x)) if isinstance(x, (int, float)) else x
            )
        )
        train_last_click["last_item_id"] = train_last_click["last_item_id"].apply(
            lambda x: (
                str(int(float(x)))
                if isinstance(x, str) and "." in x
                else str(int(x)) if isinstance(x, (int, float)) else x
            )
        )

        # Print sample of user_id and last_item_id for debugging
        print(
            f"Train last click sample - user_id: {train_last_click['user_id'].iloc[0] if len(train_last_click) > 0 else 'empty'}"
        )
        print(
            f"Train last click sample - last_item_id: {train_last_click['last_item_id'].iloc[0] if len(train_last_click) > 0 else 'empty'}"
        )
        print(
            f"Main df sample - user_id: {self.main_df['user_id'].iloc[0] if len(self.main_df) > 0 else 'empty'}"
        )
        print(
            f"Main df sample - item_id: {self.main_df['item_id'].iloc[0] if len(self.main_df) > 0 else 'empty'}"
        )

        # Merge with main dataframe to get labels
        self.main_df = pd.merge(
            self.main_df,
            train_last_click,
            left_on=["user_id", "item_id"],
            right_on=["user_id", "last_item_id"],
            how="left",
            indicator=True,
        )

        # Create labels: 1 for positive samples (recalled item = ground truth), 0 for negative
        self.main_df["label"] = (self.main_df["_merge"] == "both").astype(int)

        # Remove helper columns
        self.main_df.drop(columns=["last_item_id", "_merge"], inplace=True)

        print(f"Labels added. Positive samples: {(self.main_df['label'] == 1).sum()}")
        print(f"Negative samples: {(self.main_df['label'] == 0).sum()}")
        print(
            f"Label distribution: {(self.main_df['label'] == 1).sum() / len(self.main_df) * 100:.4f}% positive"
        )

    def _apply_binning(self):
        """Apply binning to continuous features"""
        if not self.config.enable_binning:
            return

        print("Applying binning to continuous features...")

        # Identify continuous features (numeric with >20 unique values)
        continuous_features = []
        for col in self.main_df.columns:
            if col in ["user_id", "item_id", "label"]:
                continue
            if pd.api.types.is_numeric_dtype(self.main_df[col]):
                if self.main_df[col].nunique() > 20:
                    continuous_features.append(col)

        # Apply binning to continuous features
        for feature in continuous_features:
            if feature not in self.main_df.columns:
                continue

            # Handle missing values with better strategy
            # First check if the column is all NaN
            if self.main_df[feature].isna().all():
                print(f"Warning: Feature '{feature}' is all NaN, filling with 0")
                self.main_df[feature] = 0
                continue

            # Fill NaN with median (or 0 if median is also NaN)
            median_val = self.main_df[feature].median()
            fill_value = 0 if pd.isna(median_val) else median_val
            self.main_df[feature] = self.main_df[feature].fillna(fill_value)

            # Determine number of bins
            n_unique = self.main_df[feature].nunique()
            n_bins = min(self.config.default_n_bins, n_unique)

            if n_bins < 2:
                print(
                    f"Skipping binning for '{feature}': only {n_unique} unique value(s)"
                )
                continue

            # Create discretizer
            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy=self.config.binning_strategy,  # type: ignore
            )

            try:
                # Fit and transform
                binned = discretizer.fit_transform(self.main_df[[feature]])
                self.main_df[feature] = np.asarray(binned).astype(int).flatten()
                self.discretizers[feature] = discretizer
            except Exception as e:
                print(f"Error binning {feature}: {e}")
                # If binning fails, keep the original filled values
                continue

        print(f"Binning applied to {len(self.discretizers)} features")

    def extract_features(self, save: bool = True):
        """Extract all features"""
        print("Starting feature extraction...")

        # Load data
        self.load_all()

        # Extract feature groups
        self._extract_user_profile_features()
        self._extract_item_features()
        self._extract_context_features()

        # Add user profile and item features to main_df
        self._add_features_to_main_df()

        # Add labels
        self._add_labels()

        # Apply binning
        self._apply_binning()

        if save:
            self._save_features()

        print("Feature extraction completed!")

    def _add_features_to_main_df(self):
        """Add user profile and item features to main dataframe"""
        print("Adding user and item features to main dataframe...")

        # Add user profile features
        for feature in self.user_profile_features:
            self.main_df[feature] = self.main_df["user_id"].apply(
                lambda user_id: self.user_profile_dict.get(user_id, {}).get(feature, 0)
            )

        # Add item features
        for feature in self.item_features:
            self.main_df[feature] = self.main_df["item_id"].apply(
                lambda item_id: self.item_features_dict.get(item_id, {}).get(feature, 0)
            )

        # Report feature addition
        print(
            f"Added {len(self.user_profile_features)} user profile features and {len(self.item_features)} item features to main dataframe"
        )
        print(f"Main dataframe now has {len(self.main_df.columns)} columns")

    def _save_features(self):
        """Save extracted features"""
        print("Saving features...")

        # Print feature statistics before saving
        print("\nDataFrame statistics before saving:")
        print(f"Total rows: {len(self.main_df)}")
        print(f"Total columns: {len(self.main_df.columns)}")
        print(f"Columns: {', '.join(self.main_df.columns)}")
        print(f"Positive labels: {(self.main_df['label'] == 1).sum()}")
        print(f"Negative labels: {(self.main_df['label'] == 0).sum()}")
        print(
            f"Missing values in user features: {self.main_df[self.user_profile_features].isna().sum().sum()}"
        )
        print(
            f"Missing values in item features: {self.main_df[self.item_features].isna().sum().sum()}"
        )
        print(
            f"Missing values in context features: {self.main_df[self.context_features].isna().sum().sum() if self.context_features else 0}"
        )

        # Save sample rows for inspection
        print("\nSample row from DataFrame:")
        if len(self.main_df) > 0:
            sample_row = self.main_df.iloc[0]
            for col in self.main_df.columns:
                print(f"{col}: {sample_row[col]}")

        # Save main dataframe
        self.main_df.to_csv(
            os.path.join(self.config.save_path, "main_features.csv"), index=False
        )

        # Save dictionaries
        PersistenceManager.save_pickle(
            self.user_profile_dict,
            os.path.join(self.config.save_path, "user_profile_dict.pkl"),
        )
        PersistenceManager.save_pickle(
            self.item_features_dict,
            os.path.join(self.config.save_path, "item_features_dict.pkl"),
        )
        PersistenceManager.save_pickle(
            self.user_history_dict,
            os.path.join(self.config.save_path, "user_history_dict.pkl"),
        )

        # Save feature lists
        feature_lists = {
            "user_profile_features": self.user_profile_features,
            "item_features": self.item_features,
            "context_features": self.context_features,
        }
        PersistenceManager.save_pickle(
            feature_lists, os.path.join(self.config.save_path, "feature_lists.pkl")
        )

        # Save discretizers
        PersistenceManager.save_pickle(
            self.discretizers, os.path.join(self.config.save_path, "discretizers.pkl")
        )

        print(f"\nFeatures saved to {self.config.save_path}")
        print(
            f"Saved main_features.csv with {len(self.main_df)} rows and {len(self.main_df.columns)} columns"
        )
        print(f"Saved user_profile_dict.pkl with {len(self.user_profile_dict)} users")
        print(f"Saved item_features_dict.pkl with {len(self.item_features_dict)} items")

    def get_data(self):
        """Get extracted data"""
        return {
            "main_df": self.main_df,
            "user_profile_dict": self.user_profile_dict,
            "item_features_dict": self.item_features_dict,
            "user_history_dict": self.user_history_dict,
            "user_profile_features": self.user_profile_features,
            "item_features": self.item_features,
            "context_features": self.context_features,
        }
