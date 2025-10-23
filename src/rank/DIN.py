import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from typing import Optional, List, Literal, Dict
from sklearn.preprocessing import LabelEncoder
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.rank.base import BaseRanker
from src.utils.config import RankConfig
from src.utils.persistence import PersistenceManager


class Dice(nn.Module):
    def __init__(self, alpha: float = 0.1, eps: float = 1e-8) -> None:
        """
        Dice activation function from the original paper of DIN
        """
        super().__init__()
        # CRITICAL FIX: torch.tensor needs explicit dtype for gradients
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        x_normed = (x - mean) / (std + self.eps)
        p = torch.sigmoid(x_normed)
        return p * x + (1 - p) * 0.01 * x


class ActivationUnit(nn.Module):
    def __init__(
        self,
        mlp_input_dim: int,
        hidden_units: list = [36],
        activation: Literal["dice", "prelu"] = "dice",
    ) -> None:
        """
        Activation unit based on attention mechanism

        Args:
            mlp_input_dim (int): should be 4 times original embedding dim
            hidden_units (list, optional): Defaults to [36].
            activation (Literal["dice", "prelu"], optional): Activation method. Defaults to 'dice'.
        """
        super().__init__()
        self.mlp_input_dim = mlp_input_dim
        self.hidden_units = hidden_units
        self.activation = activation

        layers = []
        cur_dim = self.mlp_input_dim
        for unit in self.hidden_units:
            layers.append(nn.Linear(cur_dim, unit))
            if self.activation == "dice":
                layers.append(Dice(unit))
            elif self.activation == "prelu":
                layers.append(nn.PReLU())
            else:
                raise ValueError(f"Unsupported activation: {self.activation}")
            cur_dim = unit

        layers.append(nn.Linear(cur_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        queries: torch.Tensor,  # [B, d]
        keys: torch.Tensor,  # [B, T, d]
        mask: Optional[torch.Tensor] = None,  # [B, T]
        normalize: bool = False,
    ):
        """
        1. concat query, keys, query-keys, query*keys
        2. MLP to get attention scores
        3. Apply mask to attention scores if provided
        4. Optionally normalize with softmax

        Args:
            queries: batch recall items (B, d)
            keys: history items (B, T, d)
            mask: padding mask (B, T), 1 for valid positions, 0 for padding
            normalize: whether to normalize attention weights with softmax

        Returns:
            weight_scores: (B, T, 1) - masked attention scores (normalized if requested)
        """
        B, T, d = keys.shape
        query_expanded = queries.unsqueeze(1).expand(B, T, d)

        minus_emb = query_expanded - keys
        hadamard_emb = query_expanded * keys

        concatenated_embs = torch.cat(
            [keys, query_expanded, minus_emb, hadamard_emb], dim=-1
        )  # (B, T, 4d)

        weight_scores = self.mlp(concatenated_embs)  # (B, T, 1)

        # Apply mask to padded positions if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, T, 1)
            if normalize:
                # For softmax normalization, set padding to large negative value
                weight_scores = weight_scores.masked_fill(mask == 0, -1e9)
            else:
                # For non-normalized weights, set padding positions to zero
                weight_scores = weight_scores * mask

        # Apply softmax normalization if requested
        if normalize:
            weight_scores = torch.softmax(weight_scores, dim=1)

        return weight_scores


class DINModel(nn.Module):
    def __init__(
        self,
        user_profile_vocab_dict: dict,
        item_vocab_dict: dict,
        context_vocab_dict: dict,
        embedding_dim: int = 64,
        attention_hidden_units: list = [36],
        mlp_hidden_units: list = [200, 80, 2],
        activation: Literal["dice", "prelu"] = "dice",
    ) -> None:
        """
        Clean DIN model using new data structure

        Args:
            user_profile_vocab_dict: {feature_name: vocab_size, ...}
            item_vocab_dict: {feature_name: vocab_size, ...}
            context_vocab_dict: {feature_name: vocab_size, ...}
            embedding_dim: embedding dimension
            attention_hidden_units: hidden units for attention MLP
            mlp_hidden_units: hidden units for final MLP
            activation: activation function
        """
        super().__init__()

        # Embedding layers
        self.user_profile_embedding_dict = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_size, embedding_dim)
                for feature, vocab_size in user_profile_vocab_dict.items()
            }
        )

        self.item_embedding_dict = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_size, embedding_dim)
                for feature, vocab_size in item_vocab_dict.items()
            }
        )

        self.context_embedding_dict = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_size, embedding_dim)
                for feature, vocab_size in context_vocab_dict.items()
            }
        )

        # Attention unit
        item_feature_len = len(item_vocab_dict)
        self.item_dim = item_feature_len * embedding_dim
        self.activation_unit = ActivationUnit(4 * self.item_dim)

        # Final MLP
        self.user_profile_dim = len(user_profile_vocab_dict) * embedding_dim
        self.context_dim = len(context_vocab_dict) * embedding_dim
        self.mlp_input_dim = (
            self.user_profile_dim + self.context_dim + 2 * self.item_dim
        )

        print(f"MLP input dim: {self.mlp_input_dim}")

        cur_dim = self.mlp_input_dim
        mlp_layers = []
        for unit in mlp_hidden_units:
            mlp_layers.append(nn.Linear(cur_dim, unit))
            if activation == "dice":
                mlp_layers.append(Dice(unit))
            elif activation == "prelu":
                mlp_layers.append(nn.PReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            cur_dim = unit

        mlp_layers.append(nn.Linear(cur_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, batch: dict):
        """
        Forward pass for DIN model.

        Args:
            batch: dictionary with keys:
                - 'user_profile': {feature_name: tensor[B]}
                - 'recall_item': {feature_name: tensor[B]}
                - 'history_items': {feature_name: tensor[B, T]}
                - 'context': {feature_name: tensor[B]}
                - 'history_mask': tensor[B, T] (optional, 1 for valid, 0 for padding)

        Returns:
            probs: (B, ) - predicted click probabilities
        """
        user_profile_dict = batch["user_profile"]
        history_item_dict = batch["history_items"]
        recall_item_dict = batch["recall_item"]
        context_dict = batch["context"]
        history_mask = batch.get("history_mask", None)  # Get mask if available

        # Get embeddings
        user_profile_embs = torch.cat(
            [
                self.user_profile_embedding_dict[feature](indices)
                for feature, indices in user_profile_dict.items()
            ],
            dim=1,
        )  # (B, user_profile_dim)

        history_embs = torch.cat(
            [
                self.item_embedding_dict[feature](indices)  # (B, T, single_dim)
                for feature, indices in history_item_dict.items()  # indices: (B, T)
            ],
            dim=2,
        )  # (B, T, item_dim)

        recall_embs = torch.cat(
            [
                self.item_embedding_dict[feature](indices)
                for feature, indices in recall_item_dict.items()
            ],
            dim=1,
        )  # (B, item_dim)

        context_embs = torch.cat(
            [
                self.context_embedding_dict[feature](indices)
                for feature, indices in context_dict.items()
            ],
            dim=1,
        )  # (B, context_dim)

        # Attention mechanism with mask
        # Keep original behavior: don't normalize by default
        weight_scores = self.activation_unit(
            queries=recall_embs, keys=history_embs, mask=history_mask, normalize=False
        )  # (B, T, 1)

        # Weighted history - use raw attention weights as designed in original code
        # This maintains the intended behavior where attention weights reflect scale
        weighted_history = (weight_scores * history_embs).sum(dim=1)  # (B, item_dim)

        # Final MLP
        mlp_input = torch.cat(
            [user_profile_embs, context_embs, recall_embs, weighted_history], dim=1
        )  # (B, mlp_input_dim)

        logits = self.mlp(mlp_input)  # (B, 1)
        probs = torch.sigmoid(logits).squeeze(-1)  # (B, )

        return probs


class DINDataset(Dataset):
    def __init__(
        self,
        main_df: pd.DataFrame,
        user_profile_dict: Dict[str, Dict],
        item_features_dict: Dict[str, Dict],
        user_history_dict: Dict[str, List[str]],
        user_profile_features: List[str],
        item_features: List[str],
        context_features: List[str],
        label_col: str = "label",
        label_encoders: Optional[Dict] = None,
    ) -> None:
        """
        Clean dataset for DIN model.

        Args:
            main_df: DataFrame with (user_id, item_id, context_features..., label)
            user_profile_dict: {user_id: {feature: value}}
            item_features_dict: {item_id: {feature: value}}
            user_history_dict: {user_id: [item_id, ...]}
            user_profile_features: list of user profile feature names
            item_features: list of item feature names
            context_features: list of context feature names
            label_col: name of the label column
            label_encoders: dictionary of LabelEncoder objects for each feature
        """
        self.main_df = main_df.reset_index(drop=True)
        self.user_profile_dict = user_profile_dict
        self.item_features_dict = item_features_dict
        self.user_history_dict = user_history_dict
        self.user_profile_features = user_profile_features
        self.item_features = item_features
        self.context_features = context_features
        self.label_col = label_col
        self.label_encoders = label_encoders or {}

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        """
        Returns a dictionary of one (user_id, recall_item_id) sample with:
        - user_profile: {feature_name: encoded_value}
        - recall_item: {feature_name: encoded_value}
        - history_items: list of {feature_name: encoded_value}
        - context: {feature_name: encoded_value}
        - label: int

        NOTE: We use encoder here
        """
        row = self.main_df.iloc[idx]
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])

        def encode_feature(feat_name, raw_value):
            if not self.label_encoders or feat_name not in self.label_encoders:
                return raw_value

            try:
                # Convert to string to handle all value types
                str_value = str(raw_value)
                # Use transform and handle potential errors for unseen values
                transformed = self.label_encoders[feat_name].transform([str_value])
                # Get first value as integer and add 1 (0 reserved for padding)
                encoded_val = int(transformed[0]) + 1
                return encoded_val
            except:
                # Default to 0 for unknown values
                return 0

        # Extract user profile features
        user_profile = {}
        if user_id in self.user_profile_dict:
            for feat in self.user_profile_features:
                raw_value = self.user_profile_dict[user_id].get(feat, 0)
                user_profile[feat] = encode_feature(feat, raw_value)
        else:
            user_profile = {feat: 0 for feat in self.user_profile_features}

        # Extract recall item features
        recall_item = {}
        if item_id in self.item_features_dict:
            for feat in self.item_features:
                raw_value = self.item_features_dict[item_id].get(feat, 0)
                recall_item[feat] = encode_feature(feat, raw_value)
        else:
            recall_item = {feat: 0 for feat in self.item_features}

        # Extract history items
        history_items = []
        if user_id in self.user_history_dict:
            for hist_item_id in self.user_history_dict[user_id]:
                hist_item = {}
                if hist_item_id in self.item_features_dict:
                    for feat in self.item_features:
                        raw_value = self.item_features_dict[hist_item_id].get(feat, 0)
                        hist_item[feat] = encode_feature(feat, raw_value)
                else:
                    hist_item = {feat: 0 for feat in self.item_features}
                history_items.append(hist_item)

        # Extract context features
        context = {}
        for feat in self.context_features:
            raw_value = row.get(feat, 0)
            context[feat] = encode_feature(feat, raw_value)

        # Extract label
        label = int(row[self.label_col]) if self.label_col in row else -1

        return {
            "user_profile": user_profile,
            "recall_item": recall_item,
            "history_items": history_items,
            "context": context,
            "label": label,
        }


def collate_fn(batch, seq_max_len):
    """
    Clean collate function to handle variable-length sequences.
    Pads/truncates history sequences to seq_max_len.

    Args:
        batch: List of dictionaries from CleanDINDataset
        seq_max_len: Maximum sequence length to pad/truncate to

    Returns:
        A dictionary of batched tensors ready for CleanDINModel forward
        {
            'user_profile': {feature_name: tensor[B]},
            'recall_item': {feature_name: tensor[B]},
            'history_items': {feature_name: tensor[B, T]},
            'context': {feature_name: tensor[B]},
            'history_mask': tensor[B, T], (1 for valid positions, 0 for padding)
            'labels': tensor[B]
        }
    """
    batch_size = len(batch)

    # Collect all feature names
    user_profile_features = list(batch[0]["user_profile"].keys())
    item_features = list(batch[0]["recall_item"].keys())
    context_features = list(batch[0]["context"].keys())

    # Initialize batch dictionaries
    user_profile_batch = {feat: [] for feat in user_profile_features}
    recall_item_batch = {feat: [] for feat in item_features}
    history_items_batch = {feat: [] for feat in item_features}
    context_batch = {feat: [] for feat in context_features}
    labels_batch = []
    history_mask_batch = []  # Track valid positions

    for sample in batch:
        # User profile features
        for feat in user_profile_features:
            user_profile_batch[feat].append(sample["user_profile"][feat])

        # Recall item features
        for feat in item_features:
            recall_item_batch[feat].append(sample["recall_item"][feat])

        # History items features - pad or truncate
        history_items = sample["history_items"]
        history_len = len(history_items)

        # Truncate if too long
        if history_len > seq_max_len:
            history_items = history_items[-seq_max_len:]
            history_len = seq_max_len

        # Create mask: 1 for valid positions, 0 for padding
        mask = [1] * history_len + [0] * (seq_max_len - history_len)
        history_mask_batch.append(mask)

        # Pad if too short
        for feat in item_features:
            feat_values = [item[feat] for item in history_items]
            if history_len < seq_max_len:
                feat_values = feat_values + [0] * (seq_max_len - history_len)
            history_items_batch[feat].append(feat_values)

        # Context features
        for feat in context_features:
            context_batch[feat].append(sample["context"][feat])

        # Label
        labels_batch.append(sample["label"])

    # Convert to tensors
    for feat in user_profile_features:
        user_profile_batch[feat] = torch.tensor(  # type: ignore
            user_profile_batch[feat], dtype=torch.long
        )

    for feat in item_features:
        recall_item_batch[feat] = torch.tensor(  # type: ignore
            recall_item_batch[feat], dtype=torch.long
        )
        history_items_batch[feat] = torch.tensor(  # type: ignore
            history_items_batch[feat], dtype=torch.long
        )

    for feat in context_features:
        context_batch[feat] = torch.tensor(context_batch[feat], dtype=torch.long)  # type: ignore

    labels_batch = torch.tensor(labels_batch, dtype=torch.float32)
    history_mask_batch = torch.tensor(history_mask_batch, dtype=torch.float32)  # (B, T)

    return {
        "user_profile": user_profile_batch,
        "recall_item": recall_item_batch,
        "history_items": history_items_batch,
        "context": context_batch,
        "history_mask": history_mask_batch,  # Add mask to batch
        "labels": labels_batch,
    }


class DINRanker(BaseRanker):
    def __init__(self, config: RankConfig):
        super().__init__(config)
        self.model = None
        self.label_encoders = {}  # store label encoders for each feature

    def load(self):
        """Load data using new clean structure"""
        print("Loading clean feature data...")

        # Load main dataframe
        self.main_df = pd.read_csv(self.config.main_features_path)

        # Load dictionaries
        self.user_profile_dict = PersistenceManager.load_pickle(
            self.config.user_profile_dict_path
        )
        self.item_features_dict = PersistenceManager.load_pickle(
            self.config.item_features_dict_path
        )
        self.user_history_dict = PersistenceManager.load_pickle(
            self.config.user_history_dict_path
        )

        # Load feature lists
        feature_lists = PersistenceManager.load_pickle(self.config.feature_lists_path)
        self.user_profile_features = feature_lists["user_profile_features"]
        self.item_features = feature_lists["item_features"]
        self.context_features = feature_lists["context_features"]

        # # Load discretizers
        # self.discretizers = PersistenceManager.load_pickle(
        #     self.config.discretizers_path
        # )

        print("feature data loaded.")

    def _prepare_vocab_dicts(self):
        """
        Prepare vocabulary dictionaries for each feature group.

        [PS]: Maybe this can be done during feature engineering when binning?
        """
        print("Preparing vocabulary dictionaries...")

        # Combine all data for consistent encoding
        all_user_profiles = list(self.user_profile_dict.values())
        all_item_features = list(self.item_features_dict.values())
        # CRITICAL FIX: Create explicit copy to avoid SettingWithCopyWarning
        all_context_data = self.main_df[self.context_features].copy()

        # Prepare vocab dicts
        user_profile_vocab_dict = {}
        item_vocab_dict = {}
        context_vocab_dict = {}

        # Encode user profile features
        for feat in self.user_profile_features:
            all_values = set()
            for user_profile in all_user_profiles:
                if feat in user_profile:
                    all_values.add(user_profile[feat])

            if all_values:
                le = LabelEncoder()
                le.fit(list(all_values))
                self.label_encoders[feat] = le
                user_profile_vocab_dict[feat] = len(le.classes_) + 1

        # Encode item features
        for feat in self.item_features:
            all_values = set()
            for item_features in all_item_features:
                if feat in item_features:
                    all_values.add(item_features[feat])

            if all_values:
                le = LabelEncoder()
                le.fit(list(all_values))
                self.label_encoders[feat] = le
                item_vocab_dict[feat] = len(le.classes_) + 1

        # Encode context features
        for feat in self.context_features:
            if feat in all_context_data.columns:
                le = LabelEncoder()
                # Now safe to modify because all_context_data is a copy
                all_context_data[feat] = all_context_data[feat].fillna(0)
                le.fit(all_context_data[feat].astype(str))
                self.label_encoders[feat] = le
                context_vocab_dict[feat] = len(le.classes_) + 1

        print(f"User profile vocab: {user_profile_vocab_dict}")
        print(f"Item vocab: {item_vocab_dict}")
        print(f"Context vocab: {context_vocab_dict}")

        return user_profile_vocab_dict, item_vocab_dict, context_vocab_dict

    def _apply_negative_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply negative sampling to balance positive and negative samples.

        Args:
            df: DataFrame with 'label' column

        Returns:
            Sampled DataFrame with reduced negative samples
        """
        if "label" not in df.columns:
            print("Warning: No 'label' column found, skipping negative sampling")
            return df

        pos_samples = df[df["label"] == 1]
        neg_samples = df[df["label"] == 0]

        n_pos = len(pos_samples)
        n_neg = len(neg_samples)

        print(f"\n=== Negative Sampling ===")
        print(f"Original distribution:")
        print(f"  - Positive samples: {n_pos} ({n_pos/(n_pos+n_neg)*100:.2f}%)")
        print(f"  - Negative samples: {n_neg} ({n_neg/(n_pos+n_neg)*100:.2f}%)")
        print(f"  - Original ratio (neg:pos): {n_neg/n_pos:.2f}:1")

        if n_pos == 0:
            print("Warning: No positive samples found, returning original data")
            return df

        # Calculate target number of negative samples
        target_neg_count = int(n_pos * self.config.negative_positive_ratio)

        if target_neg_count >= n_neg:
            print(
                f"Note: Target negative samples ({target_neg_count}) >= available ({n_neg})"
            )
            print("Using all negative samples (no downsampling needed)")
            return df

        # Perform random undersampling on negative samples
        np.random.seed(self.config.random_seed)
        sampled_neg_samples = neg_samples.sample(
            n=target_neg_count, random_state=self.config.random_seed
        )

        # Combine positive and sampled negative samples
        balanced_df = pd.concat([pos_samples, sampled_neg_samples], ignore_index=True)

        # Shuffle to mix positive and negative samples
        balanced_df = balanced_df.sample(
            frac=1, random_state=self.config.random_seed
        ).reset_index(drop=True)

        n_sampled_neg = len(sampled_neg_samples)
        print(f"\nAfter negative sampling:")
        print(f"  - Positive samples: {n_pos} ({n_pos/(n_pos+n_sampled_neg)*100:.2f}%)")
        print(
            f"  - Negative samples: {n_sampled_neg} ({n_sampled_neg/(n_pos+n_sampled_neg)*100:.2f}%)"
        )
        print(f"  - New ratio (neg:pos): {n_sampled_neg/n_pos:.2f}:1")
        print(
            f"  - Reduction: {n_neg} -> {n_sampled_neg} ({n_sampled_neg/n_neg*100:.1f}% kept)"
        )
        print(f"  - Total samples: {len(balanced_df)} (was {len(df)})")
        print("=" * 50)

        return balanced_df

    def train(self):
        """
        Train the clean DIN model.
        """
        # Load data
        self.load()

        # Split train/val/test data
        # IMPORTANT: Only train and val sets have labels (from excluded last click)
        # Test set (testA users) has NO labels - used only for final prediction
        if "is_train" in self.main_df.columns and "is_val" in self.main_df.columns:
            train_df = self.main_df[self.main_df["is_train"] == True].copy()
            val_df = self.main_df[self.main_df["is_val"] == True].copy()
            test_df = self.main_df[self.main_df["is_test"] == True].copy()

            print(f"\n=== Data Split ===")
            print(f"Total samples: {len(self.main_df)}")
            print(
                f"Train samples: {len(train_df)} ({len(train_df)/len(self.main_df)*100:.1f}%)"
            )
            print(
                f"Validation samples: {len(val_df)} ({len(val_df)/len(self.main_df)*100:.1f}%)"
            )
            print(
                f"Test samples (no labels): {len(test_df)} ({len(test_df)/len(self.main_df)*100:.1f}%)"
            )

            # Check if we have labels in train/val sets
            if "label" in train_df.columns:
                train_pos = (train_df["label"] == 1).sum()
                train_neg = (train_df["label"] == 0).sum()
                print(f"\nTrain set label distribution:")
                print(f"  - Positive: {train_pos} ({train_pos/len(train_df)*100:.2f}%)")
                print(f"  - Negative: {train_neg} ({train_neg/len(train_df)*100:.2f}%)")

                if len(val_df) > 0 and "label" in val_df.columns:
                    val_pos = (val_df["label"] == 1).sum()
                    val_neg = (val_df["label"] == 0).sum()
                    print(f"\nValidation set label distribution:")
                    print(f"  - Positive: {val_pos} ({val_pos/len(val_df)*100:.2f}%)")
                    print(f"  - Negative: {val_neg} ({val_neg/len(val_df)*100:.2f}%)")

                # Note: Test set should have no valid labels
                print(
                    f"\nNote: Test set ({len(test_df)} samples) has NO ground truth labels"
                )
                print("Test set is used only for final prediction/submission")

            # Use train_df for training, val_df for validation
            train_data = train_df
            val_data = val_df

            # Apply negative sampling to training data if enabled
            if self.config.enable_negative_sampling and "label" in train_data.columns:
                train_data = self._apply_negative_sampling(train_data)

            print(f"\nUsing {len(train_data)} samples for training (after sampling)")
            print(f"Using {len(val_data)} samples for validation")
        else:
            print("\nWarning: 'is_train' or 'is_val' column not found")
            print("Falling back to simple 80/20 split")
            # Fallback: simple split
            train_size = int(len(self.main_df) * 0.8)
            train_data = self.main_df.iloc[:train_size].copy()
            val_data = self.main_df.iloc[train_size:].copy()
            test_df = None

            # Apply negative sampling if enabled
            if self.config.enable_negative_sampling and "label" in train_data.columns:
                train_data = self._apply_negative_sampling(train_data)
            train_data = self.main_df.iloc[:train_size].copy()
            val_data = self.main_df.iloc[train_size:].copy()
            test_df = None

        # Prepare vocabulary dictionaries
        user_profile_vocab, item_vocab, context_vocab = self._prepare_vocab_dicts()

        # Prepare vocabulary dictionaries
        user_profile_vocab, item_vocab, context_vocab = self._prepare_vocab_dicts()

        # Initialize model
        self.model = DINModel(
            user_profile_vocab_dict=user_profile_vocab,
            item_vocab_dict=item_vocab,
            context_vocab_dict=context_vocab,
            embedding_dim=self.config.din_embedding_dim,
            attention_hidden_units=self.config.din_attention_hidden_units,
            mlp_hidden_units=self.config.din_mlp_hidden_units,
            activation=self.config.din_activation,  # type: ignore
        )

        # Create dataset using only train data
        train_dataset = DINDataset(
            main_df=train_data,  # Use train_data instead of self.main_df
            user_profile_dict=self.user_profile_dict,
            item_features_dict=self.item_features_dict,
            user_history_dict=self.user_history_dict,
            user_profile_features=self.user_profile_features,
            item_features=self.item_features,
            context_features=self.context_features,
            label_col="label",
            label_encoders=self.label_encoders,  # Pass encoders to dataset
        )

        # Create validation dataset using val_data (NOT test_df!)
        # test_df has no labels and should NOT be used for validation
        val_dataset = None
        val_loader = None
        if val_data is not None and len(val_data) > 0:
            val_dataset = DINDataset(
                main_df=val_data,  # Use val_data, not test_df
                user_profile_dict=self.user_profile_dict,
                item_features_dict=self.item_features_dict,
                user_history_dict=self.user_history_dict,
                user_profile_features=self.user_profile_features,
                item_features=self.item_features,
                context_features=self.context_features,
                label_col="label",
                label_encoders=self.label_encoders,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=partial(collate_fn, seq_max_len=self.config.din_seq_max_len),
            )

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, seq_max_len=self.config.din_seq_max_len),
        )

        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()

        print(f"Training on device: {device}")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of batches: {len(train_loader)}")
        if val_loader is not None and val_dataset is not None:
            print(f"Number of validation samples: {len(val_dataset)}")
            print(f"Number of validation batches: {len(val_loader)}")

        # Initialize loss tracking
        self.loss_history = []
        self.val_loss_history = []
        num_batches = len(train_loader)

        # Training loop
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, batch in tqdm(
                enumerate(train_loader), desc=f"Epoch {epoch}/{self.config.epochs}"
            ):
                # Move batch to device
                batch = {
                    k: (
                        {feat: val.to(device) for feat, val in v.items()}
                        if isinstance(v, dict)
                        else v.to(device)
                    )
                    for k, v in batch.items()
                }

                labels = batch["labels"]

                # Forward pass
                optimizer.zero_grad()
                probs = self.model(batch)
                loss = criterion(probs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Record loss
                epoch_fraction = epoch + (batch_idx + 1) / num_batches
                self.loss_history.append((epoch_fraction, loss.item()))

                # Statistics
                total_loss += loss.item()
                predicted = (probs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)  # type: ignore

                if (batch_idx + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch+1}/{self.config.epochs}], "
                        f"Batch [{batch_idx+1}/{len(train_loader)}], "
                        f"Loss: {loss.item():.4f}, "
                        f"Acc: {100*correct/total:.2f}%"
                    )

            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total

            # Validation if available
            val_loss_str = ""
            val_acc_str = ""
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        batch = {
                            k: (
                                {feat: val.to(device) for feat, val in v.items()}
                                if isinstance(v, dict)
                                else v.to(device)
                            )
                            for k, v in batch.items()
                        }
                        labels = batch["labels"]
                        probs = self.model(batch)
                        loss = criterion(probs, labels)

                        val_loss += loss.item()
                        predicted = (probs > 0.5).float()
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)  # type: ignore

                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * val_correct / val_total
                self.val_loss_history.append((epoch + 1, avg_val_loss))
                val_loss_str = f", Val Loss: {avg_val_loss:.4f}"
                val_acc_str = f", Val Acc: {val_accuracy:.2f}%"

            print(
                f"Epoch [{epoch+1}/{self.config.epochs}] completed: "
                f"Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%"
                f"{val_loss_str}{val_acc_str}"
            )

        print("Training completed!")

        # Save loss plot
        self._save_loss_plot()

        # Save the label_encoders for inference
        encoders_path = os.path.join(self.config.save_path, "label_encoders.pkl")
        PersistenceManager.save_pickle(self.label_encoders, encoders_path)
        print(f"Label encoders saved to: {encoders_path}")

    def _save_loss_plot(self):
        """Save training loss curve plot with optional validation curve."""
        if not hasattr(self, "loss_history") or len(self.loss_history) == 0:
            print("No loss history to plot.")
            return

        # Extract epochs and losses
        epochs = [item[0] for item in self.loss_history]
        losses = [item[1] for item in self.loss_history]

        # Create figure
        plt.figure(figsize=(12, 6))
        plt.plot(
            epochs,
            losses,
            linewidth=1,
            alpha=0.6,
            color="blue",
            label="Train Loss (raw)",
        )

        # Add moving average for smoothing
        window_size = min(50, len(losses) // 10)
        if window_size > 1:
            smoothed_losses = np.convolve(
                losses, np.ones(window_size) / window_size, mode="valid"
            )
            smoothed_epochs = epochs[window_size - 1 :]
            plt.plot(
                smoothed_epochs,
                smoothed_losses,
                linewidth=2,
                color="red",
                label=f"Train Loss (MA, window={window_size})",
            )

        # Add validation loss if available
        if hasattr(self, "val_loss_history") and len(self.val_loss_history) > 0:
            val_epochs = [item[0] for item in self.val_loss_history]
            val_losses = [item[1] for item in self.val_loss_history]
            plt.plot(
                val_epochs,
                val_losses,
                linewidth=2,
                color="green",
                marker="o",
                markersize=4,
                label="Validation Loss",
            )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Clean DIN Model Training Loss Curve", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save plot
        save_path = os.path.join(self.config.save_path, "clean_din_training_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss plot saved to: {save_path}")
        plt.close()

        # Also save loss history as csv
        loss_df = pd.DataFrame(self.loss_history, columns=["epoch", "loss"])
        csv_path = os.path.join(self.config.save_path, "clean_din_training_loss.csv")
        loss_df.to_csv(csv_path, index=False)
        print(f"Loss history saved to: {csv_path}")

    def predict(self):
        """
        Generate predictions on test set.

        Returns:
            probs: np.ndarray of shape (num_samples,) with predicted click probabilities
        """
        if self.model is None:
            raise ValueError(
                "Model is not trained yet. Please train the model before prediction."
            )

        # Load label encoders if not already available
        if not hasattr(self, "label_encoders") or not self.label_encoders:
            encoders_path = os.path.join(self.config.save_path, "label_encoders.pkl")
            if os.path.exists(encoders_path):
                self.label_encoders = PersistenceManager.load_pickle(encoders_path)
                print(f"Loaded label encoders from: {encoders_path}")
            else:
                print(
                    "Warning: No label encoders found. Feature encoding may be inconsistent."
                )

        self.model.eval()

        # Create dataset for prediction
        test_dataset = DINDataset(
            main_df=self.main_df,
            user_profile_dict=self.user_profile_dict,
            item_features_dict=self.item_features_dict,
            user_history_dict=self.user_history_dict,
            user_profile_features=self.user_profile_features,
            item_features=self.item_features,
            context_features=self.context_features,
            label_col="label",
            label_encoders=self.label_encoders,  # Pass encoders to dataset
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, seq_max_len=self.config.din_seq_max_len),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {
                    k: (
                        {feat: val.to(device) for feat, val in v.items()}
                        if isinstance(v, dict)
                        else v.to(device)
                    )
                    for k, v in batch.items()
                }
                probs = self.model(batch)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)
