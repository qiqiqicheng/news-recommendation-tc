import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import faiss
import collections
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from .base import BaseRecaller


class YoutubeDNNDataset(Dataset):
    def __init__(self, data_tuples):
        """
        Args:
            data_tuples: List of (user_id, hist_items, target_item, label, hist_len)
        """
        self.data = data_tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return raw data without padding - padding will be done in collate_fn"""
        user_id, hist_items, target_item, label, hist_len = self.data[idx]

        return {
            "user_id": user_id,
            "hist_items": hist_items,  # Raw list without padding
            "target_item": target_item,
            "hist_len": hist_len,
            "label": label,
        }


def collate_fn(batch, seq_max_len):
    """
    Custom collate function for efficient batch padding

    Args:
        batch: List of samples from __getitem__
        seq_max_len: Maximum sequence length

    Returns:
        Batched and padded tensors
    """
    user_ids = []
    hist_items_list = []
    target_items = []
    hist_lens = []
    labels = []

    for sample in batch:
        user_ids.append(sample["user_id"])

        # Pad or truncate history
        hist = sample["hist_items"]
        if len(hist) > seq_max_len:
            hist = hist[:seq_max_len]
            actual_len = seq_max_len
        else:
            actual_len = len(hist)
            hist = hist + [0] * (seq_max_len - len(hist))

        hist_items_list.append(hist)
        target_items.append(sample["target_item"])
        hist_lens.append(actual_len)
        labels.append(sample["label"])

    return {
        "user_id": torch.tensor(user_ids, dtype=torch.long),
        "hist_items": torch.tensor(hist_items_list, dtype=torch.long),
        "target_item": torch.tensor(target_items, dtype=torch.long),
        "hist_len": torch.tensor(hist_lens, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.float),
    }


class YoutubeDNN(nn.Module):
    def __init__(self, user_vocab_size, item_vocab_size, embedding_dim, hidden_units):
        """
        Args:
            user_vocab_size: Size of user vocabulary
            item_vocab_size: Size of item vocabulary
            embedding_dim: Dimension of embeddings
            hidden_units: List of hidden layer sizes
        """
        super(YoutubeDNN, self).__init__()

        self.embedding_dim = embedding_dim

        # Embeddings
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim)
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim)

        # User tower (sequence encoder + MLP)
        user_input_dim = embedding_dim * 2  # user_emb + avg_hist_emb
        layers = []
        prev_dim = user_input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        self.user_tower = nn.Sequential(*layers)

        # Item tower
        self.item_tower = nn.Identity()  # Simple pass-through

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, user_id, hist_items, hist_len, target_item=None):
        """
        Forward pass

        Args:
            user_id: User IDs [batch_size]
            hist_items: Historical items [batch_size, seq_len], padded with zeros
            hist_len: Actual history lengths [batch_size]
            target_item: Target items [batch_size] (optional)

        Returns:
            user_emb, item_emb (if target_item provided), else user_emb only
        """
        batch_size = user_id.size(0)

        # User embedding
        user_emb = self.user_embedding(user_id)  # [batch_size, emb_dim]

        # History embedding (average pooling with mask)
        hist_emb = self.item_embedding(hist_items)  # [batch_size, seq_len, emb_dim]

        # Create mask for padding
        mask = torch.arange(hist_items.size(1), device=hist_items.device).unsqueeze(
            0
        ) < hist_len.unsqueeze(1)
        mask = mask.unsqueeze(2).float()  # [batch_size, seq_len, 1]

        # Masked average
        hist_emb_masked = hist_emb * mask
        hist_emb_avg = hist_emb_masked.sum(dim=1) / (
            hist_len.unsqueeze(1).float() + 1e-8
        )

        # Concatenate user and history embeddings
        user_input = torch.cat([user_emb, hist_emb_avg], dim=1)

        # User tower
        user_repr = self.user_tower(user_input)  # [batch_size, emb_dim]

        # Normalize
        user_repr = nn.functional.normalize(user_repr, p=2, dim=1)

        if target_item is not None:
            # Item tower
            item_emb = self.item_embedding(target_item)  # [batch_size, emb_dim]
            item_repr = self.item_tower(item_emb)
            item_repr = nn.functional.normalize(item_repr, p=2, dim=1)
            return user_repr, item_repr

        return user_repr

    def get_user_embedding(self, user_id, hist_items, hist_len):
        """Get user embedding"""
        return self.forward(user_id, hist_items, hist_len)

    def get_item_embedding(self, item_ids):
        """Get item embeddings"""
        item_emb = self.item_embedding(item_ids)
        item_repr = self.item_tower(item_emb)
        return nn.functional.normalize(item_repr, p=2, dim=1)


class YoutubeDNNRecaller(BaseRecaller):
    def __init__(self, config):
        super().__init__(config)

        # Get parameters from config
        self.seq_max_len = getattr(config, "youtubednn_seq_max_len", 30)
        self.embedding_dim = getattr(config, "youtubednn_embedding_dim", 16)
        self.hidden_units = getattr(config, "youtubednn_hidden_units", [64, 16])
        self.negsample = getattr(config, "youtubednn_negsample", 4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.user_index_2_rawid = {}
        self.user_rawid_2_index = {}  # Reverse mapping for faster lookup
        self.item_index_2_rawid = {}
        self.item_rawid_2_index = {}  # Reverse mapping for faster lookup
        self.user_embeddings = None
        self.item_embeddings = None
        self.faiss_index = None

    def _prepare_data(self, click_df, negsample=0):
        """
        Prepare training and test data

        Args:
            click_df: Click log dataframe
            negsample: Number of negative samples per positive sample

        Returns:
            train_set, test_set
            train_set: [(user_id, hist_items, target_item, label, hist_len), ...]
            test_set: [(user_id, hist_items, target_item, label, hist_len), ...]
        """
        import random

        click_df = click_df.sort_values("click_timestamp")
        item_ids = click_df["click_article_id"].unique()

        train_set = []
        test_set = []

        for user_id, hist in tqdm(click_df.groupby("user_id"), desc="Preparing data"):
            pos_list = hist["click_article_id"].tolist()

            if len(pos_list) < 2:
                continue

            # Generate negative samples
            if negsample > 0:
                candidate_set = list(set(item_ids) - set(pos_list))
                if len(candidate_set) == 0:
                    continue
                neg_list = np.random.choice(
                    candidate_set, size=len(pos_list) * negsample, replace=True
                )

            # Sliding window
            test_size = max(1, int(len(pos_list) * 0.2))

            for i in range(1, len(pos_list)):
                hist_items = pos_list[:i]
                target = pos_list[i]

                is_test = i >= len(pos_list) - test_size

                if is_test:
                    test_set.append((user_id, hist_items, target, 1, len(hist_items)))
                else:
                    train_set.append((user_id, hist_items, target, 1, len(hist_items)))

                    if negsample > 0:
                        for negi in range(negsample):
                            neg_idx = (i - 1) * negsample + negi
                            train_set.append(
                                (
                                    user_id,
                                    hist_items,
                                    neg_list[neg_idx],
                                    0,
                                    len(hist_items),
                                )
                            )

        random.shuffle(train_set)

        return train_set, test_set

    def train(self, click_df, epochs=1, batch_size=256, learning_rate=0.001):
        """
        Train YoutubeDNN model

        Args:
            click_df: Click log dataframe
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        print("Training YoutubeDNN model...")

        # Label encoding
        click_df = click_df.copy()
        user_profile_raw = click_df[["user_id"]].drop_duplicates("user_id")
        item_profile_raw = click_df[["click_article_id"]].drop_duplicates(
            "click_article_id"
        )

        user_le = LabelEncoder()
        item_le = LabelEncoder()

        click_df["user_id"] = user_le.fit_transform(click_df["user_id"])
        click_df["click_article_id"] = item_le.fit_transform(
            click_df["click_article_id"]
        )

        user_profile = click_df[["user_id"]].drop_duplicates("user_id")
        item_profile = click_df[["click_article_id"]].drop_duplicates(
            "click_article_id"
        )

        self.user_index_2_rawid = dict(
            zip(user_profile["user_id"], user_profile_raw["user_id"])
        )
        self.item_index_2_rawid = dict(
            zip(item_profile["click_article_id"], item_profile_raw["click_article_id"])
        )

        # Create reverse mappings for faster lookup
        self.user_rawid_2_index = {v: k for k, v in self.user_index_2_rawid.items()}
        self.item_rawid_2_index = {v: k for k, v in self.item_index_2_rawid.items()}

        # Prepare data with configurable negative sampling
        train_set, test_set = self._prepare_data(click_df, negsample=self.negsample)

        # Create datasets
        train_dataset = YoutubeDNNDataset(train_set)

        # Create custom collate function with seq_max_len bound
        from functools import partial

        train_collate_fn = partial(collate_fn, seq_max_len=self.seq_max_len)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_collate_fn,
        )

        # Initialize model
        user_vocab_size = click_df["user_id"].max() + 1
        item_vocab_size = click_df["click_article_id"].max() + 1

        self.model = YoutubeDNN(
            user_vocab_size, item_vocab_size, self.embedding_dim, self.hidden_units
        ).to(self.device)

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                user_id = batch["user_id"].to(self.device)
                hist_items = batch["hist_items"].to(self.device)
                target_item = batch["target_item"].to(self.device)
                hist_len = batch["hist_len"].to(self.device)
                label = batch["label"].to(self.device)

                optimizer.zero_grad()

                # Forward
                user_emb, item_emb = self.model(
                    user_id, hist_items, hist_len, target_item
                )

                # Compute similarity score
                logits = (user_emb * item_emb).sum(dim=1)

                # Loss
                loss = criterion(logits, label)

                # Backward
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Extract embeddings
        self._extract_embeddings(test_set, item_profile)

        print("Training completed!")

    def _extract_embeddings(self, test_set, item_profile):
        """Extract user and item embeddings after training"""
        self.model.eval()

        # Extract user embeddings
        test_dataset = YoutubeDNNDataset(test_set)
        from functools import partial

        test_collate_fn = partial(collate_fn, seq_max_len=self.seq_max_len)
        test_loader = DataLoader(
            test_dataset, batch_size=1024, shuffle=False, collate_fn=test_collate_fn
        )

        user_embs_list = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Extracting user embeddings"):
                user_id = batch["user_id"].to(self.device)
                hist_items = batch["hist_items"].to(self.device)
                hist_len = batch["hist_len"].to(self.device)

                user_emb = self.model.get_user_embedding(user_id, hist_items, hist_len)
                user_embs_list.append(user_emb.cpu().numpy())

        self.user_embeddings = np.vstack(user_embs_list)

        # Extract item embeddings
        item_ids = torch.tensor(
            item_profile["click_article_id"].values, dtype=torch.long
        ).to(self.device)
        with torch.no_grad():
            item_embs_list = []
            for i in tqdm(
                range(0, len(item_ids), 1024), desc="Extracting item embeddings"
            ):
                batch_ids = item_ids[i : i + 1024]
                item_emb = self.model.get_item_embedding(batch_ids)
                item_embs_list.append(item_emb.cpu().numpy())

        self.item_embeddings = np.vstack(item_embs_list)

        # Build Faiss index
        print("Building Faiss index...")
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(self.item_embeddings.astype(np.float32))
        print("Faiss index built!")

    def recall(self, user_id: int, topk: int = 20) -> List[Tuple[int, float]]:
        """
        Recall top-k items for a user using YoutubeDNN

        Args:
            user_id: User ID (raw ID)
            topk: Number of items to recall

        Returns:
            List of (item_id, score) tuples
        """
        if self.model is None or self.user_embeddings is None:
            raise ValueError("Model not trained. Call train() first.")

        # Use reverse mapping for faster lookup
        user_idx = self.user_rawid_2_index.get(user_id, None)

        if user_idx is None or user_idx >= len(self.user_embeddings):
            return []

        # Get user embedding
        user_emb = self.user_embeddings[user_idx : user_idx + 1].astype(np.float32)

        # Search with Faiss
        sim, idx = self.faiss_index.search(user_emb, topk + 1)

        # Convert to raw IDs
        results = []
        for i in range(1, len(idx[0])):  # Skip first (might be duplicates)
            item_idx = idx[0][i]
            similarity = float(sim[0][i])

            if item_idx in self.item_index_2_rawid:
                raw_item_id = self.item_index_2_rawid[item_idx]
                results.append((raw_item_id, similarity))

            if len(results) >= topk:
                break

        return results

    def train_with_cv(
        self,
        click_df,
        param_grid: Optional[Dict] = None,
        n_splits: int = 5,
        epochs: int = 1,
        batch_size: int = 256,
        learning_rate: float = 0.001,
    ) -> Dict:
        """
        Train with cross-validation for hyperparameter tuning

        Args:
            click_df: Click log dataframe
            param_grid: Dictionary of parameters to search
                Example: {
                    'embedding_dim': [8, 16, 32],
                    'hidden_units': [[64, 16], [128, 32], [256, 64]],
                    'negsample': [2, 4, 8]
                }
            n_splits: Number of CV folds
            epochs: Training epochs per fold
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Dictionary with best parameters and scores
        """
        if param_grid is None:
            # Default parameter grid
            param_grid = {
                "embedding_dim": [16],
                "hidden_units": [[64, 16]],
                "negsample": [4],
            }

        print(f"Cross-validation with {n_splits} folds...")

        # Generate parameter combinations
        import itertools

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        best_score = -np.inf
        best_params = None
        cv_results = []

        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))
            print(f"\nTesting parameters: {params}")

            # Update model parameters
            if "embedding_dim" in params:
                self.embedding_dim = params["embedding_dim"]
            if "hidden_units" in params:
                self.hidden_units = params["hidden_units"]
            if "negsample" in params:
                self.negsample = params["negsample"]

            # K-Fold cross-validation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_scores = []

            users = click_df["user_id"].unique()

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(users)):
                print(f"  Fold {fold_idx + 1}/{n_splits}")

                train_users = users[train_idx]
                val_users = users[val_idx]

                train_fold = click_df[click_df["user_id"].isin(train_users)]
                val_fold = click_df[click_df["user_id"].isin(val_users)]

                # Train on this fold
                self.train(
                    train_fold,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )

                # Evaluate on validation fold
                score = self._evaluate_fold(val_fold, topk=20)
                fold_scores.append(score)
                print(f"    Recall@20: {score:.4f}")

            # Average score across folds
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            cv_results.append(
                {"params": params, "mean_score": avg_score, "std_score": std_score}
            )

            print(f"  Average Recall@20: {avg_score:.4f} (+/- {std_score:.4f})")

            # Update best parameters
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        print(f"\nBest parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")

        # Retrain with best parameters on full dataset
        if best_params:
            if "embedding_dim" in best_params:
                self.embedding_dim = best_params["embedding_dim"]
            if "hidden_units" in best_params:
                self.hidden_units = best_params["hidden_units"]
            if "negsample" in best_params:
                self.negsample = best_params["negsample"]

            print("\nRetraining with best parameters on full dataset...")
            self.train(
                click_df,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

        return {
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": cv_results,
        }

    def _evaluate_fold(self, val_df, topk: int = 20) -> float:
        """
        Evaluate model on validation fold

        Args:
            val_df: Validation dataframe
            topk: Top-K for recall metric

        Returns:
            Recall@K score
        """
        if self.model is None:
            return 0.0

        hits = 0
        total = 0

        # Sample a subset of users for faster evaluation
        sample_size = min(100, len(val_df["user_id"].unique()))
        sample_users = np.random.choice(
            val_df["user_id"].unique(), size=sample_size, replace=False
        )

        for user_id in sample_users:
            user_data = val_df[val_df["user_id"] == user_id]

            if len(user_data) == 0:
                continue

            # Get ground truth (last clicked item)
            ground_truth = user_data["click_article_id"].iloc[-1]

            # Get recall results
            recall_results = self.recall(user_id, topk=topk)
            recall_items = [item_id for item_id, _ in recall_results]

            # Check if ground truth in recall results
            if ground_truth in recall_items:
                hits += 1
            total += 1

        return hits / total if total > 0 else 0.0
