import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from typing import Optional, List, Literal
from sklearn.preprocessing import LabelEncoder
from functools import partial
import matplotlib.pyplot as plt

from src.rank.base import BaseRanker
from src.utils.config import RecallConfig, RankConfig


class Dice(nn.Module):
    def __init__(self, alpha: float = 0.1, eps: float = 1e-8) -> None:
        """
        Dice activation function from the original paper of DIN
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
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
        hidden_units: list = [
            36,
        ],
        activation: Literal["dice", "prelu"] = "dice",
    ) -> None:
        """
        acitvation unit based on attention machanism

        Args:
            input_dim (int): should be 4 times original embedding dim
            hidden_units (list, optional): Defaults to [80, 40].
            activation (Literal["dice";, "prelu"], optional): Activation method. Defaults to 'dice'.
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
    ):
        """
        1. concat query, keys, query-keys, query*keys
        2. MLP to get attention scores
        3. weighted sum of keys with attention scores

        Args:
            queries: batch recall items (B, d)
            keys: history items (B, T, d)

        Returns:
            weight_scores: (B, T, 1)
        """
        B, T, d = keys.shape
        query_expanded = queries.unsqueeze(1).expand(B, T, d)

        minus_emb = query_expanded - keys
        hadamard_emb = query_expanded * keys

        concatenated_embs = torch.cat(
            [keys, query_expanded, minus_emb, hadamard_emb], dim=-1
        )  # (B, T, 4d)

        weight_scores = self.mlp(concatenated_embs)

        return weight_scores


class DINModel(nn.Module):
    def __init__(
        self,
        user_profile_vocab_dict: dict,
        item_vocab_dict: dict,
        context_vocab_dict: dict,
        embedding_dim: int = 64,
        attention_hidden_units: list = [
            36,
        ],
        mlp_hidden_units: list = [200, 80, 2],
        activation: Literal["dice", "prelu"] = "dice",
    ) -> None:
        """
        DIN model

        Args:
            vocab_dict (dict): {feature_name: vocab_size, ...}
            embedding_dim (int, optional): Defaults to 64.
            activation: acivation function

        Note:
            user_id and recall_id (item_id) are excluded from embeddings.
            They are only used for indexing and will not participate in training.
        """
        super().__init__()

        # Define ID features to exclude from embeddings
        self.excluded_id_features = {"user_id", "recall_id"}

        # DIN architecture
        # 1. embedding layer - exclude user_id and recall_id
        self.user_profile_embedding_dict = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_size, embedding_dim)
                for feature, vocab_size in user_profile_vocab_dict.items()
                if feature not in self.excluded_id_features
            }
        )
        self.item_embedding_dict = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_size, embedding_dim)
                for feature, vocab_size in item_vocab_dict.items()
                if feature not in self.excluded_id_features
            }
        )
        self.context_embedding_dict = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_size, embedding_dim)
                for feature, vocab_size in context_vocab_dict.items()
                if feature not in self.excluded_id_features
            }
        )

        # 2. activation unit(attention)
        item_feature_len = len(item_vocab_dict)
        self.item_dim = item_feature_len * embedding_dim
        self.activation_unit = ActivationUnit(4 * self.item_dim)  # use 4 * input_dim

        # 3. MLP
        self.user_profile_dim = len(user_profile_vocab_dict) * embedding_dim
        self.context_dim = len(context_vocab_dict) * embedding_dim
        # MLP input dim = user_profile_dim + context_dim + recall_item_dim + weighted_history_item_dim
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
            batch: dictionary from collate_fn with keys:
                - 'user_profile': {feature_name: tensor[B]}
                - 'recall_item': {feature_name: tensor[B]}
                - 'history_items': {feature_name: tensor[B, T]}
                - 'context': {feature_name: tensor[B]}

        Returns:
            probs: (B, ) - predicted click probabilities

        Note:
            user_id and recall_id are automatically filtered out and will not
            participate in embedding computation.
        """
        user_profile_dict = batch["user_profile"]
        history_item_dict = batch["history_items"]
        recall_item_dict = batch["recall_item"]
        context_dict = batch["context"]

        # 1. get all embeddings (excluding user_id and recall_id)
        user_profile_embs = torch.cat(
            [
                self.user_profile_embedding_dict[feature](indices)
                for feature, indices in user_profile_dict.items()
                if feature not in self.excluded_id_features
            ],
            dim=1,
        )  # (B, user_profile_dim)

        history_embs = torch.cat(
            [
                self.item_embedding_dict[feature](indices)  # (B, T, single_dim)
                for feature, indices in history_item_dict.items()  # indices: (B, T)
                if feature not in self.excluded_id_features
            ],
            dim=2,
        )  # (B, T, item_dim)

        recall_embs = torch.cat(
            [
                self.item_embedding_dict[feature](indices)
                for feature, indices in recall_item_dict.items()
                if feature not in self.excluded_id_features
            ],
            dim=1,
        )  # (B, item_dim)

        context_embs = torch.cat(
            [
                self.context_embedding_dict[feature](indices)
                for feature, indices in context_dict.items()
                if feature not in self.excluded_id_features
            ],
            dim=1,
        )  # (B, context_dim)

        # 2. get weight scores from activation unit
        weight_scores = self.activation_unit(
            queries=recall_embs, keys=history_embs  # (B, item_dim)  # (B, T, item_dim)
        )  # (B, T, 1)

        # 3. construct mlp input
        weighted_history = (weight_scores * history_embs).sum(dim=1)  # (B, item_dim)
        mlp_input = torch.cat(
            [user_profile_embs, context_embs, recall_embs, weighted_history], dim=1
        )  # (B, mlp_input_dim)

        # 4. MLP
        logits = self.mlp(mlp_input)  # (B, 1)
        probs = torch.sigmoid(logits).squeeze(-1)  # (B, )

        return probs


class DINDataset(Dataset):
    def __init__(
        self,
        data_df: pd.DataFrame,
        history_dict: dict,
        article_info_dict: dict,
        user_profile_features: List[str],
        item_features: List[str],
        context_features: List[str],
        label_col: str = "label",
    ) -> None:
        """
        Dataset for DIN model.

        Args:
            data_df: DataFrame with user_id, recall_id, context features, and label
            history_dict: {user_id: [(item_id, timestamp), ...]}
            article_info_dict: {article_id: {feature_name: value, ...}}
            user_profile_features: list of user profile feature names
            item_features: list of item feature names
            context_features: list of context feature names
            label_col: name of the label column
        """
        self.data_df = data_df.reset_index(drop=True)
        self.history_dict = history_dict
        self.article_info_dict = article_info_dict
        self.user_profile_features = user_profile_features
        self.item_features = item_features
        self.context_features = context_features
        self.label_col = label_col

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Returns a dictionary with:
        - user_profile: {feature_name: value}
        - recall_item: {feature_name: value}
        - history_items: list of {feature_name: value}
        - context: {feature_name: value}
        - label: int
        """
        row = self.data_df.iloc[idx]
        user_id = row["user_id"]
        recall_id = row["recall_id"]

        # extract user profile features
        user_profile = {}
        for feat in self.user_profile_features:
            if feat in row:
                user_profile[feat] = row[feat]

        # extract recall item features
        recall_item = {"recall_id": recall_id}  # always include recall_id
        # add features from data_df (article_popularity is in the dataframe)
        for feat in self.item_features:
            if feat == "recall_id":
                continue
            if feat in row:
                recall_item[feat] = row[feat]
            elif (
                recall_id in self.article_info_dict
                and feat in self.article_info_dict[recall_id]
            ):
                recall_item[feat] = self.article_info_dict[recall_id][feat]
            else:
                recall_item[feat] = 0  # default value

        # extract history items
        history_items = []
        if user_id in self.history_dict:
            for item_id, timestamp in self.history_dict[user_id]:
                hist_item = {
                    "recall_id": item_id
                }  # use recall_id as the key for consistency
                if item_id in self.article_info_dict:
                    for feat in self.item_features:
                        if feat == "recall_id":
                            continue
                        hist_item[feat] = self.article_info_dict[item_id].get(feat, 0)
                else:
                    for feat in self.item_features:
                        if feat == "recall_id":
                            continue
                        hist_item[feat] = 0
                history_items.append(hist_item)

        # extract context features
        context = {}
        for feat in self.context_features:
            if feat in row:
                context[feat] = row[feat]
            else:
                context[feat] = 0

        # extract label
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
    Custom collate function to handle variable-length sequences.
    Pads/truncates history sequences to seq_max_len.

    Args:
        batch: List of dictionaries from DINDataset
        seq_max_len: Maximum sequence length to pad/truncate to

    Returns:
        A dictionary of batched tensors ready for DINModel forward:
        {
            'user_profile': {feature_name: tensor[B]},
            'recall_item': {feature_name: tensor[B]},
            'history_items': {feature_name: tensor[B, T]},
            'context': {feature_name: tensor[B]},
            'labels': tensor[B]
        }
    """
    batch_size = len(batch)

    # collect all feature names
    user_profile_features = list(batch[0]["user_profile"].keys())
    item_features = list(batch[0]["recall_item"].keys())
    context_features = list(batch[0]["context"].keys())

    # initialize batch dictionaries
    user_profile_batch = {feat: [] for feat in user_profile_features}
    recall_item_batch = {feat: [] for feat in item_features}
    history_items_batch = {
        feat: [] for feat in item_features
    }  # same features as recall_item
    context_batch = {feat: [] for feat in context_features}
    labels_batch = []

    for sample in batch:
        # user profile features
        for feat in user_profile_features:
            user_profile_batch[feat].append(sample["user_profile"][feat])

        # recall item features
        for feat in item_features:
            recall_item_batch[feat].append(sample["recall_item"][feat])

        # history items features - pad or truncate
        history_items = sample["history_items"]
        history_len = len(history_items)

        # truncate if too long
        if history_len > seq_max_len:
            history_items = history_items[-seq_max_len:]  # keep the most recent items
            history_len = seq_max_len

        # pad if too short
        for feat in item_features:
            feat_values = [item[feat] for item in history_items]
            # pad with zeros
            if history_len < seq_max_len:
                feat_values = feat_values + [0] * (seq_max_len - history_len)
            history_items_batch[feat].append(feat_values)

        # context features
        for feat in context_features:
            context_batch[feat].append(sample["context"][feat])

        # label
        labels_batch.append(sample["label"])

    # convert to tensors
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

    return {
        "user_profile": user_profile_batch,
        "recall_item": recall_item_batch,
        "history_items": history_items_batch,
        "context": context_batch,
        "labels": labels_batch,
    }


class DINRanker(BaseRanker):
    def __init__(self, config: RankConfig):
        super().__init__(config)
        self.model = None
        self.label_encoders = {}  # store label encoders for each feature

    def _prepare_vocab_dicts(self):
        """
        Prepare vocabulary dictionaries for each feature group by analyzing the training data.
        This creates mappings from feature values to indices for embedding layers.
        """
        # combine train and test for consistent encoding
        all_data = pd.concat([self.train_set, self.test_set], ignore_index=True)

        # prepare vocab dicts for each feature group
        user_profile_vocab_dict = {}
        item_vocab_dict = {}
        context_vocab_dict = {}

        # encode user profile features
        for feat in self.user_profile_features:
            if feat in all_data.columns:
                le = LabelEncoder()
                all_data[feat] = all_data[feat].fillna(0)
                le.fit(all_data[feat].astype(str))
                self.label_encoders[feat] = le
                user_profile_vocab_dict[feat] = len(le.classes_) + 1  # +1 for padding

        # encode item features (need to include history items from article_info_dict)
        for feat in self.item_features:
            if feat == "recall_id":
                # special handling for article IDs
                all_article_ids = set(all_data["recall_id"].unique())
                # add article IDs from article_info_dict
                all_article_ids.update(self.article_info_dict.keys())
                le = LabelEncoder()
                le.fit(list(all_article_ids))
                self.label_encoders[feat] = le
                item_vocab_dict[feat] = len(le.classes_) + 1
            elif feat in all_data.columns:
                le = LabelEncoder()
                all_data[feat] = all_data[feat].fillna(0)
                le.fit(all_data[feat].astype(str))
                self.label_encoders[feat] = le
                item_vocab_dict[feat] = len(le.classes_) + 1
            else:
                # feature not in dataframe, get from article_info_dict
                all_values = set()
                for article_info in self.article_info_dict.values():
                    if feat in article_info:
                        all_values.add(article_info[feat])
                if all_values:
                    le = LabelEncoder()
                    le.fit(list(all_values))
                    self.label_encoders[feat] = le
                    item_vocab_dict[feat] = len(le.classes_) + 1

        # encode context features
        for feat in self.context_features:
            if feat in all_data.columns:
                le = LabelEncoder()
                all_data[feat] = all_data[feat].fillna(0)
                le.fit(all_data[feat].astype(str))
                self.label_encoders[feat] = le
                context_vocab_dict[feat] = len(le.classes_) + 1

        print(f"User profile vocab: {user_profile_vocab_dict}")
        print(f"Item vocab: {item_vocab_dict}")
        print(f"Context vocab: {context_vocab_dict}")

        return user_profile_vocab_dict, item_vocab_dict, context_vocab_dict

    def train(self):
        """
        Train the DIN model and record loss history.
        """
        # load data
        self.load(load_din_specific=True)

        # prepare vocabulary dictionaries
        user_profile_vocab, item_vocab, context_vocab = self._prepare_vocab_dicts()

        # initialize model
        self.model = DINModel(
            user_profile_vocab_dict=user_profile_vocab,
            item_vocab_dict=item_vocab,
            context_vocab_dict=context_vocab,
            embedding_dim=self.config.din_embedding_dim,
            attention_hidden_units=self.config.din_attention_hidden_units,
            mlp_hidden_units=self.config.din_mlp_hidden_units,
            activation=self.config.din_activation,  # type: ignore
        )

        # create dataset
        train_dataset = DINDataset(
            data_df=self.train_set,
            history_dict=self.train_history_dict,
            article_info_dict=self.article_info_dict,
            user_profile_features=self.user_profile_features,
            item_features=self.item_features,
            context_features=self.context_features,
            label_col="label",
        )

        # create dataloader with custom collate_fn

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, seq_max_len=self.config.din_seq_max_len),
        )

        # setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()

        print(f"Training on device: {device}")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of batches: {len(train_loader)}")

        # initialize loss tracking
        self.loss_history = []  # list of (epoch_fraction, loss) tuples
        num_batches = len(train_loader)

        # training loop
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(train_loader):
                # move batch to device
                batch = {
                    k: (
                        {feat: val.to(device) for feat, val in v.items()}
                        if isinstance(v, dict)
                        else v.to(device)
                    )
                    for k, v in batch.items()
                }

                labels = batch["labels"]

                # forward pass
                optimizer.zero_grad()
                probs = self.model(batch)
                loss = criterion(probs, labels)

                # backward pass
                loss.backward()
                optimizer.step()

                # record loss with fractional epoch
                epoch_fraction = epoch + (batch_idx + 1) / num_batches
                self.loss_history.append((epoch_fraction, loss.item()))

                # statistics
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
            print(
                f"Epoch [{epoch+1}/{self.config.epochs}] completed: "
                f"Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )

        print("Training completed!")

        # save loss plot
        self._save_loss_plot()

    def _save_loss_plot(self):
        """
        Save training loss curve plot.
        """
        if not hasattr(self, "loss_history") or len(self.loss_history) == 0:
            print("No loss history to plot.")
            return

        # extract epochs and losses
        epochs = [item[0] for item in self.loss_history]
        losses = [item[1] for item in self.loss_history]

        # create figure
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, linewidth=1, alpha=0.6, color="blue")

        # add moving average for smoothing
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
                label=f"Moving Average (window={window_size})",
            )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("DIN Model Training Loss Curve", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # save plot
        save_path = os.path.join(self.config.save_path, "din_training_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss plot saved to: {save_path}")
        plt.close()

        # also save loss history as csv
        loss_df = pd.DataFrame(self.loss_history, columns=["epoch", "loss"])
        csv_path = os.path.join(self.config.save_path, "din_training_loss.csv")
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

        self.model.eval()
        test_dataset = DINDataset(
            data_df=self.test_set,
            history_dict=self.test_history_dict,
            article_info_dict=self.article_info_dict,
            user_profile_features=self.user_profile_features,
            item_features=self.item_features,
            context_features=self.context_features,
            label_col="label",
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
