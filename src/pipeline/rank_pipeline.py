import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os

from ..rank.DIN import DINRanker
from ..features.feature_extractor import FeatureExtractor
from ..utils.config import RankConfig, RecallConfig
from ..utils.persistence import PersistenceManager


class RankPipeline:
    """
    Pipeline for ranking stage in news recommendation system.

    Workflow:
    1. Feature extraction from recall results (using CleanFeatureExtractor)
    2. Model training (using CleanDINRanker)
    3. Prediction and ranking

    Offline mode (training):
    - Extract features with ground truth labels
    - Train ranking model on historical data
    - Evaluate model performance

    Online mode (serving):
    - Extract features from recall results (no labels)
    - Load pre-trained model
    - Generate ranking scores for recommendation
    """

    def __init__(self, config: RankConfig):
        self.config = config
        self.ranker = None

    def extract_features(self, recall_config: RecallConfig):
        """
        Extract features from recall results for ranking.

        Args:
            recall_config: RecallConfig object with recall pipeline settings

        This step:
        - Loads recall results from recall pipeline
        - Extracts user profile, item, and context features
        - Generates training labels in offline mode
        - Saves features for ranking model
        """
        print("=" * 60)
        print("STEP 1: Feature Extraction")
        print("=" * 60)

        # Initialize feature extractor with recall config
        feature_extractor = FeatureExtractor(recall_config)

        # Extract and save features
        feature_extractor.extract_features(save=True)

        print("\n✓ Feature extraction completed!")
        print(f"Features saved to: {recall_config.save_path}")

    def train(self):
        """
        Train the ranking model using extracted features.

        This step:
        - Loads features from feature extraction step
        - Initializes CleanDIN model
        - Trains model with BCE loss
        - Saves trained model and encoders
        """
        # Initialize ranker
        self.ranker = DINRanker(self.config)

        # Train model
        self.ranker.train()

        print("\nModel training completed")

    def load_model(self, model_dir: Optional[str] = None):
        """
        Load a pre-trained ranking model.

        Args:
            model_dir: Directory containing the saved model. If None, uses config.save_path

        This step:
        - Initializes ranker
        - Loads feature data
        - Loads pre-trained model weights and metadata
        """
        print("=" * 60)
        print("STEP: Load Pre-trained Model")
        print("=" * 60)

        # Initialize ranker
        self.ranker = DINRanker(self.config)

        # Load feature data
        self.ranker.load()

        # Load pre-trained model
        self.ranker.load_model(load_dir=model_dir)

        print("\n✓ Model loaded successfully!")

    def predict(
        self, use_pretrained: bool = False, model_dir: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate ranking scores for all user-item pairs.

        Args:
            use_pretrained: If True, load pre-trained model before prediction
            model_dir: Directory containing saved model (only used if use_pretrained=True)

        Returns:
            probs: Array of predicted click probabilities

        This step:
        - Loads pre-trained model (if use_pretrained=True)
        - Generates predictions for all recall results
        - Returns ranking scores
        """
        # Load pre-trained model if requested
        if use_pretrained:
            self.load_model(model_dir=model_dir)

        if self.ranker is None:
            raise ValueError(
                "Model is not trained/loaded. Train or load the model before prediction."
            )

        # Generate predictions
        probs = self.ranker.predict()

        print(f"Generated {len(probs)} predictions")
        print(f"Score range: [{probs.min():.4f}, {probs.max():.4f}]")
        print(f"Mean score: {probs.mean():.4f}")

        return probs

    def rank_and_recommend(
        self, top_k: int = 10, save_path: Optional[str] = None
    ) -> Dict[str, List[tuple]]:
        """
        Generate final recommendations by ranking predicted scores.

        Args:
            top_k: Number of top items to recommend per user
            save_path: Path to save recommendations (optional)

        Returns:
            recommendations: {user_id: [(item_id, score), ...]}
        """
        # Get predictions
        if self.ranker is None:
            probs = self.predict()
        else:
            probs = self.ranker.predict()

        # Load main dataframe to get user-item pairs
        main_df = pd.read_csv(self.config.main_features_path)
        main_df["rank_score"] = probs

        # Group by user and rank items
        recommendations = {}
        for user_id, group in main_df.groupby("user_id"):
            # Sort by score descending and take top_k
            top_items = group.nlargest(top_k, "rank_score")[["item_id", "rank_score"]]
            recommendations[str(user_id)] = [
                (str(row["item_id"]), float(row["rank_score"]))
                for _, row in top_items.iterrows()
            ]

        print(f"\nRanking completed!")
        print(f"Generated recommendations for {len(recommendations)} users")
        print(f"Top-{top_k} items per user")

        # Show sample recommendations
        sample_user = list(recommendations.keys())[0]
        print(f"\nSample recommendations for user {sample_user}:")
        for idx, (item_id, score) in enumerate(recommendations[sample_user][:5], 1):
            print(f"  {idx}. Item {item_id}: {score:.4f}")

        # Save recommendations if path provided
        if save_path:
            PersistenceManager.save_pickle(recommendations, save_path)
            print(f"\nRecommendations saved to: {save_path}")

        return recommendations

    def run_full_pipeline(self, recall_config: RecallConfig, top_k: int = 10):
        """
        Run complete ranking pipeline: feature extraction → training → prediction → ranking.

        Args:
            recall_config: RecallConfig for feature extraction
            top_k: Number of recommendations per user

        Returns:
            recommendations: {user_id: [(item_id, score), ...]}
        """

        # Step 1: Extract features
        self.extract_features(recall_config)

        # Step 2: Train model (only in offline mode)
        if self.config.offline:
            self.train()
        else:
            print("\nOnline mode: Skipping training, will use pre-trained model")

        # Step 3 & 4: Predict and rank
        recommendations = self.rank_and_recommend(
            top_k=top_k, save_path=self.config.save_path + "/final_recommendations.pkl"
        )

        return recommendations


if __name__ == "__main__":
    rank_config = RankConfig(offline=True)
    recall_config = RecallConfig()

    pipeline = RankPipeline(rank_config)
    pipeline.train()
    pipeline.predict()
    pipeline.rank_and_recommend(
        top_k=10,
        save_path=os.path.join(rank_config.save_path, "final_recommendations.pkl"),
    )

    # python -m src.pipeline.rank_pipeline
