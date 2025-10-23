"""
Example script for running the ranking pipeline with Clean DIN model.

This demonstrates:
1. Feature extraction from recall results
2. Training Clean DIN ranking model
3. Generating predictions and recommendations

Usage:
    python examples/rank_pipeline_example.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.rank_pipeline import RankPipeline
from src.utils.config import RankConfig, RecallConfig


def main():
    """Run complete ranking pipeline example."""

    print("\n" + "=" * 80)
    print(" " * 25 + "RANKING PIPELINE EXAMPLE")
    print("=" * 80)

    # ============================================================================
    # Configuration
    # ============================================================================

    # Recall config (needed for feature extraction)
    recall_config = RecallConfig(
        debug_mode=True,  # Use debug mode for faster iteration
        debug_sample_size=10000,
        offline=True,  # Training mode: generate labels
        last_N=3,  # Use last 3 items for context features
        embedding_dim=64,
        enable_binning=True,
    )

    # Rank config
    rank_config = RankConfig(
        debug_mode=True,
        offline=True,  # Training mode
        # DIN model hyperparameters
        din_embedding_dim=32,
        din_attention_hidden_units=[36],
        din_mlp_hidden_units=[200, 80],
        din_activation="dice",
        din_seq_max_len=30,
        # Training hyperparameters
        batch_size=256,
        learning_rate=0.001,
        epochs=5,  # Use more epochs (e.g., 10-20) for production
    )

    print("\nConfiguration:")
    print(f"  - Debug mode: {rank_config.debug_mode}")
    print(f"  - Offline mode: {rank_config.offline}")
    print(f"  - Embedding dim: {rank_config.din_embedding_dim}")
    print(f"  - Batch size: {rank_config.batch_size}")
    print(f"  - Epochs: {rank_config.epochs}")
    print(f"  - Learning rate: {rank_config.learning_rate}")

    # ============================================================================
    # Initialize Pipeline
    # ============================================================================

    pipeline = RankPipeline(rank_config)

    # ============================================================================
    # Option 1: Run Full Pipeline (Feature Extraction + Training + Prediction)
    # ============================================================================

    print("\n" + "=" * 80)
    print("OPTION 1: Running Full Pipeline")
    print("=" * 80)

    recommendations = pipeline.run_full_pipeline(
        recall_config=recall_config, top_k=10  # Top 10 recommendations per user
    )

    # ============================================================================
    # Option 2: Run Step-by-Step (More Control)
    # ============================================================================

    # Uncomment below to run step-by-step instead:
    """
    print("\n" + "=" * 80)
    print("OPTION 2: Running Step-by-Step")
    print("=" * 80)
    
    # Step 1: Extract features
    pipeline.extract_features(recall_config)
    
    # Step 2: Train model
    pipeline.train()
    
    # Step 3: Generate predictions
    probs = pipeline.predict()
    print(f"\nPrediction statistics:")
    print(f"  - Number of predictions: {len(probs)}")
    print(f"  - Score range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  - Mean score: {probs.mean():.4f}")
    print(f"  - Std score: {probs.std():.4f}")
    
    # Step 4: Generate recommendations
    recommendations = pipeline.rank_and_recommend(
        top_k=10,
        save_path=rank_config.save_path + "/final_recommendations.pkl"
    )
    """

    # ============================================================================
    # Analyze Results
    # ============================================================================

    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)

    print(f"\nTotal users with recommendations: {len(recommendations)}")

    # Show sample recommendations for first 3 users
    print("\nSample recommendations:")
    for idx, (user_id, items) in enumerate(list(recommendations.items())[:3], 1):
        print(f"\n{idx}. User {user_id}:")
        for rank, (item_id, score) in enumerate(items[:5], 1):
            print(f"   Rank {rank}: Item {item_id} (score: {score:.4f})")

    # Score distribution
    all_scores = [score for items in recommendations.values() for _, score in items]
    print(f"\nScore distribution across all recommendations:")
    print(f"  - Min: {min(all_scores):.4f}")
    print(f"  - Max: {max(all_scores):.4f}")
    print(f"  - Mean: {sum(all_scores)/len(all_scores):.4f}")
    print(f"  - Median: {sorted(all_scores)[len(all_scores)//2]:.4f}")

    print("\n" + "=" * 80)
    print("RANKING PIPELINE EXAMPLE COMPLETED!")
    print("=" * 80)

    print("\nGenerated files:")
    print(f"  - Features: {recall_config.save_path}/main_features.csv")
    print(f"  - Model encoders: {rank_config.save_path}/label_encoders.pkl")
    print(
        f"  - Training loss plot: {rank_config.save_path}/clean_din_training_loss.png"
    )
    print(f"  - Loss history: {rank_config.save_path}/clean_din_training_loss.csv")
    print(f"  - Recommendations: {rank_config.save_path}/final_recommendations.pkl")


def evaluate_model():
    """
    Optional: Evaluate model performance on labeled test set.

    This function demonstrates how to evaluate the ranking model
    when you have ground truth labels available.
    """
    from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
    import pandas as pd
    import numpy as np

    rank_config = RankConfig(
        offline=True,  # Need labels for evaluation
    )

    pipeline = RankPipeline(rank_config)

    # Generate predictions
    probs = pipeline.predict()

    # Load labels
    main_df = pd.read_csv(rank_config.main_features_path)
    labels = main_df["label"].values

    # Filter out samples without labels (label == -1 in online mode)
    valid_mask = labels != -1
    probs_valid = probs[valid_mask]
    labels_valid = labels[valid_mask]

    # Calculate metrics
    auc = roc_auc_score(labels_valid, probs_valid)
    logloss = log_loss(labels_valid, probs_valid)

    # Accuracy at threshold 0.5
    predictions = (probs_valid > 0.5).astype(int)
    accuracy = accuracy_score(labels_valid, predictions)

    print("\n" + "=" * 80)
    print("MODEL EVALUATION METRICS")
    print("=" * 80)
    print(f"\nAUC-ROC: {auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Accuracy (threshold=0.5): {accuracy:.4f}")

    # Positive/negative distribution
    pos_rate = labels_valid.mean()
    print(f"\nLabel distribution:")
    print(f"  - Positive rate: {pos_rate:.2%}")
    print(f"  - Negative rate: {1-pos_rate:.2%}")

    return auc, logloss, accuracy


if __name__ == "__main__":
    # Run main pipeline
    main()

    # Uncomment to run evaluation
    # evaluate_model()
