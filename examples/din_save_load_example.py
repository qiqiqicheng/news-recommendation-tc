"""
Example: DIN Model Save and Load

This example demonstrates how to:
1. Train a DIN model and save it
2. Load a pre-trained DIN model for inference
3. Use the loaded model for prediction
"""

from src.rank.DIN import DINRanker
from src.utils.config import RankConfig
import os


def example_train_and_save():
    """Example: Train model and save it"""
    print("=" * 80)
    print("EXAMPLE 1: Train and Save Model")
    print("=" * 80)

    # Initialize config
    config = RankConfig(
        offline=True,
        epochs=2,
        batch_size=256,
        din_embedding_dim=32,
    )

    # Initialize ranker
    ranker = DINRanker(config)

    # Train model (will automatically save after training)
    ranker.train()

    print("\n✓ Model trained and saved!")
    print(f"Model files saved to: {config.save_path}")
    print("  - din_model.pth (model weights)")
    print("  - din_model_metadata.pkl (architecture info)")
    print("  - label_encoders.pkl (feature encoders)")


def example_load_and_predict():
    """Example: Load pre-trained model and make predictions"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Load Pre-trained Model and Predict")
    print("=" * 80)

    # Initialize config (same as training)
    config = RankConfig(
        offline=False,  # Set to False for inference mode
        batch_size=256,
    )

    # Initialize ranker
    ranker = DINRanker(config)

    # Load feature data (needed for prediction)
    ranker.load()

    # Load pre-trained model
    ranker.load_model()

    # Generate predictions
    print("\nGenerating predictions...")
    probs = ranker.predict()

    print(f"\n✓ Predictions generated!")
    print(f"Number of predictions: {len(probs)}")
    print(f"Score range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"Mean score: {probs.mean():.4f}")
    print(f"Median score: {np.median(probs):.4f}")

    # Show sample predictions
    print("\nSample predictions (first 10):")
    for i, prob in enumerate(probs[:10]):
        print(f"  Sample {i+1}: {prob:.4f}")


def example_custom_save_location():
    """Example: Save and load from custom location"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Save/Load Location")
    print("=" * 80)

    # Initialize config
    config = RankConfig(offline=True, epochs=1)
    ranker = DINRanker(config)

    # Train model
    print("Training model...")
    ranker.train()

    # Save to custom location
    custom_dir = os.path.join(config.save_path, "custom_models", "din_v1")
    ranker.save_model(save_dir=custom_dir)

    print(f"\n✓ Model saved to custom location: {custom_dir}")

    # Load from custom location
    print("\nLoading from custom location...")
    new_ranker = DINRanker(config)
    new_ranker.load()
    new_ranker.load_model(load_dir=custom_dir)

    print("✓ Model loaded from custom location!")


def example_model_versioning():
    """Example: Save multiple model versions"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Model Versioning")
    print("=" * 80)

    config = RankConfig(offline=True, epochs=1)

    # Train and save different versions
    versions = ["v1.0", "v1.1", "v1.2"]

    for version in versions:
        print(f"\n--- Training model {version} ---")
        ranker = DINRanker(config)
        ranker.train()

        # Save with version tag
        version_dir = os.path.join(config.save_path, "models", version)
        ranker.save_model(save_dir=version_dir)
        print(f"✓ Model {version} saved to: {version_dir}")

    # Load specific version
    print(f"\n--- Loading model v1.1 ---")
    ranker = DINRanker(config)
    ranker.load()
    ranker.load_model(load_dir=os.path.join(config.save_path, "models", "v1.1"))
    print("✓ Model v1.1 loaded successfully!")


if __name__ == "__main__":
    import numpy as np

    print("\n" + "=" * 80)
    print("DIN Model Save & Load Examples")
    print("=" * 80)

    # Run examples
    try:
        # Example 1: Basic train and save
        example_train_and_save()

        # Example 2: Load and predict
        example_load_and_predict()

        # Example 3: Custom save location
        # example_custom_save_location()

        # Example 4: Model versioning
        # example_model_versioning()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


# Usage:
# python -m examples.din_save_load_example
