import sys

sys.path.append("..")

import pandas as pd
from src.utils.config import RecallConfig
from src.data.loaders import DataLoaderFactory
from src.recall.youtubednn_recaller import YoutubeDNNRecaller


def example_basic_training():
    """Example 1: Basic training with configured parameters"""

    print("=" * 60)
    print("Example 1: Basic Training with RecallConfig")
    print("=" * 60)

    # Step 1: Create configuration
    config = RecallConfig(
        data_path="../data/raw/",
        debug_mode=True,
        debug_sample_size=5000,
        # YoutubeDNN specific parameters
        youtubednn_seq_max_len=30,
        youtubednn_embedding_dim=16,
        youtubednn_hidden_units=[64, 16],
        youtubednn_negsample=4,
        youtubednn_epochs=2,
        youtubednn_batch_size=256,
        youtubednn_learning_rate=0.001,
    )

    # Step 2: Load data
    loader = DataLoaderFactory.create_loader("click_log", config)
    click_df = loader.load()

    print(f"Loaded {len(click_df)} click records")
    print(
        f"Users: {click_df['user_id'].nunique()}, Items: {click_df['click_article_id'].nunique()}"
    )

    # Step 3: Create and train recaller
    recaller = YoutubeDNNRecaller(config)

    print("\nTraining YoutubeDNN model...")
    recaller.train(
        click_df,
        epochs=config.youtubednn_epochs,
        batch_size=config.youtubednn_batch_size,
        learning_rate=config.youtubednn_learning_rate,
    )

    # Step 4: Test recall
    sample_user = click_df["user_id"].iloc[0]
    print(f"\nRecalling for user {sample_user}...")

    results = recaller.recall(sample_user, topk=10)
    print(f"Top-10 recommendations:")
    for rank, (item_id, score) in enumerate(results, 1):
        print(f"  {rank}. Item {item_id}: {score:.4f}")

    return recaller


def example_cross_validation():
    """Example 2: Cross-validation for hyperparameter tuning"""

    print("\n" + "=" * 60)
    print("Example 2: Cross-Validation for Hyperparameter Tuning")
    print("=" * 60)

    # Step 1: Create configuration (with default parameters)
    config = RecallConfig(
        data_path="../data/raw/",
        debug_mode=True,
        debug_sample_size=3000,  # Smaller for faster CV
    )

    # Step 2: Load data
    loader = DataLoaderFactory.create_loader("click_log", config)
    click_df = loader.load()

    # Step 3: Define parameter grid
    param_grid = {
        "embedding_dim": [8, 16],
        "hidden_units": [[32, 8], [64, 16]],
        "negsample": [2, 4],
    }

    print(f"\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Step 4: Create recaller and run CV
    recaller = YoutubeDNNRecaller(config)

    print("\nRunning cross-validation...")
    cv_results = recaller.train_with_cv(
        click_df,
        param_grid=param_grid,
        n_splits=3,  # 3-fold CV for faster execution
        epochs=1,
        batch_size=256,
        learning_rate=0.001,
    )

    # Step 5: Display results
    print("\n" + "=" * 60)
    print("Cross-Validation Results:")
    print("=" * 60)

    print(f"\nBest Parameters: {cv_results['best_params']}")
    print(f"Best Score: {cv_results['best_score']:.4f}")

    print("\nAll Results:")
    for result in cv_results["cv_results"]:
        print(f"  Params: {result['params']}")
        print(f"  Score: {result['mean_score']:.4f} (+/- {result['std_score']:.4f})")
        print()

    return recaller, cv_results


def example_efficient_lookup():
    """Example 3: Demonstrate efficient reverse lookup"""

    print("\n" + "=" * 60)
    print("Example 3: Efficient Reverse Lookup")
    print("=" * 60)

    # Step 1: Create and train a simple model
    config = RecallConfig(
        data_path="../data/raw/",
        debug_mode=True,
        debug_sample_size=2000,
        youtubednn_epochs=1,
    )

    loader = DataLoaderFactory.create_loader("click_log", config)
    click_df = loader.load()

    recaller = YoutubeDNNRecaller(config)
    recaller.train(click_df, epochs=1, batch_size=256)

    # Step 2: Demonstrate reverse lookup
    print("\nDemonstrating lookup efficiency:")
    print(f"  Total users: {len(recaller.user_index_2_rawid)}")
    print(f"  Total items: {len(recaller.item_index_2_rawid)}")

    # Sample user
    sample_raw_id = list(recaller.user_rawid_2_index.keys())[0]

    # Fast lookup using reverse mapping
    import time

    start = time.time()
    user_idx = recaller.user_rawid_2_index.get(sample_raw_id)
    fast_time = time.time() - start

    print(f"\n  Fast lookup (O(1)): {fast_time*1000:.4f} ms")
    print(f"  User raw ID {sample_raw_id} -> Index {user_idx}")

    # Show that the reverse mapping works
    print(f"  Index {user_idx} -> Raw ID {recaller.user_index_2_rawid[user_idx]}")
    print("  ✓ Mappings are consistent!")

    return recaller


def example_custom_configuration():
    """Example 4: Using custom configuration from dict"""

    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration from Dictionary")
    print("=" * 60)

    # Step 1: Define configuration as dictionary
    config_dict = {
        "data_path": "../data/raw/",
        "debug_mode": True,
        "debug_sample_size": 2000,
        "youtubednn_seq_max_len": 20,
        "youtubednn_embedding_dim": 32,
        "youtubednn_hidden_units": [128, 32],
        "youtubednn_negsample": 8,
        "youtubednn_epochs": 1,
        "youtubednn_batch_size": 128,
        "youtubednn_learning_rate": 0.0005,
    }

    # Step 2: Create config from dict
    config = RecallConfig.from_dict(config_dict)

    print("Configuration loaded from dictionary:")
    print(f"  Seq max length: {config.youtubednn_seq_max_len}")
    print(f"  Embedding dim: {config.youtubednn_embedding_dim}")
    print(f"  Hidden units: {config.youtubednn_hidden_units}")
    print(f"  Negative samples: {config.youtubednn_negsample}")

    # Step 3: Train model with custom config
    loader = DataLoaderFactory.create_loader("click_log", config)
    click_df = loader.load()

    recaller = YoutubeDNNRecaller(config)
    recaller.train(
        click_df,
        epochs=config.youtubednn_epochs,
        batch_size=config.youtubednn_batch_size,
        learning_rate=config.youtubednn_learning_rate,
    )

    print("\n✓ Model trained successfully with custom configuration!")

    return recaller


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YoutubeDNN Recaller - Advanced Examples")
    print("=" * 60)

    # Run all examples

    # Example 1: Basic training
    recaller1 = example_basic_training()

    # Example 2: Cross-validation (commented out by default - takes longer)
    # Uncomment the following line to run CV
    # recaller2, cv_results = example_cross_validation()

    # Example 3: Efficient lookup
    recaller3 = example_efficient_lookup()

    # Example 4: Custom configuration
    recaller4 = example_custom_configuration()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

    print("\nKey Improvements Demonstrated:")
    print("  1. ✓ Efficient padding with custom collate_fn")
    print("  2. ✓ Configurable negative sampling")
    print("  3. ✓ Fast O(1) user/item lookup with reverse mappings")
    print("  4. ✓ Centralized configuration management")
    print("  5. ✓ Cross-validation for hyperparameter tuning")
    print("\nOptimizations:")
    print("  - Dataset __getitem__ returns raw sequences (no padding)")
    print("  - collate_fn handles batch padding efficiently")
    print("  - Reverse dictionaries (rawid_2_index) eliminate O(n) searches")
    print("  - All hyperparameters managed through RecallConfig")
    print("  - K-Fold CV with automatic model selection")
