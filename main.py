import argparse
import os
import sys
from pathlib import Path
import traceback

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.recall_pipeline import RecallPipeline
from src.pipeline.rank_pipeline import RankPipeline
from src.utils.config import RecallConfig, RankConfig
from src.utils.persistence import PersistenceManager


def parse_args():
    """
    Parse command line arguments.
    Maybe remaining some bug?
    
    Built from AI (Claude Sonnet 4.5)
    """
    parser = argparse.ArgumentParser(
        description="News Recommendation System - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "recall", "rank", "inference"],
        help="Pipeline mode: full (recall+rank), recall only, rank only, or inference",
    )

    # General settings
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Offline mode (exclude last click for training labels)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (use small sample of data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for reproducibility",
    )

    # Recall stage settings
    parser.add_argument(
        "--recall-topk",
        type=int,
        default=30,
        help="Number of candidates to recall per user",
    )

    # Ranking stage settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs for DIN model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension for DIN model",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (0 for CPU, 4-8 for GPU)",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=10.0,
        help="Negative to positive sampling ratio",
    )
    parser.add_argument(
        "--no-negative-sampling",
        action="store_true",
        help="Disable negative sampling",
    )

    # Recommendation settings
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of final recommendations per user",
    )

    # Model paths
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to load pre-trained model from (for inference mode)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./temp",
        help="Directory to save models and results",
    )

    return parser.parse_args()


def run_recall_stage(recall_config: RecallConfig):
    """
    Execute recall stage to generate candidate items.

    Args:
        recall_config: Configuration for recall pipeline
    """
    recall_pipeline = RecallPipeline(recall_config)

    print("\nBuilding similarity matrices...")
    recall_pipeline.build_similarities()

    print("\nGenerating recall results...")
    recall_pipeline.recall()

    print("\nFusing recall results...")
    all_recall_results = recall_pipeline.fuse()

    print("\nRecall stage completed!")
    print(f"  Total users: {len(all_recall_results)}")

    return recall_pipeline


def run_ranking_stage(recall_config: RecallConfig, rank_config: RankConfig, args):
    """
    Execute ranking stage to train model and rank candidates.

    Args:
        recall_config: Configuration for recall pipeline (needed for feature extraction)
        rank_config: Configuration for ranking pipeline
        args: Command line arguments
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "RANKING STAGE")
    print("=" * 80)

    rank_pipeline = RankPipeline(rank_config)

    print("\nExtracting features...")
    rank_pipeline.extract_features(recall_config)

    if rank_config.offline:
        print("\nTraining DIN model...")
        rank_pipeline.train()
    else:
        print("\nSkipping training (online mode)")

    print("\nGenerating rankings...")
    recommendations = rank_pipeline.rank_and_recommend(
        top_k=args.top_k,
        save_path=os.path.join(rank_config.save_path, "final_recommendations.pkl"),
    )

    print("\nRanking stage completed!")
    print(f"  Total recommendations: {len(recommendations)}")

    return rank_pipeline, recommendations


def run_inference(recall_config: RecallConfig, rank_config: RankConfig, args):
    """
    Run inference with pre-trained models.

    Args:
        recall_config: Configuration for recall pipeline
        rank_config: Configuration for ranking pipeline
        args: Command line arguments
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "INFERENCE MODE")
    print("=" * 80)

    # Load pre-trained model
    rank_pipeline = RankPipeline(rank_config)

    print("\nLoading pre-trained model...")
    rank_pipeline.load_model(model_dir=args.model_dir)

    print("\nGenerating predictions...")
    recommendations = rank_pipeline.rank_and_recommend(
        top_k=args.top_k,
        save_path=os.path.join(rank_config.save_path, "final_recommendations.pkl"),
    )

    print("\nInference completed!")
    print(f"  Total predictions: {len(recommendations)}")

    return recommendations


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize configurations
    recall_config = RecallConfig(
        debug_mode=args.debug,
        offline=args.offline,
        random_seed=args.seed,
        fuse_topk=args.recall_topk,
    )

    rank_config = RankConfig(
        debug_mode=args.debug,
        offline=args.offline,
        random_seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        din_embedding_dim=args.embedding_dim,
        num_workers=args.num_workers,
        enable_negative_sampling=not args.no_negative_sampling,
        negative_positive_ratio=args.negative_ratio,
    )

    # Override save path if specified
    if args.save_dir != "./temp":
        recall_config.save_path = args.save_dir
        rank_config.save_path = args.save_dir
        os.makedirs(args.save_dir, exist_ok=True)

    # Execute pipeline based on mode
    try:
        if args.mode == "full":
            # Full pipeline: recall → ranking
            print("Running FULL pipeline (Recall + Ranking)")

            # Stage 1: Recall
            recall_pipeline = run_recall_stage(recall_config)

            # Stage 2: Ranking
            rank_pipeline, recommendations = run_ranking_stage(
                recall_config, rank_config, args
            )

            print(" " * 20 + "PIPELINE COMPLETED SUCCESSFULLY!")

        elif args.mode == "recall":
            # Only recall stage
            print("Running RECALL stage only")
            recall_pipeline = run_recall_stage(recall_config)

            print(" " * 20 + "RECALL STAGE COMPLETED!")

        elif args.mode == "rank":
            # Only ranking stage (assumes recall results exist)
            print("Running RANKING stage only")
            print("(Assuming recall results exist in temp/)")

            rank_pipeline, recommendations = run_ranking_stage(
                recall_config, rank_config, args
            )

            print(" " * 20 + "RANKING STAGE COMPLETED!")

        elif args.mode == "inference":
            # Inference mode (use pre-trained models)
            print("Running INFERENCE mode")

            if args.model_dir is None:
                args.model_dir = args.save_dir
                print(f"No model directory specified, using: {args.model_dir}")

            recommendations = run_inference(recall_config, rank_config, args)

            print(" " * 20 + "INFERENCE COMPLETED!")

        print("\n✓ All tasks completed successfully!")
        print(f"Results saved to: {args.save_dir}")

    except Exception as e:
        print(" " * 25 + "ERROR OCCURRED")
        print(f"\nError: {str(e)}")
        
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
