import os, sys
import pandas as pd
from typing import Optional, Dict, List, Tuple, Literal

from ..recall.fusion import RecallFusion, RecallEnsemble, RecallConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import RecallConfig
from src.utils.persistence import PersistenceManager
from src.data.loaders import ClickLogLoader, ArticleInfoLoader
from src.data.extractors import (
    UserFeatureExtractor,
    ItemFeatureExtractor,
    InteractionFeatureExtractor,
)
from src.similarity import ItemCFSimilarity, UserCFSimilarity, EmbeddingSimilarity
from src.recall import ItemCFRecaller, UserCFRecaller, RecallFusion, YoutubeDNNRecaller


class RecallPipeline:
    def __init__(self, config):
        self.config = config

    def load(self):
        """
        Load data for recall pipeline

        NOTE: In offline mode (training), MUST exclude last click from history
        - This ensures recall algorithms don't use ground truth
        - Enables proper positive/negative sample generation
        - Simulates real-world recommendation scenario
        """
        click_loader = ClickLogLoader(self.config)
        article_loader = ArticleInfoLoader(self.config)

        # Load click data based on mode
        if self.config.debug_mode:
            # Debug mode: sample users, respect offline mode
            train_clicks, test_clicks = click_loader.load(
                debug=True, sample_size=self.config.debug_sample_size
            )

            # In offline mode: exclude last click from training data for recall
            if self.config.offline:
                print(
                    "Offline mode: excluding last click from training history for recall"
                )
                train_history, self.train_last_click = (
                    InteractionFeatureExtractor.get_hist_and_last_click(
                        train_clicks, offline=True
                    )
                )
                # Use history (without last click) for recall + all test data
                self.all_clicks = pd.concat(
                    [train_history, test_clicks], ignore_index=True
                )
                print(f"  Train history (excl. last): {len(train_history)} records")
                print(
                    f"  Train last clicks (ground truth): {len(self.train_last_click)} records"
                )
                print(f"  Test data: {len(test_clicks)} records")
            else:
                # Online mode: use all data
                print("Online mode: using all training data for recall")
                self.all_clicks = pd.concat(
                    [train_clicks, test_clicks], ignore_index=True
                )
                self.train_last_click = pd.DataFrame()

            # Load corresponding articles
            self.articles_df, self.articles_emb_df = (
                article_loader.load_with_user_sample(
                    self.all_clicks["click_article_id"].unique().tolist()
                )
            )
        else:
            # Normal mode: load all data, respect offline mode
            train_clicks, test_clicks = click_loader.load(debug=False)

            if self.config.offline:
                print(
                    "Offline mode: excluding last click from training history for recall"
                )
                train_history, self.train_last_click = (
                    InteractionFeatureExtractor.get_hist_and_last_click(
                        train_clicks, offline=True
                    )
                )
                self.all_clicks = pd.concat(
                    [train_history, test_clicks], ignore_index=True
                )
                print(f"  Train history (excl. last): {len(train_history)} records")
                print(
                    f"  Train last clicks (ground truth): {len(self.train_last_click)} records"
                )
                print(f"  Test data: {len(test_clicks)} records")
            else:
                print("Online mode: using all training data for recall")
                self.all_clicks = pd.concat(
                    [train_clicks, test_clicks], ignore_index=True
                )
                self.train_last_click = pd.DataFrame()

            self.articles_df, self.articles_emb_df = article_loader.load(debug=False)

        # Remove duplicates
        self.all_clicks = self.all_clicks.drop_duplicates(
            ["user_id", "click_article_id", "click_timestamp"]
        )

        print(f"Total click records for recall: {len(self.all_clicks)}")
        print(f"Loaded {len(self.articles_df)} articles")

        print("start extracting features...")
        self.item_type_dict, self.item_words_dict, self.item_created_time_dict = (
            ItemFeatureExtractor.get_item_info_dict(self.articles_df)
        )

        self.item_topk_click = ItemFeatureExtractor.get_item_topk_click(
            self.all_clicks, k=self.config.itemcf_hot_topk
        )

        # CRITICAL: user_item_time_dict is built from self.all_clicks
        # In offline mode, this already excludes last clicks
        self.user_item_time_dict = UserFeatureExtractor.get_user_item_time_dict(
            self.all_clicks
        )
        self.user_activate_degree_dict = (
            UserFeatureExtractor.get_user_activate_degree_dict(self.all_clicks)
        )
        print("features extracted")

    def calculate_similarity(
        self,
        item_cf: bool = True,
        # user_cf: bool=True,
        embedding_cf: bool = True,
    ):
        if not hasattr(self, "all_clicks"):
            raise ValueError("Please run load() before calculating similarity")

        if item_cf:
            print("start calculating ItemCF similarity...")
            itemcf_sim_calc = ItemCFSimilarity(self.config)
            self.i2i_sim = itemcf_sim_calc.calculate(
                self.all_clicks, self.item_created_time_dict
            )
            print("ItemCF similarity calculated")

        # if user_cf:
        #     print("start calculating UserCF similarity...")
        #     usercf_sim_calc = UserCFSimilarity(self.config)
        #     self.usercf_sim_matrix = usercf_sim_calc.calculate(self.all_clicks, self.user_activate_degree_dict)
        #     print("UserCF similarity calculated")

        if embedding_cf:
            print("start calculating Embedding similarity...")
            embedding_sim_clac = EmbeddingSimilarity(self.config)
            self.embedding_sim_matrix = embedding_sim_clac.calculate(
                self.articles_emb_df
            )
            print("Embedding similarity calculated")

        print("all similarity calculations done")

    def fusion_recall(
        self,
        save: bool = True,
        results: Optional[Dict] = None,
        fusion_strategy: Literal[
            "weighted_sum",
            "weighted_avg",
            "max_score",
            "harmonic_mean",
            "diversity_weighted",
            "rrf",
        ] = "weighted_avg",
        normalize_method: Literal["local", "global", "z-score"] = "global",
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Fuse the recall results from itemcf, usercf and youtubednn methods

        Args:
            save: Whether to save the fused results and individual method results
            results: Optional precomputed recall results to use instead of running recall again
                Format: {
                    "itemcf": {user_id: [(item_id, score), ...], ...},
                    "youtubednn": {user_id: [(item_id, score), ...], ...}
                }
            fusion_strategy: Fusion strategy to use
                Options: weighted_sum, weighted_avg, max_score, harmonic_mean, diversity_weighted, rrf
                Default: weighted_avg (recommended)
            normalize_method: Normalization method
                Options: local, global, z-score
                Default: global (recommended)
            weights: Optional custom weights for each method
                Format: {"itemcf": 1.0, "youtubednn": 1.2}
                Default: {"itemcf": 1.0, "youtubednn": 1.0}

        Returns:
            Fused recall results: {user_id: [(item_id, score), ...], ...}
        """
        # Set default weights
        if weights is None:
            weights = {"itemcf": 1.0, "youtubednn": 1.0}

        if results is not None and "itemcf" in results and "youtubednn" in results:
            print("Using provided recall results for fusion")
            for method_name in results:
                print(f"  Method '{method_name}': {len(results[method_name])} users")
            self.recall_results = results
        else:
            if not hasattr(self, "i2i_sim"):
                raise ValueError("Run calculate_similarity() before fusion_recall()")

            print("Starting recall for each method...")
            print(
                f"Mode: {'Offline (training)' if self.config.offline else 'Online (evaluation)'}"
            )

            all_users = list(set(self.user_item_time_dict.keys()))
            print(f"Total users for recall: {len(all_users)}")

            # ItemCF recall
            # NOTE: user_item_time_dict already excludes last click in offline mode (built from self.all_clicks)
            print("\nItemCF Recall...")
            itemcf_recaller = ItemCFRecaller(
                config=self.config,
                similarity_matrix=self.i2i_sim,
                item_created_time_dict=self.item_created_time_dict,
                user_item_time_dict=self.user_item_time_dict,  # Excludes last click in offline mode
                item_topk_click=self.item_topk_click,
                emb_similarity_matrix=self.embedding_sim_matrix,
            )
            itemcf_results = itemcf_recaller.batch_recall(
                all_users, topk=self.config.itemcf_recall_num
            )

            # YoutubeDNN recall
            # NOTE: self.all_clicks already excludes last click in offline mode
            print("\nYoutubeDNN Recall...")
            youtubednn_recaller = YoutubeDNNRecaller(config=self.config)
            youtubednn_recaller.train(
                self.all_clicks,  # Excludes last click in offline mode
                epochs=self.config.youtubednn_epochs,
                batch_size=self.config.youtubednn_batch_size,
                learning_rate=self.config.youtubednn_learning_rate,
            )

            # Build and save embeddings
            youtubednn_recaller.construct_embedding_dict(save=True)

            youtubednn_results = youtubednn_recaller.batch_recall(
                all_users, topk=self.config.youtubednn_topk
            )

            # Combine all results
            self.recall_results = {
                "itemcf": itemcf_results,
                "youtubednn": youtubednn_results,
            }

            # Save all method results together
            if save:
                all_methods_path = os.path.join(
                    self.config.save_path, "all_recall_methods_results.pkl"
                )
                PersistenceManager.save_pickle(self.recall_results, all_methods_path)

        # Fusion with enhanced options
        print(
            f"\nStarting fusion with strategy='{fusion_strategy}', normalization='{normalize_method}'"
        )
        fusion = RecallFusion(
            self.config,
            fusion_strategy=fusion_strategy,
            normalize_method=normalize_method,
        )

        # Add recall results with weights
        for method_name, method_results in self.recall_results.items():
            weight = weights.get(method_name, 1.0)
            fusion.add_recall_result(method_name, method_results, weight=weight)

        # Perform fusion
        self.fused_results = fusion.fuse(topk=self.config.fuse_topk)

        # Save fused results
        if save:
            PersistenceManager.save_pickle(
                self.fused_results, self.config.all_recall_results_path
            )

        return self.fused_results


if __name__ == "__main__":
    config = RecallConfig()
    pipeline = RecallPipeline(config)
    pipeline.load()
    pipeline.calculate_similarity()
    fused_results = pipeline.fusion_recall()

    # python -m src.pipeline.recall_pipeline
