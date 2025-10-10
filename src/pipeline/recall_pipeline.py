import os, sys

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

class RecallPipeLine:
    def __init__(self, config):
        self.config = config
    
    def load(self):
        click_loader = ClickLogLoader(self.config)
        article_loader = ArticleInfoLoader(self.config)
        self.all_clicks = click_loader.load_all(debug=self.config.debug_mode, offline=self.config.offline)
        self.articles_df, self.articles_emb_df = article_loader.load(debug=self.config.debug_mode)
        self.all_clicks['click_timestamp'] = InteractionFeatureExtractor.normalize_timestamp(self.all_clicks)
        print(f"Loaded {len(self.all_clicks)} click records")
        print(f"Loaded {len(self.articles_df)} articles")
        
        print("start extracting features...")
        self.item_type_dict, self.item_words_dict, self.item_created_time_dict = (
            ItemFeatureExtractor.get_item_info_dict(self.articles_df)
        )
        
        self.item_topk_click = ItemFeatureExtractor.get_item_topk_click(self.all_clicks, k=self.config.itemcf_hot_topk)
        
        self.user_item_time_dict = UserFeatureExtractor.get_user_item_time_dict(self.all_clicks)
        self.user_activate_degree_dict = UserFeatureExtractor.get_user_activate_degree_dict(self.all_clicks)
        print("features extracted")
        
    def calculate_similarity(
        self, 
        item_cf: bool=True, 
        user_cf: bool=True, 
        embedding_cf: bool=True
    ):
        if not hasattr(self, 'all_clicks'):
            raise ValueError("Please run load() before calculating similarity")
        
        if item_cf:
            print("start calculating ItemCF similarity...")
            itemcf_sim_calc = ItemCFSimilarity(self.config)
            self.i2i_sim = itemcf_sim_calc.calculate(self.all_clicks, self.item_created_time_dict)
            print("ItemCF similarity calculated")
        
        if user_cf:
            print("start calculating UserCF similarity...")
            usercf_sim_calc = UserCFSimilarity(self.config)
            self.usercf_sim_matrix = usercf_sim_calc.calculate(self.all_clicks, self.user_activate_degree_dict)
            print("UserCF similarity calculated")
        
        if embedding_cf:
            print("start calculating Embedding similarity...")
            embedding_sim_clac = EmbeddingSimilarity(self.config)
            self.embedding_sim_matrix = embedding_sim_clac.calculate(self.articles_emb_df, self.config.embedding_topk)
            print("Embedding similarity calculated")
            
        print("all similarity calculations done")
            
    def fusion_recall(self, save: bool=True, results: dict=None):
        """
        fuse the recall results from itemcf, usercf and youtubednn methods
        you should run calculate_similarity() before calling this function if results is None

        Args:
            save: Whether to save the fused results
            results: Optional precomputed recall results to use instead of running recall again
                it should be like {
                    "itemcf": {user_id: [(item_id, score), ...], ...},
                    "usercf": {user_id: [(item_id, score), ...], ...},
                    "youtubednn": {user_id: [(item_id, score), ...], ...}
                }
                
        Return:
            Fused recall results: {user_id: [(item_id, score), ...], ...}
        """
        if results is not None and "itemcf" in results and "usercf" in results and "youtubednn" in results:
            print("using provided recall results for fusion")

            for mn in results:
                print(f"method {mn}: {len(results[mn])} users")
                
            self.recall_results = results
        else:
            if not hasattr(self, 'i2i_sim') or not hasattr(self, 'usercf_sim_matrix'):
                raise ValueError("Please run calculate_similarity() before fusion_recall()")
            
            print("start recalling...")
            all_users = list(set(self.user_item_time_dict.keys()))
            itemcf_recaller = ItemCFRecaller(
                config=self.config,
                similarity_matrix=self.i2i_sim,
                item_created_time_dict=self.item_created_time_dict,
                user_item_time_dict=self.user_item_time_dict,
                item_topk_click=self.item_topk_click,
                emb_similarity_matrix=self.embedding_sim_matrix
            )
            itemcf_results = itemcf_recaller.batch_recall(all_users, topk=self.config.recall_topk)
            
            usercf_recaller = UserCFRecaller(
                config=self.config,
                similarity_matrix=self.usercf_sim_matrix,
                user_item_time_dict=self.user_item_time_dict,
                item_created_time_dict=self.item_created_time_dict,
                item_topk_click=self.item_topk_click,
                emb_similarity_matrix=self.embedding_sim_matrix
            )
            usercf_results = usercf_recaller.batch_recall(all_users, topk=self.config.recall_topk)
            
            youtubednn_recaller = YoutubeDNNRecaller(config=self.config)
            youtubednn_recaller.train(self.all_clicks)
            youtubednn_results = youtubednn_recaller.batch_recall(all_users, topk=self.config.recall_topk)
            
            self.recall_results = {
                "itemcf": itemcf_results,
                "usercf": usercf_results,
                "youtubednn": youtubednn_results
            }
            
            if save:
                PersistenceManager.save_pickle(
                    self.recall_results, os.path.join(self.config.save_path, "all_recall_results.pkl")
                )
                
            
        print("start fusing recall results...")
        fusion = RecallFusion(self.config)
        fusion.add_recall_result("itemcf", self.recall_results["itemcf"])
        fusion.add_recall_result("usercf", self.recall_results["usercf"])
        fusion.add_recall_result("youtubednn", self.recall_results["youtubednn"])
        self.fused_results = fusion.fuse(topk=self.config.recall_topk)

        if save:
            PersistenceManager.save_pickle(
                self.fused_results, os.path.join(self.config.save_path, "fused_recall_results.pkl")
            )
            
        print("recall fusion done")
        
        return self.fused_results
            
            
        

        

        
        
        
        
        