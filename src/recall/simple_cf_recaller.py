from ..dataset import get_click_df

import pandas as pd
import numpy as np
import os, math
from collections import defaultdict, Counter
from typing import Dict, List
import pickle
from tqdm import tqdm

class CF_Recaller:
    def __init__(self, data_path, save_path) -> None:
        self._data_path = data_path
        self._save_path = save_path
        self.user_item_time_dict: Dict[int, List[tuple]]
        self.top_k_item: list = []

        self.click_train_df, self.click_test_df = get_click_df(self._data_path)
        
    def _get_article_timestamp_list(self, df: pd.DataFrame):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    
    def _get_user_item_timestamp(self):
        click_df = self.click_train_df.sort_values(by='click_timestamp')
        user_item_timestamp = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(self._get_article_timestamp_list)\
            .reset_index().rename(columns={0: 'item_time_list'})
            
        self.user_item_time_dict = dict(zip(user_item_timestamp['user_id'], user_item_timestamp['item_time_list']))
    
    def _get_itemcf_sim(self, save: bool=True):
        # 1. 记录每个item的出现次数Counter，维护item与item之间的权重矩阵（一开始定义为dict）
        # 2. 遍历user, item_time_dict in dict
        # 3. 遍历item, time in item_time_dict
        # 3.1 再次遍历item, time in item_time_dict，但是排除本身
        # 3.2 记录交互的item，更新权重
        # 4. 退出循环进行权重的归一化
        
        item_count = Counter()
        item_cf_sim = {}
        
        for user, item_time_list in tqdm(self.user_item_time_dict.items()):
            for i, _ in item_time_list:
                item_count[i] += 1
                item_cf_sim.setdefault(i, {})
                
                for j, _ in item_time_list:
                    if j == i:
                        continue
                    
                    item_cf_sim[i].setdefault(j, 0)
                    item_cf_sim[i][j] += 1 / math.log(len(item_time_list) + 1)
                    
        for i, items in item_cf_sim.items():
            for j, wij in items.items():
                item_cf_sim[i][j] = wij / math.sqrt(item_count[i] * item_count[j])
                
        self._item_cf_sim = item_cf_sim
        if save:
            with open(os.path.join(self._save_path, 'item_cf_sim.pkl'), 'wb') as f:
                pickle.dump(item_cf_sim, f)
            
    def _get_top_k_items(self, k: int=10):
        top_k_item = self.click_train_df['click_article_id'].value_counts().index[:k].tolist()
        self.top_k_item = top_k_item
    
    def train(self, save: bool=True):
        self._get_user_item_timestamp()
        self._get_itemcf_sim(save=save)
        self._get_top_k_items()
        print("finish training, now we get the item-item similarity dict")
        
    def predict_user(self, user_id, sim_item_num: int=10):
        """predict for single user_id

        Args:
            user_id (_type_): _description_
            sim_item_num (int): the number of similar items to return
        """
        cur_items = self.user_item_time_dict[user_id]
        cur_items_set = {item for item, _ in cur_items}
        predict_res = defaultdict(float)
        for i, _ in cur_items:
            sorted_items = sorted(self._item_cf_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_num]
            for j, wij in sorted_items:
                if j in cur_items_set:
                    continue
                
                predict_res.setdefault(j, 0)
                predict_res[j] += wij;
                
        if len(predict_res) < sim_item_num:
            for i, item in enumerate(self.top_k_item):
                if item in predict_res.keys():
                    continue
                predict_res[item] = -i - 100  # set a negative number for weight
                if len(predict_res) == sim_item_num:
                    break
                
        return sorted(predict_res.items(), key=lambda x: x[1], reverse=True)[:sim_item_num]
            
    
    def predict(self, train_data: bool=True, test_data:bool=True):
        if train_data:
            train_data_predict = {}
            print("\nstart prediction for train data:")
            for user_id in tqdm(self.click_train_df['user_id'].unique()):
                train_data_predict[user_id] = self.predict_user(user_id=user_id)
                
            self.train_data_predict = train_data_predict
                
        if test_data:
            test_data_predict = {}
            print("\nstart prediction for test data:")
            for user_id in tqdm(self.click_test_df['user_id'].unique()):
                test_data_predict[user_id] = self.predict_user(user_id=user_id)
                
            self.test_data_predict = test_data_predict
                
        
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'data', 'raw')
    save_path = os.path.join(project_root, 'temp')
    
    cf_recaller = CF_Recaller(data_path=data_path, save_path=save_path)
    cf_recaller.train()
    cf_recaller.predict(train_data=True, test_data=False)
    
if __name__ == "__main__":
    main()