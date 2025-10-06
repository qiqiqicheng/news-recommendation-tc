import pandas as pd
import numpy as np
import os

def get_click_df(path):
    """get click dataframe(train + test)

    Args:
        path (_type_): _description_
    Returns:
        click_train_df, click_test_df
    """
    click_train_df = pd.read_csv(os.path.join(path, 'train_click_log.csv'))
    # click_train_df = pd.read_csv(path + 'train_click_log.csv')
    click_test_df = pd.read_csv(os.path.join(path, 'testA_click_log.csv'))
    # click_test_df = pd.read_csv(path + 'testA_click_log.csv')
    return click_train_df, click_test_df

def get_article_df(path):
    """get article dataframe(meta + embedding)

    Args:
        path (_type_): _description_
    Returns:
        articles_df, articles_emb_df
    """
    articles_df = pd.read_csv(os.path.join(path, 'articles.csv'))
    articles_emb_df = pd.read_csv(os.path.join(path, 'articles_emb.csv'))
    return articles_df, articles_emb_df

def get_sample_submit(path):
    """get sample submit dataframe

    Args:
        path (_type_): _description_
    """
    sample_submit_df = pd.read_csv(os.path.join(path, 'submit_example.csv'))
    return sample_submit_df

