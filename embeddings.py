import numpy as np
import pandas as pd
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

# calculate tf-idf embeddings
def get_tfidf_embd(df:pd.DataFrame):
    """
    Get the tf-idf embeddings for the input data
    :param df: pd.DataFrame: input data
    :return X: np.ndarray: tf-idf embeddings
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    X = tfidfconverter.fit_transform(data).toarray()
    return X

def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)

