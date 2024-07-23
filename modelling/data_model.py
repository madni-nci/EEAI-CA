import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        
        y = df[Config.CLASS_COLS]
        y.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.y_train = {target:[] for target in Config.CLASS_COLS}
        # self.X_test = {target:[] for target in Config.CLASS_COLS}
        self.y_test = {target:[] for target in Config.CLASS_COLS}

        data_idx = list(range(0, X.shape[0]))

        random.seed(0)

        random.shuffle(data_idx)

        split_index = int(0.8 * len(data_idx))

        self.X_train = X[data_idx[:split_index]]
        self.X_test = X[data_idx[split_index:]]
        self.test_df = df.iloc[data_idx[split_index:], :]
        
        for target in Config.CLASS_COLS:
            self.y_train[target], self.y_test[target] = y[target][data_idx[:split_index]], y[target][data_idx[split_index:]]
            
        self.embeddings = X # df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]


    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df
    def get_X_DL_test(self):
        return self.X_DL_test
    def get_X_DL_train(self):
        return self.X_DL_train

