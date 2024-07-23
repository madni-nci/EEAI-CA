import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
from Config import *
import random

num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str, data) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.data = data
        self.mdl = [RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample') for _ in Config.CLASS_COLS]
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        print('training started')
        self.mdl =[self.mdl[i].fit(data.X_train, data.y_train[Config.CLASS_COLS[i]]) for i in range(len(Config.CLASS_COLS))]
        print('training finished')

    def predict(self, X_test: pd.Series):
        print('prediction started')
        predictions = {Config.CLASS_COLS[i]: self.mdl[i].predict(X_test) for i in range(len(Config.CLASS_COLS))}
        self.predictions = pd.DataFrame(predictions)
        print('prediction ended')
    
    def calculateAccuracy(self, row: pd.Series) -> double:
        correct_pred = 0
        for value in row:
            if value == True:
                correct_pred += 1
            else:
                break
        return correct_pred / len(Config.CLASS_COLS) * 100

    def print_results(self, data):
        results = {f'Type {i+2}':[] for i in range(len(Config.CLASS_COLS))}

        for i in range(len(Config.CLASS_COLS)):
            results[f'Type {i+2}'] = self.predictions.iloc[:,i].to_numpy() == data.y_test[Config.CLASS_COLS[i]].to_numpy()
        
        results_df = pd.DataFrame(results)

        results_df['Accuracy'] = pd.DataFrame({'accuracy': results_df.apply(self.calculateAccuracy, axis=1)})
        total_accuracy = results_df["Accuracy"].mean()

        for i in range(data.test_df.shape[0]):
            print(f'Content: {data.test_df.iloc[i,5]}')
            print(f'Predcited Classes:{list(self.predictions.iloc[i,:].to_numpy())}')
            print(f'True Classes:{list(self.data.y_test.iloc[i,:].to_numpy())}')
            print(f'Accuracy: {self.calculateAccuracy(results_df.iloc[i,:])}%')
            print('\n')
        
        print('\n')

        return total_accuracy

        # print(f'Total Accuracy: {results_df['Accuracy'].mean()}')
        # print(classification_report(data.y_test, self.predictions))


    def data_transform(self) -> None:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        self.data.y_train = pd.DataFrame(self.data.get_type_y_train())
        self.data.y_test = pd.DataFrame(self.data.get_type_y_test())

        for column in Config.CLASS_COLS:
            le = LabelEncoder()
            # Fit and transform the data for each column
            le.fit(self.data.y_train[column])
            self.data.y_train[column] = le.transform(self.data.y_train[column])
            self.data.y_test[column] = le.transform(self.data.y_test[column])
