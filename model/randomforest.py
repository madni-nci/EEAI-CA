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
        # initialize classifier for each target variable
        self.mdl = [RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample') for _ in Config.CLASS_COLS]
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        print('training started')
        # train the classifier for each target variable
        for i in range(len(Config.CLASS_COLS)):
            if i > 0:
               # add actual target (y_train) variable as a feature
               data.X_train =  np.column_stack((data.X_train,data.y_train[Config.CLASS_COLS[i-1]].to_numpy()))
            
            # fit the i_th model
            self.mdl[i].fit(data.X_train, data.y_train[Config.CLASS_COLS[i]])
        print('training finished')

    def predict(self, X_test: pd.Series):
        print('prediction started')

        # predict the classifier for each target variables
        for i in range(len(Config.CLASS_COLS)):
            if i > 0:
                # add prediction of previous model as a feature
                X_test =  np.hstack((X_test, self.predictions[:,i-1].reshape(-1,1)))
            predictions = self.mdl[i].predict(X_test)

            # Initially self.predictions is None, so we need to initialize it
            if i == 0:
                self.predictions = predictions.reshape(-1,1)
            else:
                # append new predictions to the existing predictions
                self.predictions = np.column_stack((self.predictions,predictions))
        
        # Convert the predictions to a pandas dataframe for easy manipulation
        self.predictions = pd.DataFrame(self.predictions)
        print('prediction ended')
    
    def calculateAccuracy(self, row: pd.Series) -> double:
        correct_pred = 0

        # calculate the accuracy for each row
        for value in row:
            if value == True:
                correct_pred += 1
            else:
                # no need to check further if the prediction is wrong
                break

        # len(Config.CLASS_COLS) is the total number of target variables, currently equals to 3
        return correct_pred / len(Config.CLASS_COLS) * 100

    def print_results(self, data):
        results = {f'Type {i+2}':[] for i in range(len(Config.CLASS_COLS))}

        # compare prediction and actual y_test, then store the boolean in results variable
        for i in range(len(Config.CLASS_COLS)):
            results[f'Type {i+2}'] = self.predictions.iloc[:,i].to_numpy() == data.y_test[Config.CLASS_COLS[i]].to_numpy()
        
        # Convert the results to a pandas dataframe for easy manipulation
        results_df = pd.DataFrame(results)

        # Calculate the accuracy for each row
        results_df['Accuracy'] = pd.DataFrame({'accuracy': results_df.apply(self.calculateAccuracy, axis=1)})

        # Calculate the total accuracy for all rows
        total_accuracy = results_df["Accuracy"].mean()

        # print the results for each row
        for i in range(data.test_df.shape[0]):
            print(f'Content: {data.test_df.iloc[i,5]}')
            print(f'Predcited Classes:{list(self.predictions.iloc[i,:].to_numpy())}')
            print(f'True Classes:{list(self.data.y_test.iloc[i,:].to_numpy())}')
            print(f'Accuracy: {self.calculateAccuracy(results_df.iloc[i,:]):.2f}%')
            print('\n')

        return total_accuracy

    def data_transform(self) -> None:
        # Transform the target variables to numerical values
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

