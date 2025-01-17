from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    """
    Load the input data
    :return df: pd.DataFrame: input data
    """
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    """
    Preprocess the input data
    :param df: pd.DataFrame: input data
    :return df: pd.DataFrame: preprocessed data
    """
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    """
    Get the embeddings for the input data
    :param df: pd.DataFrame: input data
    :return X: np.ndarray: embeddings
    """
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    """
    Get the data object
    :param X: np.ndarray: embeddings
    :param df: pd.DataFrame: input data
    :return data: Data: data object
    """
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame):
    """
    Perform the modelling
    :param data: Data: data object
    :param df: pd.DataFrame: input data
    :return accuracy: double: accuracy of the model
    """
    return model_predict(data, df)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)
    accuracies = {name:[] for name, group_df in grouped_df}
    for name, group_df in grouped_df:
        # Get TF-IDF embeddings
        X, group_df = get_embeddings(group_df)

        # Get the data object
        data = get_data_object(X, group_df)

        # Train, predict and calculate the accuracy
        accuracies[name] = perform_modelling(data, group_df)
    
    # print the accuracies
    for name, group_df in grouped_df:
        print(f'Average Accuracy for {name} group: {accuracies[name]:.2f}%')
    print(f'Overall Average Accuracy for all groups: {np.mean(list(accuracies.values())):.2f}%')