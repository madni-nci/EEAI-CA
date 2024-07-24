from model.randomforest import RandomForest



def model_predict(data, df):
    """
    Perform the modelling
    :param data: Data: data object
    :param df: pd.DataFrame: input data
    :return accuracy: double: accuracy of the model
    """
    results = []
    model = RandomForest("RandomForest", data)
    model.train(data)
    model.predict(data.X_test)
    return model.print_results(data)



def model_evaluate(model, data):
    model.print_results(data)