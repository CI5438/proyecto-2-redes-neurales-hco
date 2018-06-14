"""
Universidad Simon Bolivar
Artificial Intelligence II - CI5438

Neural Networks
Iris.py

Authors:
    David Cabeza 1310191
    Rafael Blanco 1310156
    Fabiola Martinez 1310838
"""
import sys

import pandas as pd
from main import *

def drop_column(df, column):
    """Remove column by specifying label names.
    """
    try:
        df = df.drop(column, axis=1)
    except ValueError as err:
        print(err)

    return df

def dummies(df):
    """Create a set of dummy variables from the 'var' variable
    
    @param df dataframe
    @param var variable to create a dummy
    @return dataframe
    """
    return pd.get_dummies(df)

def read_file(filename):
    """Reads file and process it using panda dataframes.
    
    @param name of the file
    @return dataframe
    """
    try:
        df = pd.read_csv(filename)
        return df
    except IOError:
        print('File "%s" could not be read' % filename)
        sys.exit()

def setosa_binary_classifier(df):
    """Binary classifier: separates iris-setosa from the rest.

    @param dataframe with iris dataset already dummied
    @return separated dataframe
    """
    df = drop_column(df, 'Class_Iris-versicolor')
    df = drop_column(df, 'Class_Iris-virginica')
    
    return df

def split(df, p):
    """Splits data in a subset of 'p' percentage 

    @param dataframe to split
    @param p percentage of desired data
    @return sampled dataframe
    """    
    x = (p * len(df))/100
    training_size = round(x) 
    
    return df.sample(n=training_size)

def start_training(df):
    data_size_percentages = [50, 60, 70, 80, 90]

    for p in data_size_percentages:
        # Split data in p percentage and prepare it to the
        # data type supported by Network Class
        training_df = split(df, p)
        #x, y = process_data_for_training(df)
        for n in range(4, 11):
            network = Network([4, n, 1])
            #network.training(n, x, y)

def init():
    """Main Program. Executes methods for solving third question.
    """
    df = read_file("iris.data")
    df = dummies(df)

    setosa_df = setosa_binary_classifier(df)
    
    start_training(setosa_df)
    start_training(df)

if __name__ == '__main__':
    init()
