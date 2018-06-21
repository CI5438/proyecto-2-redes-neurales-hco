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

def calculate_fit(row, fitset, size)
    for col in range(size):
        fitset.append(row[col])
    
    return fitset


def fit_data(df, ys=1):
    """
    """
    x, y = [], []
    ncols = len(df.columns)
    xsize = ncols-ys

    # Calculate x, y subsets 
    for index, row in df.iterrows():
        xdata, ydata = [], []

        # Calculate x subarray for current row
        xdata = calculate_fit(row, xdata, xsize)
        x.append(xdata)

        # Calculate y subarray for current row
        ydata = calculate_fit(row, ydata, ys)
        y.append(ydata)

    return x, y

def print_info(p, x, n, y):
    print("\nCreando una red con las siguientes características") 
    print("Neuronas: %d entrada, %d capa oculta, %d salida" % (x, n, y) )

def start_training(df, ys=1):
    """
    """
    data_size_percentages = [50, 60, 70, 80, 90]

    for p in data_size_percentages:
        print("\nEntrenando con el %d porciento de los datos" % p)
        # Split data in p percentage and prepare it to the
        # data type supported by Network Class
        training_df = split(df, p)

        # Fit the data to the format supported by the
        # Network class 
        x, y = fit_data(training_df, ys)
        xs = len(x[0])
        
        for n in range(4, 11):
            print_info(p, xs, n, ys)
            network = Network([xs, n, ys])
            network.training(1, x, y)
        print("-------------------------------------------")

    return 

def init():
    """Main Program. Executes methods for solving third question.
    """
    df = read_file("iris.data")
    df = dummies(df)

    setosa_df = setosa_binary_classifier(df)
    
    start_training(setosa_df, 1)
    start_training(df, 3)

if __name__ == '__main__':
    init()
