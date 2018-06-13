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

def init():
    """Main Program. Executes methods for solving third question.
    """
    df = read_file("iris.data")
    df = dummies(df)
    df = drop_column(df, 'Class_Iris-versicolor')
    df = drop_column(df, 'Class_Iris-virginica')
    #df.to_csv('x.txt')

if __name__ == '__main__':
    init()
