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


def read_file(filename):
    """
    Reads file and process it using panda dataframes.
    
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
    """
    Main Program. Executes methods for solving third question.
    """
    df = read_file("iris.data")

if __name__ == '__main__':
    init()
