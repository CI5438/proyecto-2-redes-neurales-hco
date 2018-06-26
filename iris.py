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

    @param df, pandas dataframe to drop the column
    @param column, name of the column to drop
    @return pandas dataframe with the dropped column
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
    validation_size = len(df) - training_size 
    
    return df.sample(n=training_size), df.sample(n=validation_size)

def calculate_fit(row, fitset, lower, upper):
    """Stores in the fitset array the data of a row in 
    the interval of [lower, upper) limit.

    @param row where is taken the data
    @param fitset array where is stored the data taken from the row
    @param lower lower limit column to start taking the data from the row
    @param upper upper limit column (not included) to finish taking the data 
    from the row
    @return fitset array filled with data of the row.
    """
    for col in range(lower, upper):
        fitset.append(row[col])
    
    return fitset

def teacher_output(val):
    """The purpose of this function is to make more understandable the evaluation 
    method by taking the name of the term used in neural networks for getting the
    goal value 
    """
    return val

def net_output(network, x_instance):
    """Get the net output (prediction) of the network.
    
    @param network, network where to predict the value
    @param x_instance, instance of a row to predict
    @return predicted array of values
    """
    return network.get_o(x_instance)

def evaluate(network, x, y, is_binary=True):
    """Measures the performance of the network by validating
    predictions with actual values.
    
    This method outputs via print the results.
    
    @param network, network belonging to network class to be tested
    @param x, array instance of one row that the network is using to 
              predict the value
    @param y, expected value (teacher-output) that is going to be
              compared with the prediction
    @param is_binary, used to output false-positives and false-negatives.
                      A non binary network is a network that has more
                      than one column of values to predict.
    @return nothing
    """
    hits, failures, n_inputs = 0, 0, len(x)
    
    if (is_binary):
        false_positive, false_negative = 0, 0 
   
    for i in range(n_inputs):
        # We assume the prediction is going to succeed.
        prediction, hit = net_output(network, x[i]), True
        last_element = len(prediction) - 1
        n_predictions = len(prediction[last_element])
        
        # Round the prediction to zero because the decisition is binary.
        for j in range(n_predictions):
            prediction_rounded = round(prediction[last_element][j], 0)
            actual = teacher_output(y[i][j])
            if (prediction_rounded != actual):
                hit = False
            continue
        
        if hit: 
            hits += 1
        else: 
            failures += 1
            if (is_binary and prediction_rounded == 1.0): false_positive += 1
            elif (is_binary and prediction_rounded == 0.0): false_negative += 1
    
    accuracy = hits*100/(hits+failures)
    error = 100 - accuracy
    print("Acertados: %d, No Acertados: %d" % (hits, failures) )
    print("Precisión: %f%%, Error: %f%%" % (accuracy, error) )
    
    if (is_binary):
        print("Falsos positivos: %d, Falsos negativos: %d" % (false_positive, false_negative) )
    
    return

def fit_data(df, ys=1):
    """Fits data into arrays that are suitable for the Network 
    class.
    
    @param df, panda dataframe
    @param ys, num of values to be predicted. This value is bigger
               than one if the clasification is not binary.
    @return x, data fitted in arrays
    @return y, goal data fitted in arrays 
    """
    x, y = [], []
    ncols = len(df.columns)
    xsize = ncols-ys

    # Calculate x, y subsets 
    for index, row in df.iterrows():
        xdata, ydata = [], []

        # Calculate x subarray for current row
        calculate_fit(row, xdata, 0, xsize)
        x.append(xdata)

        # Calculate y subarray for current row
        calculate_fit(row, ydata, xsize, xsize+ys)
        y.append(ydata)

    return x, y

def print_info(x, n, y):
    """Outputs information of the Network that is going to be
    build.

    @param x, the number of neurons of the input layer
    @param n, the number of neurons of the hidden layer
    @param y, the number of neurons of the output layer
    """
    print("\nCreando una red con las siguientes características:") 
    print("Neuronas por capa: %d entrada, %d intermedia, %d salida" % (x, n, y) )

def start_evaluation(df, ys=1):
    """Trains the network and test it with values.

    @param df, pandas dataframe with the data
    @param ys, number of values that are going to be predicted
    """
    data_size_percentages = [50, 60, 70, 80, 90]
    is_binary = ys==1

    for p in data_size_percentages:
        print("\nEntrenando con el %d porciento de los datos" % p)
        # Split data in p percentage and prepare it to the
        # data type supported by Network Class
        training_df, validation_df = split(df, p)

        # Fit the data to the format that is supported by the
        # Network class 
        x, y = fit_data(training_df, ys)
        x_validation, y_validation = fit_data(validation_df, ys)
        xs = len(x[0])
        
        for n in range(4, 11):
            # Print some information
            print_info(xs, n, ys)
            # Create network
            network = Network([xs, n, ys])
            # Train network
            network.training(1, x, y)
            # Test network
            evaluate(network, x_validation, y_validation, is_binary)
        print("-------------------------------------------")

def init():
    """Main Program. Executes methods for solving third question
    of the project.
    """
    df = read_file("iris.data")
    df = dummies(df)

    setosa_df = setosa_binary_classifier(df)

    print("Clasificador binario")
    start_evaluation(setosa_df, 1)
    
    print("Clasificador de las tres clases")
    start_evaluation(df, 3)

if __name__ == '__main__':
    init()
