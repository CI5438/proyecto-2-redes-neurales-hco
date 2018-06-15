"""
Universidad Simon Bolivar
Artificial Intelligence II - CI5438

Neural Networks

Authors:
    David Cabeza 1310191
    Rafael Blanco 1310156
    Fabiola Martinez 1310838
"""

from project_1 import * 
from generador import *

def read_dataset(filename):
    dataset = open(filename, "r")
    t = []
    goal=[]

    lines=dataset.readlines()
    
    for line in lines:
        t.append([line.split(' ')[0],line.split(' ')[1]])
        goal.append([(line.split(' ')[2]).rstrip()]) # removing /n
    dataset.close()

    return t,goal

n = Network([2, 2, 1])
x,y=read_dataset("datosP2_AJ2018_B1_N1000.txt")
n.training(0.01,x,y)
