"""
Universidad Simon Bolivar
Artificial Intelligence II - CI5438

Description : question 2.

Authors:
    David Cabeza 1310191
    Rafael Blanco 1310156
    Fabiola Martinez 1310838
"""

from project_1 import * 
from generador import *
import matplotlib.pyplot as plt

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

def plot_circle_points(x_points,y_points):
    fig = plt.figure(1)
    plt.axis([0,20,0,20])
    ax = fig.add_subplot(1,1,1)
    circle = plt.Circle((10,10), radius=6, color='b', fill=False)
    ax.add_patch(circle)
    p = plt.scatter(x_points,y_points, c='r', marker='.')
    plt.show()


# Dataset : B1_N1000.txt -------------------------------------------------------

x1,y1 = read_dataset("datosP2_AJ2018_B1_N1000.txt")
n1 = Network([2, 2, 1])
n1.training(0.01,x1,y1)
a1,b1 = generador(2000,6)
# llamar plot circle

# Dataset : B1_N2000.txt -------------------------------------------------------

x1,y1 = read_dataset("datosP2_AJ2018_B1_N2000.txt")
n1 = Network([2, 2, 1])
n1.training(0.01,x1,y1)
a1,b1 = generador(2000,6)
# llamar plot circle

