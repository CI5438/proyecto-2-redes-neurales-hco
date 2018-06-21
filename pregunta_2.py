"""
Universidad Simon Bolivar
Artificial Intelligence II - CI5438

Description : question 2.

Authors:
    David Cabeza 1310191
    Rafael Blanco 1310156
    Fabiola Martinez 1310838
"""

from main import * 
from generador import *
import matplotlib.pyplot as plt

def get_point(tuple_array):

    x = []
    y = []
    for i in range(len(tuple_array)):
        x.append(tuple_array[i][0])
        y.append(tuple_array[i][1])


def plot_circle_points(x_points,y_points):
    fig = plt.figure(1)
    plt.axis([0,20,0,20])
    ax = fig.add_subplot(1,1,1)
    circle = plt.Circle((10,10), radius=6, color='b', fill=False)
    ax.add_patch(circle)
    p = plt.scatter(x_points,y_points, c='r', marker='.')
    plt.show()

def plot_normal(x,y,xlabel,ylabel,title, color):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x,y,c=color)
    plt.show

# BEST LEARNING RATE -----------------------------------------------------------

# alpha = [0.01,0.05,0.1,0.2,0.3,0.4]
# datos = "datosP2_AJ2018_B2_N2000.txt"
# prueba = "prueba_B2_barrido_100_por_100.txt"

# for i in range(len(alpha)):
#     corrida(datos,prueba,3,alpha[i],6)

# BEST TRAINING SET  -----------------------------------------------------------

neurons = [2,3,4,5,6,7,8,9,10]
alpha = 0.01
n = 10

b1_archivos = ["datosP2_AJ2018_B1_N500.txt","datosP2_AJ2018_B1_N1000.txt","datosP2_AJ2018_B1_N2000.txt"]
b2_archivos = ["datosP2_AJ2018_B2_N500.txt","datosP2_AJ2018_B2_N1000.txt","datosP2_AJ2018_B2_N2000.txt"]
b1_generados_archivos = ["datos_entrenamiento_N500_B1.txt", "datos_entrenamiento_N1000_B1.txt", "datos_entrenamiento_N2000_B1.txt"]
b2_generados_archivos = ["datos_entrenamiento_N500_B2.txt", "datos_entrenamiento_N1000_B2.txt", "datos_entrenamiento_N2000_B2.txt"]
prueba_archivos = ["prueba_B1_barrido_100_por_100.txt","prueba_B2_barrido_100_por_100.txt"]

h=open("result_training.csv", "w")
h.write("Archivo_de_datos, Archivo_de_prueba, tasa_aprendizaje, num_neurona, catidad_corridas, error_entrenamiento, error_prueba, falso_positivo, falso_negativo, casos_acertados, casos_no_acertados, efectividad\n")


for i in range(len(neurons)):
    a,b,c,d,e,f,g = corrida(b1_generados_archivos[0],prueba_archivos[0],n,alpha,neurons[i])
    h.write(b1_generados_archivos[0]+", "+prueba_archivos[0]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

for i in range(len(neurons)):
    a,b,c,d,e,f,g = corrida(b1_generados_archivos[1],prueba_archivos[0],n,alpha,neurons[i])
    h.write(b1_generados_archivos[1]+", "+prueba_archivos[0]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

for i in range(len(neurons)):
    a,b,c,d,e,f,g = corrida(b1_generados_archivos[2],prueba_archivos[0],n,alpha,neurons[i])
    h.write(b1_generados_archivos[2]+", "+prueba_archivos[0]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

for i in range(len(neurons)):
    a,b,c,d,e,f,g = corrida(b2_generados_archivos[0],prueba_archivos[1],n,alpha,neurons[i])
    h.write(b2_generados_archivos[0]+", "+prueba_archivos[0]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

for i in range(len(neurons)):
    a,b,c,d,e,f,g = corrida(b2_generados_archivos[1],prueba_archivos[1],n,alpha,neurons[i])
    h.write(b2_generados_archivos[1]+", "+prueba_archivos[0]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

for i in range(len(neurons)):
    a,b,c,d,e,f,g = corrida(b2_generados_archivos[2],prueba_archivos[1],n,alpha,neurons[i])
    h.write(b2_generados_archivos[2]+", "+prueba_archivos[0]+" ,"+str(alpha)+" ,"+str(neurons[i])+" ,"+str(n)+","+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+" ,"+str(e)+" ,"+str(f)+" ,"+str(g)+"\n")

h.close()
