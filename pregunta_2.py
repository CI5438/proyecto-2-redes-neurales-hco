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

def corrida(datos, prueba, n, tasa, neuronas_intermedia):
    print("Nombre del archivo de datos: "+datos)
    x,y = read_dataset(datos)
    datos_prueba_x, datos_prueba_y = read_dataset(prueba)
    dentro=[] #dentro[0]
    fuera=[]
    aciertos=[]
    desaciertos=[]
    falso_positivo=[]
    falso_negativo=[]
    for i in range(n):
        net = Network([2,neuronas_intermedia,1])
        net.training(tasa, x, y)
        print("Estadistica de datos de prueba: ")
        a,b,c,d,e,f = net.eval_area(datos_prueba_x, datos_prueba_y)
        dentro.append(a)
        fuera.append(b)
        aciertos.append(c)
        desaciertos.append(d)
        falso_positivo.append(e)
        falso_negativo.append(f)
    print("\n\nPromedio de resultados para "+str(n)+" corridas:")
    print("Casos acertados: "+str(promedio(aciertos))+" ,Casos no acertados: "+str(promedio(desaciertos))+" Efectividad: "+str(promedio(aciertos)*100/(promedio(aciertos)+promedio(desaciertos)))+"%")
    print("Falso Positivo: "+str(promedio(falso_positivo))+" Falso negativo: "+str(promedio(falso_negativo))+"\n")

def plot_circle_points(x_points,y_points):
    fig = plt.figure(1)
    plt.axis([0,20,0,20])
    ax = fig.add_subplot(1,1,1)
    circle = plt.Circle((10,10), radius=6, color='b', fill=False)
    ax.add_patch(circle)
    p = plt.scatter(x_points,y_points, c='r', marker='.')
    plt.show()

# BEST LEARNING RATE -----------------------------------------------------------

alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
datos = "datosP2_AJ2018_B2_N2000.txt"
prueba = "prueba_B2_barrido_100_por_100.txt"

for i in range(len(alpha)):
    corrida(datos,prueba,3,0.1,6)

# BEST TRAINING SET  -----------------------------------------------------------

neurons = [2,3,4,5,6,7,8,9,10]

b1_archivos = ["datosP2_AJ2018_B1_N500.txt","datosP2_AJ2018_B1_N1000.txt","datosP2_AJ2018_B1_N2000.txt"]
b2_archivos = ["datosP2_AJ2018_B2_N500.txt","datosP2_AJ2018_B2_N1000.txt","datosP2_AJ2018_B2_N2000.txt"]
b1_generados_archivos = ["datos_entrenamiento_N500_B1.txt", "datos_entrenamiento_N1000_B1.txt", "datos_entrenamiento_N2000_B1.txt"]
b2_generados_archivos = ["datos_entrenamiento_N500_B2.txt", "datos_entrenamiento_N1000_B2.txt", "datos_entrenamiento_N2000_B2.txt"]
prueba_archivos = ["prueba_B1_barrido_100_por_100.txt","prueba_B2_barrido_100_por_100.txt"]

for i in range(len(neurons)):
    corrida(b1_generados_archivos[0],prueba_archivos[0],3,0.2,neurons[i])

for i in range(len(neurons)):
    corrida(b1_generados_archivos[1],prueba_archivos[0],3,0.2,neurons[i])

for i in range(len(neurons)):
    corrida(b1_generados_archivos[2],prueba_archivos[0],3,0.2,neurons[i])

for i in range(len(neurons)):
    corrida(b2_generados_archivos[0],prueba_archivos[1],3,0.2,neurons[i])

for i in range(len(neurons)):
    corrida(b2_generados_archivos[1],prueba_archivos[1],3,0.2,neurons[i])

for i in range(len(neurons)):
    corrida(b2_generados_archivos[2],prueba_archivos[1],3,0.2,neurons[i])
