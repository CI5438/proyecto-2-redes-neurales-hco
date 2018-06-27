"""
Universidad Simon Bolivar
Artificial Intelligence II - CI5438

Description: Neural Networks algorithm

Authors:
    David Cabeza 1310191
    Rafael Blanco 1310156
    Fabiola Martinez 1310838
"""
from random import uniform
from math import exp, sqrt

"""
Description: subtracts two vectors.

Parameters:
    @param a: a vector.
    @param b: a vector.
"""
def sub_vec(a,b):
    c=[]
    for i in range (0,len(a)):
        c.append(a[i]-b[i])
    return c
"""
Description: calculates the norm of a vector.

Parameters:
    @param x: values of dataset variable.
"""
def norm2(x):
    plus=0
    for i in range (0, len(x)):
        plus += x[i]**2
    return sqrt(plus)
"""
Description: sigmoid unit

Parameters:
    @param x: value of variable
"""
def function_s(x):
    return (1/(1+exp(-x)))

"""
Description: class that represents a Multilayer Network
"""
class Network:
    """
    Description: Initializer
    Parameters:
        @param q: array that contains the number of neurons of each layer
    """
    def __init__(self, q):
        self.layers = []
        self.weights = []
        self.x0_weights = []
        self.err = []

        self.init_layers(q)
        self.init_weights(q)
    """
    Description: Make the arrray-structure of the Network
    Parameters:
        @param q: array that contains the number of neurons of each layer
    """
    def init_layers(self, q):
        for i in range(len(q)):
            self.layers.append([x for x in range(q[i])])

        return True
    """
    Description: initialize the random weights of the Network

    Parameters:
        @param q: array that contains the number of neurons of each layer
    """
    def init_weights(self, q):
        for i in range(len(q)-1):
            aux = []
            for j in range(q[i]):
                aux.append([uniform(-0.5, 0.5) for k in range(q[i+1])])
            self.weights.append(aux)

        for i in range(1, len(q)):
            self.x0_weights.append([uniform(-0.5, 0.5) for k in range(q[i])])

        return True
    """
    Description: Network's training

    Parameters:
        @param n: learning rate
        @param t: array of x instances
        @param goal: array of goal of each instance
    """
    def training(self, n, t, goal):
        self.err = []  #contain training error of each iteration
        it = 1         #iteration counter
        max_it = 1000
        epsilon = 10**-5
        w_new=[1]
        w_old=[2]
        print("Inicio de Entrenamiento")
        while ((norm2(sub_vec(w_new,w_old))>epsilon) and (it<max_it)):
            acum=0.0
            for i in range(len(t)):                         #for each instance
                o=self.get_o(t[i])                          #calculate outputs of each neuron
                s=self.get_s(o, goal[i])                    #calculate error of each neuron
                w_old,w_new = self.update_weights(n,s,o)  #update weights
                for j in range(len(o[len(o)-1])):           #accumulate the error fo each instance
                    acum = acum + ((o[len(o)-1][j]-goal[i][j])**2)
            self.err.append(acum/2)
            it=it+1
        print("Entrenamiento terminado.\n numero de neuronas capa intermedia: "+str(len(self.layers[1]))+"\n numero de iteraciones: "+str(it)+"\n tasa de aprendizaje: "+str(n))
    """
    Description: returns training error's array
    """
    def get_err(self):
        return self.err
    """
    Description: get ouputs

    Parameters:
        @ param instance: instance
    """
    def get_o(self,instance):
        o=[instance]
        for i in range(len(self.weights)):#i capas de la red
            aux=[]
            for j in range (len(self.layers[i+1])):#j neuronas de la capa i+1
                acumulador = self.x0_weights[i][j] #peso de x0
                for k in range(len(o[i])):
                    acumulador = acumulador + o[i][k]*self.weights[i][k][j]
                aux.append(function_s(acumulador))
            o.append(aux)
        return o
    """
    Description: get error

    Parameters:
        @ param o: array of outputs
        @ param goal: array of goal of the last layer
    """
    def get_s(self,o,goal):
        s=[]
        aux=[]
        for i in range(len(self.layers[len(self.layers)-1])): #Last layer
            aux.append(o[len(self.layers)-1][i]*(1-o[len(self.layers)-1][i])*(goal[i]-o[len(self.layers)-1][i]))
        s.append(aux)

        for i in range(len(self.layers)-2,-1,-1):#for each layer in reverse
            aux=[]
            for j in range(len(self.layers[i])):#for each neuron of the layer
                accum = 0
                for k in range(len(self.layers[i+1])): #for each neuron of the following layer
                    accum = accum + self.weights[i][j][k]*s[len(s)-1][k]
                aux.append(o[i][j]*(1-o[i][j])*accum)
            s.append(aux)
        s.reverse()
        return s

    def update_weights(self, n,s,o):
        w_old=[]
        w_new=[]
        for i in range(len(self.weights)): #for each layer
            for j in range(len(self.weights[i])): #for each in neuron
                for k in range(len(self.weights[i][j])): #for each out neuron
                    w_old.append(self.weights[i][j][k])
                    self.weights[i][j][k] = self.weights[i][j][k] + n*s[i+1][k]*o[i][j]
                    w_new.append(self.weights[i][j][k])
        for i in range(len(self.x0_weights)): #for each layer
            for j in range(len(self.x0_weights[i])):#for each weight to that layer
                w_old.append(self.x0_weights[i][j])
                self.x0_weights[i][j] = self.x0_weights[i][j] + n*s[i+1][j]
                w_new.append(self.x0_weights[i][j])
        return w_old, w_new

    """
    Description: instance evaluation of problem 1

    Parameters:
        @param t:x instance
        @param goal: goal of instance
    """
    def eval_area(self, t, goal):
        dentro=[]
        fuera=[]
        aciertos=0
        desaciertos=0
        falso_positivo=0
        falso_negativo=0
        err_acum = 0.0
        for i in range(len(t)):
            o = self.get_o(t[i])
            correcto=True
            for j in range(len(o[len(o)-1])):
                if(round(o[len(o)-1][j],0)!=goal[i][j]):
                    correcto=False
            if(round(o[len(o)-1][0],0)==1.0):
                dentro.append(t[i])
            else:
                fuera.append(t[i])
            if (correcto):
                aciertos=aciertos+1
            else:
                desaciertos=desaciertos+1
                if (round(o[len(o)-1][0],0)==1.0):
                    falso_positivo=falso_positivo+1
                if (round(o[len(o)-1][0],0)==0.0):
                    falso_negativo=falso_negativo+1
            err_acum=err_acum+((o[len(o)-1][0]-goal[i][0])**2)
        err_acum=err_acum/2
        efec=aciertos*100/(aciertos+desaciertos)
        print("Casos acertados: "+str(aciertos)+" ,Casos no acertados: "+str(desaciertos)+" ,Efectividad: "+str(efec)+"%")
        print("Error acumulado: "+str(err_acum)+" ,Falso Positivo: "+str(falso_positivo)+" ,Falso negativo: "+str(falso_negativo)+"\n")

        return dentro, fuera, aciertos, desaciertos, falso_positivo, falso_negativo, err_acum
