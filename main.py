"""
Universidad Simon Bolivar
Artificial Intelligence II - CI5438

Neural Networks

Authors:
    David Cabeza 1310191
    Rafael Blanco 1310156
    Fabiola Martinez 1310838
"""
from random import uniform
from math import exp, sqrt

def sub_vec(a,b):
    c=[]
    for i in range (0,len(a)):
        c.append(a[i]-b[i])
    return c

def norm2(x):
    plus=0
    for i in range (0, len(x)):
        plus += x[i]**2
    return sqrt(plus)

def read_dataset(filename):
    dataset = open(filename, "r")
    t = []
    goal=[]

    lines=dataset.readlines()
    
    for line in lines:
        t.append([float(line.split(' ')[0]),float(line.split(' ')[1])])
        goal.append([float((line.split(' ')[2]).rstrip())]) # removing /n
    dataset.close()

    return t,goal

class Network:

    def __init__(self, q):
        self.alpha = 0.01
        
        self.layers = []
        self.weights = []
        self.x0_weights = []

        self.init_layers(q)
        self.init_weights(q)

        #print("capas:")
        #print(self.layers)
        #print("pesos:")
        #print(self.weights)
        #print("x0:")
        #print(self.x0_weights)
        #self.training(0.1)

    def init_layers(self, q):
        for i in range(len(q)):
            self.layers.append([x for x in range(q[i])])
        
        return True

    def init_weights(self, q):
        for i in range(len(q)-1):
            aux = []
            for j in range(q[i]):
                aux.append([uniform(-0.5, 0.5) for k in range(q[i+1])])
                #aux.append([0.5 for k in range(q[i+1])])
            
            self.weights.append(aux)

        for i in range(1, len(q)):
            self.x0_weights.append([uniform(-0.5, 0.5) for k in range(q[i])])
            #self.x0_weights.append([0.5 for k in range(q[i])])

        return True

    def training(self, n, t,goal):
        it = 1
        max_it = 100000
        epsilon = 10**-5
        w_new=[1]
        w_old=[2]
        while ((norm2(sub_vec(w_new,w_old))>epsilon) and (it<max_it)): #para todos los conjuntos de ejemplos
            #para cada ejemplo
            #supongamos que el dato esta almacenado en t
            #t = [[0,0],[0,1],[1,0],[1,1]]
            #goal = [[0,1], [1,0], [1,0], [0,1]]
            #goal = [[1],[2],[3]]
            for i in range(len(t)):  #para cada ejemplo
                o=[t[i]]
                s=[]
                #Calculo de O
                for j in range(1,len(self.layers)): #para cada capa, excepto la inicial 
                    aux=[]
                    for k in range(len(self.layers[j])): #para cada elemento de esa capa
                        net=self.x0_weights[j-1][k]
                        for l in range(len(o[j-1])):
                            net = net + o[j-1][l]*self.weights[j-1][l][k]
                        aux.append(1/(1+exp(-net)))
                    o.append(aux)

                #Calculo de S
                for j in range(len(self.layers)-1,-1,-1): #para cada capa
                    aux=[]

                    if (j==len(self.layers)-1): #Si es la ultima capa, el proceso es distinto
                        for k in range(len(self.layers[j])):
                            sk=o[j][k]*(1-o[j][k])*(goal[i][k]-o[j][k])
                            aux.append(sk)
                    else:
                        for k in range(len(self.layers[j])): #para cada neurona de la capa
                            aux_2=0
                            for l in range(len(self.weights[j][k])): #cada peso que sale de esa neurona
                                aux_2 = aux_2 + self.weights[j][k][l]*s[len(s)-1][l]
                            sk=o[j][k]*(1-o[j][k])*aux_2
                            aux.append(sk)


                    s.append(aux)

                s.reverse()
                print("O:")
                print(o)
                print("S:")
                print(s)


                w_old=[]
                w_new=[]
                for j in range(len(self.weights)): #j -> capa
                    for k in range(len(self.weights[j])): #k -> neurona inicial
                        for l in range(len(self.weights[j][k])): #l -> neurona final
                            w_old.append(self.weights[j][k][l]) 
                            self.weights[j][k][l]=self.weights[j][k][l] + n*s[j+1][l]*self.weights[j][k][l]*o[j][k]
                            w_new.append(self.weights[j][k][l]) 

                #weights x0
                for j in range(len(self.x0_weights)): #j -> capa desde 1
                    for k in range(len(self.x0_weights[j])): #k -> hacia neurona de la capa j
                        w_old.append(self.x0_weights[j][k])
                        self.x0_weights[j][k] = self.x0_weights[j][k] + n*s[j+1][k]*self.x0_weights[j][k]
                        w_new.append(self.x0_weights[j][k])

            it=it+1
            print("weights:")
            print(self.weights)
            print("x0_weights:")
            print(self.x0_weights)
            print("IT: ", it)
            print((norm2(sub_vec(w_new,w_old))))    


n = Network([2, 2, 1])
x,y=read_dataset("datosP2_AJ2018_B1_N1000.txt")
#print(x)
#print("y:")
#print(y)

n.training(0.01,x,y)


#Lectura de datos
#Verificar el entrenamiento con esos datos
#todas y cada una de las pruebas que manda caromar
    #Hacer el trabajo de los datos que ella pide
#Trabajar con la base de datos que ella pide (Iris data set)
