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

def function_s(x):
    return (1/(1+exp(-x)))

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
                #aux.append([uniform(-1.0, 1.0) for k in range(q[i+1])])
                aux.append([uniform(-0.1, 0.1) for k in range(q[i+1])])
                #aux.append([0.5 for k in range(q[i+1])])
            
            self.weights.append(aux)

        for i in range(1, len(q)):
            #self.x0_weights.append([uniform(-1.0, 1.0) for k in range(q[i])])
            self.x0_weights.append([uniform(-0.1, 0.1) for k in range(q[i])])
            #self.x0_weights.append([0.5 for k in range(q[i])])

        return True

    def training(self, n, t,goal):
        it = 1
        max_it = 5000
        epsilon = 10**-5
        w_new=[1]
        w_old=[2]
        #while ((norm2(sub_vec(w_new,w_old))>epsilon) and (it<max_it)): #para todos los conjuntos de ejemplos
        while ((it<max_it)):
            #para cada ejemplo
            #supongamos que el dato esta almacenado en t
            for i in range(len(t)):  #para cada ejemplo
                o=self.get_o(t[i])
                s=self.get_s(o, goal[i])
                self.actualizar_pesos(n,s,o)

            it=it+1
        #print("weights:")
        #print(self.weights)
        #print("x0_weights:")
        #print(self.x0_weights)

    def eval(self,example, y):
        o = self.get_o(example)
        #print(o)
        print("Resultado obtenido: ")
        for i in range(len(o[len(o)-1])):
            print(str(round(o[len(o)-1][i],0))+" ",end="")
        print("\nResultado correcto: ")
        print(y)

    def get_o(self,example):
        o=[example]
        for i in range(len(self.weights)):#i capas de la red
            aux=[]
            for j in range (len(self.layers[i+1])):#j neuronas de la capa i+1
                acumulador = self.x0_weights[i][j] #peso de x0
                #print("Acumulacion inicial: "+str(acumulador))
                for k in range(len(o[i])):
                    #print("Estoy multiplicando:"+str(o[i][k])+" y "+str(self.weights[i][k][j]))
                    acumulador = acumulador + o[i][k]*self.weights[i][k][j]
                #print("Valor final: "+str(acumulador))
                aux.append(function_s(acumulador))
            o.append(aux)

        return o

    def get_s(self,o,goal):
        #print("Valores de o:")
        #print(o)
        #print("Valores de w: ")
        #print(self.weights)
        s=[]

        aux=[]
        for i in range(len(self.layers[len(self.layers)-1])): #Ultima capa
            aux.append(o[len(self.layers)-1][i]*(1-o[len(self.layers)-1][i])*(goal[i]-o[len(self.layers)-1][i]))
        #print("Valores de la ultima capa: ")
        #print(aux)
        s.append(aux)

        for i in range(len(self.layers)-2,-1,-1):#para cada capa en reversa
            aux=[]
            for j in range(len(self.layers[i])):#paara cada neurona de la capa
                acumulador = 0
                for k in range(len(self.layers[i+1])): #por cada neurona de la siguiente capa
                    acumulador = acumulador + self.weights[i][j][k]*s[len(s)-1][k]
                    #print("Acumulador: "+str(acumulador)+"Se le agregó: "+str(self.weights[i][j][k]*s[len(s)-1][k]))
                #print("Valor de o: "+str(o[i][j]))
                aux.append(o[i][j]*(1-o[i][j])*acumulador)
                #print("Valor de S: "+str(o[i][j]*(1-o[i][j])*acumulador))
            s.append(aux)
        #print("S sin reverso: ")
        #print(s)
        s.reverse()
        #print("S con reverso: ")
        #print(s)
        return s

    def actualizar_pesos(self, n,s,o):
        #print("Valores de o:")
        #print(o)
        #print("Valores de s:")
        #print(s)
        #print("Valores de w: ")
        #print(self.weights)

        for i in range(len(self.weights)): #para cada capa
            for j in range(len(self.weights[i])): #para cada neurona inicial
                for k in range(len(self.weights[i][j])): #para cada neurona final
                    #self.weights[i][j][k] = self.weights[i][j][k] + n*s[i+1][k]*self.weights[i][j][k]*o[i][j]
                    self.weights[i][j][k] = self.weights[i][j][k] + n*s[i+1][k]*o[i][j]
                    #print("Capa: "+str(i)+"Neurona inicial: "+str(j)+"Neurona final: "+str(k))
                    #print("Valor agregado a peso: "+str(n*s[i+1][k]*self.weights[i][j][k]*o[i][j]))

        for i in range(len(self.x0_weights)): #Para cada capa
            for j in range(len(self.x0_weights[i])):#para cada peso hacia esa capa
                #self.x0_weights[i][j] = self.x0_weights[i][j] + n*s[i+1][j]*self.x0_weights[i][j]
                self.x0_weights[i][j] = self.x0_weights[i][j] + n*s[i+1][j]
                #print("Capa: "+str(i)+"Neurona final: "+str(j))
                #print("Valor agregado a peso: "+str(n*s[i+1][j]*self.x0_weights[i][j]))





n = Network([8, 3, 8])
#x,y=read_dataset("datosP2_AJ2018_B1_N1000.txt")
#print(x)
#print("y:")
#print(y)
#t = [[0,0],[0,1],[1,0],[1,1]]
#goal = [[0,1], [1,0], [1,0], [0,1]]
#goal = [[1],[2],[3]]

x=[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]]
y=[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]]
#print("\nImpresion de evaluaciones sin entrenar")
#for i in range(5):
#    print (y_1[i][0])
    #n.eval(x[i],y[i])
n.training(0.3,x,y)
#x_1,y_1=read_dataset("datos_prueba_N2000_B1.txt")

print("Impresion de resultados \n\n\n")
for i in range(len(x)):
#    print (y_1[i][0])
    n.eval(x[i],y[i])


#Lectura de datos
#Verificar el entrenamiento con esos datos
#todas y cada una de las pruebas que manda caromar
    #Hacer el trabajo de los datos que ella pide
#Trabajar con la base de datos que ella pide (Iris data set)
