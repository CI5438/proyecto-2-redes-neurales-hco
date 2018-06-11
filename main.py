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

class Network:

    def __init__(self, q):
        self.alpha = 0.01
        
        self.layers = []
        self.weights = []

        self.init_layers(q)
        self.init_weights(q)

        print(self.weights)

    def init_layers(self, q):
        for i in range(len(q)):
            self.layers.append([x for x in range(q[i])])
        
        return True

    def init_weights(self, q):
        for i in range(len(q)-1):
            aux = []
            for j in range(q[i]):
                aux.append([uniform(-0.5, 0.5) for k in range(q[i+1])])
            
            self.weights.append(aux)

n = Network([5, 3, 2])