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

def promedio(x):
    aux=0.0
    for i in range(len(x)):
        aux=aux+x[i]
    return (aux/len(x))

class Network:

    def __init__(self, q):    
        self.layers = []
        self.weights = []
        self.x0_weights = []
        self.err = []

        self.init_layers(q)
        self.init_weights(q)

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

        for i in range(1, len(q)):
            self.x0_weights.append([uniform(-0.5, 0.5) for k in range(q[i])])

        return True

    def training(self, n, t, goal):
        #t,goal = read_dataset(nombre)
        self.err = []
        it = 1
        aciertos=0
        desaciertos=0
        max_it = 3000
        epsilon = 10**-5
        w_new=[1]
        w_old=[2]
        print("Inicio de Entrenamiento")
        while ((norm2(sub_vec(w_new,w_old))>epsilon) and (it<max_it)): #para todos los conjuntos de ejemplos
            acum=0.0
            for i in range(len(t)):  #para cada ejemplo
                o=self.get_o(t[i])
                s=self.get_s(o, goal[i])
                w_old,w_new = self.actualizar_pesos(n,s,o)
                if(round(o[len(o)-1][0],0)==goal[i][0]):
                    aciertos=aciertos+1
                else:
                    desaciertos=desaciertos+1
                for j in range(len(o[len(o)-1])):
                    acum = acum + ((o[len(o)-1][j]-goal[i][j])**2)
            self.err.append(acum/2)
            it=it+1
        print("Entrenamiento terminado.\n numero de neuronas capa intermedia: "+str(len(self.layers[1]))+"\n numero de iteraciones: "+str(it)+"\n tasa de aprendizaje: "+str(n))
        #efec=aciertos*100/(aciertos+desaciertos)
        #print("Estadisticas durante entrenamiento:")
        #print("Casos acertados: "+str(aciertos)+" ,Casos no acertados: "+str(desaciertos)+" Efectividad: "+str(efec)+"%")
        #print("Estadisticas de los datos de entrenamiento: ")
        #self.eval_area(t,goal)

    def get_err(self    ):
        return self.err

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

    def get_o(self,example):
        o=[example]
        for i in range(len(self.weights)):#i capas de la red
            aux=[]
            for j in range (len(self.layers[i+1])):#j neuronas de la capa i+1
                acumulador = self.x0_weights[i][j] #peso de x0
                for k in range(len(o[i])):
                    acumulador = acumulador + o[i][k]*self.weights[i][k][j]
                aux.append(function_s(acumulador))
            o.append(aux)
        return o

    def get_s(self,o,goal):
        s=[]
        aux=[]
        for i in range(len(self.layers[len(self.layers)-1])): #Ultima capa
            aux.append(o[len(self.layers)-1][i]*(1-o[len(self.layers)-1][i])*(goal[i]-o[len(self.layers)-1][i]))
        s.append(aux)

        for i in range(len(self.layers)-2,-1,-1):#para cada capa en reversa
            aux=[]
            for j in range(len(self.layers[i])):#paara cada neurona de la capa
                acumulador = 0
                for k in range(len(self.layers[i+1])): #por cada neurona de la siguiente capa
                    acumulador = acumulador + self.weights[i][j][k]*s[len(s)-1][k]
                aux.append(o[i][j]*(1-o[i][j])*acumulador)
            s.append(aux)
        s.reverse()
        return s

    def actualizar_pesos(self, n,s,o):
        w_old=[]
        w_new=[]
        for i in range(len(self.weights)): #para cada capa
            for j in range(len(self.weights[i])): #para cada neurona inicial
                for k in range(len(self.weights[i][j])): #para cada neurona final
                    w_old.append(self.weights[i][j][k])
                    self.weights[i][j][k] = self.weights[i][j][k] + n*s[i+1][k]*o[i][j]
                    w_new.append(self.weights[i][j][k])
        for i in range(len(self.x0_weights)): #Para cada capa
            for j in range(len(self.x0_weights[i])):#para cada peso hacia esa capa
                w_old.append(self.x0_weights[i][j])
                self.x0_weights[i][j] = self.x0_weights[i][j] + n*s[i+1][j]
                w_new.append(self.x0_weights[i][j])
        return w_old, w_new

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
    err_acum=[]
    err_entrenamiento=[]
    for i in range(n):
        net = Network([2,neuronas_intermedia,1])
        net.training(tasa, x, y)
        print("Estadistica de datos de prueba: ")
        a,b,c,d,e,f,g = net.eval_area(datos_prueba_x, datos_prueba_y)
        dentro.append(a)
        fuera.append(b)
        aciertos.append(c)
        desaciertos.append(d)
        falso_positivo.append(e)
        falso_negativo.append(f)
        err_acum.append(g)
        err_entrenamiento.append(net.get_err()[len(net.get_err())-1])
    print("\n\nPromedio de resultados para "+str(n)+" corridas:")
    print("Casos acertados: "+str(promedio(aciertos))+" ,Casos no acertados: "+str(promedio(desaciertos))+" ,Efectividad: "+str(promedio(aciertos)*100/(promedio(aciertos)+promedio(desaciertos)))+"%")
    print("Error de entrenamiento: "+str(promedio(err_entrenamiento))+" ,Error de prueba: "+str(promedio(err_acum))+ " ,Falso Positivo: "+str(promedio(falso_positivo))+" ,Falso negativo: "+str(promedio(falso_negativo))+"\n")
    return promedio(err_entrenamiento), promedio(err_acum), promedio(falso_positivo), promedio(falso_negativo)

# b1_archivos = ["datosP2_AJ2018_B1_N500.txt","datosP2_AJ2018_B1_N1000.txt","datosP2_AJ2018_B1_N2000.txt"]
# b2_archivos = ["datosP2_AJ2018_B2_N500.txt","datosP2_AJ2018_B2_N1000.txt","datosP2_AJ2018_B2_N2000.txt"]
# b1_generados_archivos = ["datos_entrenamiento_N500_B1.txt", "datos_entrenamiento_N1000_B1.txt", "datos_entrenamiento_N2000_B1.txt"]
# b2_generados_archivos = ["datos_entrenamiento_N500_B2.txt", "datos_entrenamiento_N1000_B2.txt", "datos_entrenamiento_N2000_B2.txt"]
# prueba_archivos = ["prueba_B1_barrido_100_por_100.txt","prueba_B2_barrido_100_por_100.txt"]
# corrida(b1_archivos[0],prueba_archivos[0], 3, 0.3, 6)

if __name__ == '__main__':
    alpha = [0.1,0.2,0.3]
    data = "datosP2_AJ2018_B2_N2000.txt"
    proving = "prueba_B2_barrido_100_por_100.txt"


    h=open("results.csv", "w")
    h.write("Archivo_de_datos, Archivo_de_prueba, tasa_aprendizaje, catidad_corridas, error_entrenamiento, error_prueba, falso_positivo, falso_negativo\n")
    for i in range(len(alpha)):
        print("alpha: ",alpha[i])
        print("---------------------------------------------------------------------")
        n=1
        a,b,c,d=corrida(data,proving,n,alpha[i],6)
        h.write(data+", "+proving+" ,"+str(alpha[i])+" ,"+str(n)+" ,"+str(a)+" ,"+str(b)+" ,"+str(c)+" ,"+str(d)+"\n")
    h.close()
