"""
Universidad Simon Bolivar
Artificial Intelligence II - CI5438

Description : dataset generator.

Authors:
    David Cabeza 1310191
    Rafael Blanco 1310156
    Fabiola Martinez 1310838
"""

from random import uniform

def pertenece_B(x,y,radio):
	return ((x-10)**2) + ((y-10)**2) <= (radio**2)

def generador(n, radio, nombre):
	area_A=0
	area_B=0
	a=[]
	b=[]
	while (area_A<(n/2)):
		x=uniform(0.0, 20.0)
		y=uniform(0.0, 20.0)
		if (pertenece_B(x,y,radio)):
			continue
		else:
			area_A=area_A+1
			a.append([x,y])

	while (area_B<(n/2)):
		x=uniform(0.0, 20.0)
		y=uniform(0.0, 20.0)
		if (pertenece_B(x,y,radio)):
			area_B=area_B+1
			b.append([x,y])

	h=open(nombre, "w")
	for i in range (len(a)):
		h.write(str(b[i][0])+' '+str(b[i][1])+" "+str(1)+'\n')
		h.write(str(a[i][0])+' '+str(a[i][1])+" "+str(0)+'\n')
	h.close()
	
n=[500,1000,2000, 500, 1000, 2000]
radio=[6,6,6,8,8,8]
nombre=["datos_entrenamiento_N500_B1.txt", "datos_entrenamiento_N1000_B1.txt", "datos_entrenamiento_N2000_B1.txt","datos_entrenamiento_N500_B2.txt", "datos_entrenamiento_N1000_B2.txt", "datos_entrenamiento_N2000_B2.txt"]

for i in range(len(n)):
	generador(n[i], radio[i], nombre[i])