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

"""
Description: verify if a point on the plane is part of a circle.

Parameters:
    @param x: x coord.
    @param y: y coord.
    @param radio: circle's radio
"""
def pertenece_B(x,y,radio):
	return ((x-10)**2) + ((y-10)**2) <= (radio**2)
"""
"""

def generador_barrido_zona(n,x1,x2,y1,y2,radio, nombre):
	division_eje_x=(x2-x1)/(n-1)
	division_eje_y=(y2-y1)/(n-1)
	h=open(nombre, "w")
	aux_x=x1
	while (aux_x<x2):
		aux_y=y1
		while(aux_y<y2):
			pertenece=0
			if (pertenece_B(aux_x, aux_y, radio)):
				pertenece=1
			h.write(str(aux_x)+' '+str(aux_y)+" "+str(pertenece)+'\n')
			aux_y = aux_y + division_eje_y
		aux_x = aux_x + division_eje_x
	h.close()

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

generador_barrido_zona(100.0,0.0,20.0,0.0,20.0,6,"prueba_B1_barrido_100_por_100.txt")
generador_barrido_zona(100.0,0.0,20.0,0.0,20.0,8,"prueba_B2_barrido_100_por_100.txt")

