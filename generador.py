from random import uniform

def pertenece_B(x,y,radio):
	return ((x-10)**2) + ((y-10)**2) <= (radio**2)

def generador(n, radio):
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

	f=open("1.txt","w")
	g=open("2.txt","w")
	h=open("datos_prueba_N2000_B1.txt", "w")
	for i in range (len(a)):
		f.write(str(a[i][0])+' '+str(a[i][1])+'\n')
	for i in range (len(b)):
		g.write(str(b[i][0])+' '+str(b[i][1])+'\n')
	for i in range (len(a)):
		h.write(str(b[i][0])+' '+str(b[i][1])+" "+str(1)+'\n')
		h.write(str(a[i][0])+' '+str(a[i][1])+" "+str(0)+'\n')
	f.close()
	g.close()
	h.close()
	#print(a)
	#print(b)

generador(2000,6)