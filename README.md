# Proyecto 2

## Redes Neurales 

### Resumen

El objetivo de este proyecto es la familiarización del estudiante con el algoritmo de backpropagation sobre redes neuronales del tipo feedforward,  así como su uso sobre diversos conjuntos de datos. Para ello se pide que se implemente dicho algoritmo en el lenguaje imperativo de su preferencia; para luego evaluar esa implementación sobre 2 conjuntos de datos: uno generado artificialmente y el clásico Iris Dataset.


### Descripción

1. Implemente el algoritmo de backpropagation sobre una red feedforward multicapa.

2. Pruebe su red con el siguiente experimento de clasificación de patrones: Se pretende lograr la clasificación de puntos en un plano pertenecientes a dos regiones predeterminadas (A  y B), como se muestran en la siguiente figura:

circulo.jpg

El área A  es el cuadrado cuya diagonal es el segmento de recta que une los puntos (0,  0) y (20,  20), menos el área que ocupa B .
Trabajaremos sobre dos áreas para B:
1. El área B1 que esta delimitada por una circunferencia centrada en (10,10) con radio 6. La ecuación canónica de esta circunferencia es:  (x - 10)2 + (y -10)2 = 36

2. El área B2  que esta delimitada por una circunferencia centrada en (10,10) con radio 8. La ecuación canónica de esta circunferencia es:  (x - 10)2 + (y -10)2 = 64

Se entiende por patrón (o ejemplo un punto (x, y) dentro del rectángulo, etiquetado con el área al que este pertenece (A  o B ).
Se le suministrarán, para  cada configuración (de cuadrados y círculos), tres conjuntos de entrenamiento de 500, 1000 y 2000 patrones ya clasificados con los que usted deberá entrenar su red. Los puntos en los datos que se le proporcionan fueron generados uniformemente sobre todo el cuadrado. Estos conjuntos de datos se encuentran en el repositorio de github del proyecto, y tienen como nombre "datosP2_AJ2018_BX_NY.txt, donde X puede ser B1 o B2, y Y es el cantidad de datos en el archivo. La visualización de de estos datos para N500, con los ejemplos etiquetados "círculo" de color azul y los "no círculo" en rojo, pueden verlas en las imágenes datos_B1_500.png y datos_B2_500.png  
Adicionalmente usted debe generar otros conjuntos de datos (500, 1000 y 2000 patrones para cada configuración) de manera que el número de patrones que corresponde cada área sea igual.
Cada red ha de entrenarse de tal forma que se aprendan la clasificación correcta de los puntos.
Usando 6 neuronas en la capa intermedia y el conjuntos de datos de 2000 patrones suministrado el área B2,  pruebe varios valores para la tasa de aprendizaje y escoja el mejor para usarlo en la prueba de variación de número de neuronas en la capa intermedia.
Pruebe con redes de 2 a 10 neuronas en la capa intermedia, para cada configuración y con los 6 conjuntos de entrenamiento explicados anteriormente . Reporte los errores de entrenamiento  de cada combinación de alternativas (Número de neuronas x conjunto de entrenamiento).
Tome como conjunto de prueba los puntos (aproximadamente 10.000 = 100 x 100) de un barrido completo de la región cuadrada correctamente etiquetados. Evalúe las configuraciones en base a: 1) error en entrenamiento, 2) error en prueba, 3) falsos positivos, 4) falsos negativos.
Para el mejor conjunto de entrenamiento encontrado. Muestre para cada red (difiere en el número de neuronas):
Muestre la gráfica de convergencia
Muestre visualmente la validación del aprendizaje, tomando puntos de un barrido completo de la región cuadrada y coloreando cada punto con un color dependiendo de la clasificación que arroje la red.
Recuerde que por ser las redes neurales algoritmos estocásticos la evaluación de las configuraciones debe hacerse sobre los promedios de al menos 10 corridas. 
OPCIONAL: Pruebe con una red de dos capas.
 

3.  Entrene su red para construir dos clasificadores sobre los datos del conjunto Iris Data Set

(http://archive.ics.uci.edu/ml/datasets/Iris ): 

3.1 Uno que separe las \Iris Setosa" del resto (Clasificador binario)

3.2. Uno que separe cada una de las 3 clases.

Pruebe con redes de 4 a 10 neuronas en la capa intermedia, usando como conjunto de entrenamiento los siguientes porcentajes de los datos: 50 %, 60 %, 70 %, 80 %, 90 %.
Muestre sus resultados. 
OPCIONAL: Pruebe con una red de dos capas.

 

### Entrega

La fecha de entrega sugerida es el día Miércoles 13 de Junio, a la hora de clases (1:30 am).

Deberán dejar en el repositorio de github de su grupo su código y un breve informe. El link para esta asignación en github es: https://classroom.github.com/g/Yxy2RdKS

Cada grupo deberá entregar una copia impresa de su informe. El informe debe ser breve y conciso, debe incluir:

Resumen.
Detalles de implementación/experimentación. (Lenguaje usado, detalles del algoritmo, etc). 
Presentación y discusión de los resultados (En base a los elementos requeridos para cada conjunto de datos)
Conclusiones
Referencias
