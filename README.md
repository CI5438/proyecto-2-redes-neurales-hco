# Proyecto 2

## Redes Neurales

### Resumen

El siguiente estudio consiste en implementar el algoritmo de Backpropagation sobre una Red Neuronal Multicapa para entrenar dos conjuntos de datos. Por una parte se tienen puntos que se ubican dentro o fuera del área B1 o B2 de un circulo con radio 6 y 8 respectivamente. Para dicho conjunto de datos se realizaran pruebas y análisis de los resultados para obtener el mejor conjunto de entrenamiento de acuerdo a una tasa de aprendizaje y cantidad de neuronas en la capa intermedia que mejor se adapte al problema. Por otro lado, se tiene el conjunto de datos clásico Iris dataset introducido por Ronald Fisher en 1936. Por una parte, se prueba el conjunto utilizando una clasificación binaria que separa Iris setosa del resto. Por otra parte, se prueba el mismo conjunto separando las tres clases. Ambas pruebas son hechas con porcentajes de datos de entrenamiento de 50, 60, 70, 80 y 90 y con redes de 4 a 10 neuronas en la capa intermedia.

### Requisitos

El proyecto se encuentra implementado en el lenguaje Python en su versión 3.

Las librerías utilizadas en el proyecto son math, random, las cuales están incluidas en la instalación del lenguaje y pandas que es instalable a través del Python package manager (pip)

```bash
pip install pandas
```

### Ejecución

Ubicarse en el directorio del proyecto y ejecutar el comando correspondiente a la pregunta que quiere resolver (suponiendo que python3 es el comando para ejecutar el intérprete de Python en su versión 3)

**Solución de la pregunta dos:**

```bash
python3 clasificacion.py
```

**Solución de la pregunta tres:**

```bash
python3 iris.py
```

### Autores

* David Cabeza
* Fabiola Martínez
* Rafael Blanco