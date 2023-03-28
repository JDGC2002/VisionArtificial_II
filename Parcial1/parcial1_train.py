import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from parcial1_regresion_logistica import LogisticRegression
import pandas as pd

# Cargar los datos
bc = pd.read_csv('dataset_imagenes.csv', sep=';')
X = bc['imagen'].tolist()
y = bc['clase'].tolist()

for i in range(len(X)):
    # Eliminar corchetes al inicio y al final de la cadena
    X[i] = X[i].strip('[]')
    # Convertir la cadena en una lista de números de coma flotante
    X[i] = [float(x) for x in X[i].split(',')]

X = np.array(X)
y = np.array(y)

# Separar los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Entrenar el modelo
clf = LogisticRegression(lr=0.01, n_iters=2000)
clf.fit(X_train,y_train)

# Realizar predicciones para train y test
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Calcular accuracies para train y test
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc_train = accuracy(y_pred_train, y_train)
acc_test = accuracy(y_pred_test, y_test)

# Imprimir accuracies
print(f"Accuracy en train: {acc_train}")
print(f"Accuracy en test: {acc_test}")