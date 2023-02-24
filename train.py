import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from parcial_regresion_logistica import LogisticRegression
import pandas as pd

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression(lr=0.01)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)