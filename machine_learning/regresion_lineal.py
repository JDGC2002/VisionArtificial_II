"""
Punto de partida
"""

import numpy as np
import matplotlib.pyplot as plt

LR = 0.00005
EPOCHS = 20



dataset = np.loadtxt('train.csv', delimiter=";")
X = dataset[:,0,None] # Preservar la dimensión y que quede como un vector
y = dataset[:,1,None]

# Estandarizar datos

X = (X-np.mean(X))/np.std(X)

# Agregar columna de 1 a X para multiplicar por theta_0
m = X.shape[0]
X_0 = np.ones((m,1))
X = np.hstack((X_0,X))

# Parámetros a optimizar
theta = np.random.rand(2)
print(theta)
y_gorrito = X@theta
error = y_gorrito - y

for i in range(EPOCHS):
    dpar1 = 1/m * np.sum(error)
    dpar2 = 1/m * np.sum(error * X[:,1,None])

    theta[0] = theta[0] - LR*dpar1
    theta[1] = theta[1] - LR*dpar2

    y_gorrito = X @ theta

    # plt.scatter(dataset[:, 0], dataset[:, 1])
    # plt.plot(dataset[:, 0], y_gorrito, color='red')
    # plt.xlabel('Area de la casa')
    # plt.ylabel('Precio')
    # plt.show()

# nueva y_gorrito
y_gorrito = X @ theta

#Gráfica de los datos
plt.scatter(dataset[:, 0], dataset[:, 1])
plt.plot(dataset[:, 0], y_gorrito, color='red')
plt.xlabel('Area de la casa')
plt.ylabel('Precio')
plt.show()
