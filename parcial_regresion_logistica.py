import cv2
import pandas as pd
import numpy as np
import matplotlib as plt

LR = 0.02
EPOCHS = 6
BATCH_SIZE = 1000
LAMBDA = 1

def accuracy(h, y):
    m = y.shape[0]
    h[h>=0.5] = 1
    h[h<0.5] = 0
    c = np.zeros(y.shape)
    c[y==h] = 1
    return c.sum()/m

def shuffle_dataset(df):
    df = df.

def split_data(df):
    p_train = 0.80 # Porcentaje de train.
    train = df[:int((len(df))*p_train)] 
    test = df[int((len(df))*p_train):]
    return train, test

def main():
    dataset = pd.read_csv("dataset_imagenes.csv")
    train, test = split_data(dataset)
    X = dataset[:,0,None] 
    y = dataset[:,1,None]

    # Agregar columna de 1 a X para multiplicar por theta_0
    m = X.shape[0]
    X_0 = np.ones((m,1))
    X = np.hstack((X_0,X))

    # Parámetros a optimizar
    theta = np.random.rand(2)
    y_gorrito = X@theta
    error = y_gorrito - y

    for i in range(EPOCHS):
        dpar1 = 1/m * np.sum(error)
        dpar2 = 1/m * np.sum(error * X[:,1,None])

        theta[0] = theta[0] - LR*dpar1
        theta[1] = theta[1] - LR*dpar2

        y_gorrito = X @ theta

    # nueva y_gorrito
    y_gorrito = X @ theta


if __name__ == '__main__': main()