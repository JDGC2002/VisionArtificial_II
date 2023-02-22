import cv2
import pandas as pd
import numpy as np
import matplotlib as plt

LR = 0.02
EPOCHS = 6
BATCH_IND = 0
BATCH_SIZE = 300
LAMBDA = 1

def accuracy(h, y):
    m = y.shape[0]
    h[h>=0.5] = 1
    h[h<0.5] = 0
    c = np.zeros(y.shape)
    c[y==h] = 1
    return c.sum()/m

def shuffle_dataset(df):
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def split_data(df):
    p_train = 0.80 # Porcentaje de train.
    train = df[:int((len(df))*p_train)] 
    test = df[int((len(df))*p_train):]
    return train, test

def main():
    dataset = np.loadtxt("dataset_imagenes.csv")
    dataset = shuffle_dataset(dataset)
    train, test = split_data(dataset)
    X = train[:,0,None] 
    y = train[:,1,None]

    m = X.shape[0]
    X_0 = np.ones((m,1))
    X = np.hstack((X_0,X))

    theta = np.random.rand(2)
    y_gorrito = X@theta
    error = y_gorrito - y

    for epoch in range(EPOCHS):
        if epoch > 0:
            BATCH_IND = BATCH_IND + BATCH_SIZE
        else:
            BATCH_IND = 0
        for i in range(BATCH_SIZE):
            X_batch = X[BATCH_IND:(BATCH_IND+BATCH_SIZE)]
            y_batch = y[BATCH_IND:(BATCH_IND+BATCH_SIZE)]
            
if __name__ == '__main__': main()