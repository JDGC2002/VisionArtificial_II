import cv2
import pandas as pd
import numpy as np
import matplotlib as plt

LR = 0.02
EPOCHS = 6
BATCH_IND = 0
BATCH_SIZE = 300
LAMBDA = 1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta, lmbda):
    m = X.shape[0]
    h = sigmoid(X @ theta)
    J = (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) + (lmbda / (2 * m)) * np.sum(theta[1:] ** 2)
    return J

def gradient(X, y, theta, lmbda):
    m = X.shape[0]
    h = sigmoid(X @ theta)
    grad = (1 / m) * (X.T @ (h - y)) + (lmbda / m) * np.vstack([0, theta[1:]])
    return grad

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
    dataset = pd.read_csv("dataset_imagenes.csv", sep=";", header=None)
    dataset = shuffle_dataset(dataset)
    train, test = split_data(dataset)
    
    X = train.iloc[:, 0]
    y = train.iloc[:, 1]

    m = X.shape[0]
    X_0 = np.ones(m)
    X = np.hstack((X_0,X))

    theta = np.random.rand(2)
    y_gorrito = X @ theta
    error = y_gorrito - y

    for epoch in range(EPOCHS):
        if epoch > 0:
            BATCH_IND = BATCH_IND + BATCH_SIZE
        else:
            BATCH_IND = 0
        for i in range(BATCH_SIZE):
            X_batch = X[BATCH_IND:(BATCH_IND + BATCH_SIZE)]
            y_batch = y[BATCH_IND:(BATCH_IND + BATCH_SIZE)]
            grad = gradient(X_batch, y_batch, theta, LAMBDA)
            theta = theta - LR * grad
        cost = cost_function(X, y, theta, LAMBDA)
        train_accuracy = accuracy(sigmoid(X @ theta), y)
        print("Epoch:", epoch+1, "Cost:", cost, "Training accuracy:", train_accuracy)
            
if __name__ == '__main__':
    main()