import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

LR = 0.001
EPOCHS = 500
BATCH_SIZE = 10

dt_train = pd.read_csv('train.csv')
dt_train['Title'] = dt_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
dt_train['Title'] = dt_train['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dt_train['Title'] = dt_train['Title'].replace('Mlle', 'Miss')
dt_train['Title'] = dt_train['Title'].replace('Ms', 'Miss')
dt_train['Title'] = dt_train['Title'].replace('Mme', 'Mrs')

title_pclass = dt_train.groupby(['Parch', 'Pclass']).size().unstack(fill_value=0)

print(title_pclass)