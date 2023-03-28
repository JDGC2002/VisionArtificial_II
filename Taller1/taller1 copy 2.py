import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

LR = 0.001
EPOCHS = 1000
BATCH_SIZE = 128

dt_train = pd.read_csv('train.csv')

le = LabelEncoder()

#Extracción de título de nombre

dt_train['Title'] = dt_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
dt_train['Title'] = dt_train['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dt_train['Title'] = dt_train['Title'].replace('Mlle', 'Miss')
dt_train['Title'] = dt_train['Title'].replace('Ms', 'Miss')
dt_train['Title'] = dt_train['Title'].replace('Mme', 'Mrs')
    
#Modificación de cabin en OneHot

dt_train['Title'] = le.fit_transform(dt_train['Title'])
dt_train['Sex'] = le.fit_transform(dt_train['Sex'])

#Creo subset de trabajo con las variables que voy a usar "data"
data = dt_train[['SibSp', 'Parch','Pclass', 'Age', 'Fare', 'Title']].dropna()
features = ['SibSp', 'Parch','Age', 'Fare', 'Title']
# Defino las capas que voy a usar

x = data[features].to_numpy()
x = tf.constant(x)
y = data['Pclass']
y = tf.one_hot(y, depth=3)

inputs = layers.Input(shape=(5,))
normalizer = layers.experimental.preprocessing.Normalization(axis=-1)
dense0 = layers.Dense(units=6, activation='relu', name='hidden')
dense1 = layers.Dense(units=3, activation='softmax', name='output')

# Defino relación entre capas
z = dense0(normalizer(inputs))
out = dense1(z)

# Defino Modelo
model = Model(inputs, out)

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.summary()
hist = model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split = 0.1)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()