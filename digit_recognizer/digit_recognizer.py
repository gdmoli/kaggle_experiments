import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

number_of_classes = 10

dir_path = os.path.dirname(os.path.realpath(__file__))
training_data_filename = os.path.join(dir_path, 'data', 'train.csv')
test_data_filename = os.path.join(dir_path, 'data', 'test.csv')

train_data = pd.read_csv(training_data_filename)
test_data = pd.read_csv(test_data_filename)

X_train = (train_data.iloc[:,1:].values).astype('float32')
y_train = (train_data.iloc[:,0].values.astype('int32'))
X_test = test_data.values.astype('float32')

X_train = X_train.reshape(X_train.shape[0], 28, 28)

#for i in range(6,9):
#    plt.subplot( 310 + (i-6+1) )
#    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
#    plt.title(y_train[i])

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes=number_of_classes)

seed=43
np.random.seed(seed)

# Simple linear model
model=keras.models.Sequential()
model.add(keras.layers.Conv2D(activation='relu', padding='Same', filters=32, kernel_size=(5,5), input_shape=(28,28,1)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(activation='relu', padding='Same', filters=32, kernel_size=(5,5)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(265, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(number_of_classes, activation='softmax'))
model.compile( optimizer=keras.optimizers.Adam(),
                loss="categorical_crossentropy",
                metrics=["accuracy"])

X=X_train
Y=y_train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.10, random_state=seed)
model.fit( X_train, y_train, validation_data=(X_test,y_test), epochs=50, batch_size=32 )

plt.show()
print('hi')
#training_data = train_data.drop(columns=["label"])
#training_labels = train_data.label
