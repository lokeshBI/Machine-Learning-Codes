
# import packages 
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import to_categorical


# load dataset
from keras.datasets import cifar10
(trainX, trainy), (testX, testy) = cifar10.load_data()

# trainX is an ndarray
trainX
trainy # class numbers
testy
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

# check images
for i in range(9):
    plt.subplot(330+1+i)
    plt.imshow(trainX[i])
plt.show()


# one hot encode target values
trainy = to_categorical(trainy)
testy = to_categorical(testy)

trainy

# convert from integers to floats
trainX_norm = trainX.astype('float32')
testX_norm = testX.astype('float32')


# normalize to range 0-1
trainX_norm = trainX_norm / 255.0
testX_norm = testX_norm / 255.0


# example of a 3-block vgg style architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model summary
model.summary()

# fit model
history = model.fit(trainX_norm, trainy, epochs=100, batch_size=64, validation_data=(testX_norm, testy), verbose=0)

# evaluate model
_, acc = model.evaluate(testX_norm, testy, verbose=0)
print('> %.3f' % (acc * 100.0))














