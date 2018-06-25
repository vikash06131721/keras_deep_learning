"""
Simple Keras model to get started for recognizing MNIST
"""

from __future__  import print_function

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(1671)


#Network And Training
NB_EPOCH= 200
BATCH_SIZE= 128
VERBOSE = 1
NB_CLASSES = 10 #Number of outputs
OPTIMIZER = SGD()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2

#data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print ("Training Set:", X_train.shape, y_train.shape)
print ("Test set:", X_test.shape, y_test.shape)
#X_train has 60k images with 28*28 pixels --> reshaped to 28*28= 784
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Normalize

X_train /= 255
X_test /= 255

print (X_train.shape[0], 'train samples')
print (X_test.shape[0], 'test samples')

#convert class vectors to binary class metrics

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test= np_utils.to_categorical(y_test, NB_CLASSES)

#10 outputs
#final stage is softmax

model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER, metrics=['accuracy'])

history = model.fit(X_train,Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose= VERBOSE, validation_split=VALIDATION_SPLIT)

score= model.evaluate(X_test, Y_test, verbose=VERBOSE)
print ("Test Score:", score[0])
print ("Test accuracy:", score[1])