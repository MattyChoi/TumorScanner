from keras.backend import learning_phase
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity


def cnn(input_shape=(150, 150, 3)):
    # outputs 3d feature maps (height, width, features)
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', kernel_initializer=glorot_uniform))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_uniform))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_uniform))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # coverts 3d features to 1d vectors
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # compile your model with adam optimizer and binary cross entropy
    model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])

    return model