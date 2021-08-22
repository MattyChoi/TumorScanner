import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity, he_normal

# use keras's sequential api to build convolutional model for classification
def vgg16(input_shape=(224, 224, 3)):
    model = Sequential()

    # add input layer
    model.add(InputLayer(input_shape=input_shape))

    # first conv block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    # second conv block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    # third conv block
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    # fourth conv block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    # fifth conv block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(Flatten())

    # chaning original model a little bit at the top level
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile your model with adam optimizer and binary cross entropy
    model.compile(optimizer=Adam(learning_rate=1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy"])


    return model


# did not use real time data augmentation and used saved augmented files

# data augmentation for training set
    # train_datagen = ImageDataGenerator(
#         rescale = 1./255,
#         # rotation_range=15,
#         # width_shift_range=0.1,
#         # height_shift_range=0.1,
#         # shear_range=0.1,
#         # zoom_range=0.1,
#         # horizontal_flip=True,
#         # vertical_flip=True,
#         # brightness_range=[0.5, 1.5],
#     )

# if __name__ == "__main__":
#     # process the dataset
#     undoPreprocess()
#     preprocess(num_augment_gen=5)

#     # create the model
#     model = vgg16()
#     print(model.summary())

#     # train the model
#     train(model, "vgg16", BATCH_SIZE=32, EPOCHS=50, IMG_SIZE=(224,224))
