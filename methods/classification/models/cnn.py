import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity, he_normal


def cnn(input_shape=(224, 224, 3)):
    model = Sequential()

    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=he_normal))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # compile your model with adam optimizer and binary cross entropy
    model.compile(optimizer=Adam(learning_rate=1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy"])

    return model

# did not use real time data augmentation and used saved augmented files

# train_datagen = ImageDataGenerator(
#     rescale = 1./255,
#     # rotation_range=15,
#     zoom_range=0.1,
#     shear_range=0.1,
#     horizontal_flip=True,
#     # vertical_flip=True,
#     # width_shift_range=0.1,
#     # height_shift_range=0.1,
#     # brightness_range=[0.5, 1.5],
# )


# if __name__ == "__main__":
#     # process the dataset
#     undoPreprocess()
#     preprocess(num_augment_gen=0)

#     # create the model
#     model = cnn()
#     print(model.summary())

#     # train the model
#     train(model, "cnn", BATCH_SIZE=32, EPOCHS=75, IMG_SIZE=(224,224))