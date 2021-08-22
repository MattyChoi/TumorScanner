import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam, RMSprop
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity

# use keras's sequential api to build convolutional model for classification
def trained_vgg16(input_shape=(224, 224, 3)):

    # get pretrained model
    vgg_model= VGG16(weights='imagenet', input_shape=input_shape, include_top=False)

    # make the model
    model= Sequential()

    # add the layers from the pretrained model
    for layer in vgg_model.layers:
        model.add(layer)

    # make these untrainable
    for layer in model.layers:
        layer.trainable=False

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile your model with adam optimizer and binary cross entropy
    model.compile(optimizer=Adam(learning_rate=5e-5),
              loss="binary_crossentropy",
              metrics=["accuracy"])

    return model


# used real time data augmentation since only a small layer needed to be trained

# data augmentation for training set
# train_datagen = ImageDataGenerator(
#     rescale = 1./255,
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     horizontal_flip=True,
#     zoom_range=0.1,
#     brightness_range=[0.5, 1.5],
# )


# if __name__ == "__main__":
#     # process the dataset
#     undoPreprocess()
#     preprocess(num_augment_gen=30)

#     # create the model
#     model = cnn()
#     print(model.summary())

#     # train the model
#     train(model, "cnn", BATCH_SIZE=32, EPOCHS=50, IMG_SIZE=(224,224))
