import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn import *
from preprocess import *

def train(model, BATCH_SIZE=32, EPOCHS=50, IMG_SIZE=(150,150)):
    current = os.getcwd()
    dataset = os.path.join(current, "classify_data")

    # set variables
    EPOCHS = EPOCHS
    BATCH_SIZE = BATCH_SIZE
    IMG_SIZE = IMG_SIZE
    TRAIN_DIR = os.path.join(dataset, "TRAIN")
    VALID_DIR = os.path.join(dataset, "VALIDATE")

    # augment training data to get a larger training data set
    
    # data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        # rotation_range=15,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        # vertical_flip=True,
        # brightness_range=[0.5, 1.5],
    )

    # rescale validation testset
    valid_datagen = ImageDataGenerator(
        rescale = 1./255
    )

    train_data = train_datagen.flow_from_directory(TRAIN_DIR,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                target_size=IMG_SIZE,
                                                class_mode='binary',
                                                color_mode='rgb')
                                                
    validation_data = valid_datagen.flow_from_directory(VALID_DIR,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE // 2,
                                                target_size=IMG_SIZE,
                                                class_mode='binary',
                                                color_mode='rgb')

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=validation_data
    )

    # create a json file that lists the classification model contents
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)

    # serialize weights to HDF5
    model.save("../../models/cnn_model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    # process the dataset
    undoPreprocess()
    preprocess(num_augment_gen=0)

    # create the model
    model = cnn()
    
    # train the model
    train(model, BATCH_SIZE=32, EPOCHS=70, IMG_SIZE=(150,150))