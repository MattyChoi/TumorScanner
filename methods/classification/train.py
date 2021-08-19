import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn import *
from preprocess import *

def train(model):
    current = os.getcwd()
    dataset = os.path.join(current, "classify_data")

    # set variables
    EPOCHS = 100
    BATCH_SIZE = 32
    IMG_SIZE = (150, 150)
    TRAIN_DIR = os.path.join(dataset, "TRAIN")
    VALID_DIR = os.path.join(dataset, "VALIDATE")

    # augment training data to get a larger training data set
    
    # data augmentation for training set
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip = True)

    # rescale validation testset
    valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_data = train_datagen.flow_from_directory(TRAIN_DIR,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                target_size=IMG_SIZE,
                                                class_mode='binary',
                                                color_mode='rgb',)
                                                
    validation_data = valid_datagen.flow_from_directory(VALID_DIR,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE // 2,
                                                target_size=IMG_SIZE,
                                                class_mode='binary',
                                                color_mode='rgb',)

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=validation_data
    )

    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)

    # serialize weights to HDF5
    model.save("model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    # process the dataset
    preprocess()

    # create the model
    model = cnn()
    
    # train the model
    train(model)