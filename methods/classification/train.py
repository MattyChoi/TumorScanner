import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint

sys.path.insert(1, os.path.join(os.getcwd(), 'models'))
from cnn import *
from vgg16 import *
from trained_vgg16 import *
from preprocess import *

def train(model, name, BATCH_SIZE=32, EPOCHS=50, IMG_SIZE=(224,224)):
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
        rotation_range=15,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
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

    # callbacks
    # stop = EarlyStopping(
    #     monitor='val_accuracy', 
    #     mode='max',
    #     patience=6
    # )

    # checkpoint= ModelCheckpoint(
    #     filepath='./',
    #     save_weights_only=True,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True
    # )

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=validation_data,
        # callbacks=[stop,checkpoint]
    )

    # create a json file that lists the classification model contents
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)

    # serialize weights to HDF5
    model.save("../../trained_models/{}_model.h5".format(name))
    print("Saved model to disk")


if __name__ == "__main__":
    # process the dataset
    undoPreprocess()
    preprocess(num_augment_gen=30)

    # create the model
    model = vgg16()
    print(model.summary())

    # train the model
    train(model, "vgg16", BATCH_SIZE=32, EPOCHS=50, IMG_SIZE=(224,224))
