import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess():
    BATCH_SIZE = 32
    IMG_SIZE = (150, 150)

    current = os.getcwd()
    

    train_dataset = image_dataset_from_directory(directory,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE,
                                                validation_split=0.2,
                                                subset='training',
                                                seed=42)
    validation_dataset = image_dataset_from_directory(directory,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE,
                                                validation_split=0.2,
                                                subset='validation',
                                                seed=42)