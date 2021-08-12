import numpy as np
import tensorflow as tf
import keras as layers
from blocks import encode_block, decode_block

def unet_model(input_size=(256, 256, 3), n_filters=32, n_classes=10):
    inputs = layers.Input(input_size)
