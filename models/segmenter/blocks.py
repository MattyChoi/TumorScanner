import numpy as np
import tensorflow as tf
from keras import layers


def encode_block(inputs, n_filters, dropout_prob=0, max_pooling=True):

    conv = layers.Conv2D(n_filters, # Number of filters
                  3,   # Kernel size   
                  activation="relu",
                  padding="same",
                  kernel_initializer="he_normal")(inputs)
    conv = layers.Conv2D(n_filters, # Number of filters
                  3,   # Kernel size
                  activation="relu",
                  padding="same",
                  kernel_initializer="he_normal")(conv)
    
    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = layers.Dropout(dropout_prob)(conv)
        
    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = layers.MaxPooling2D((2, 2))(conv)
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection


def decode_block(expansive_input, contractive_input, n_filters):
    
    up = layers.Conv2DTranspose(
                 n_filters,    # number of filters
                 3,    # Kernel size
                 strides=2,
                 padding="same")(expansive_input)
                 
    # Merge the previous output and the contractive_input
    merge = layers.concatenate([up, contractive_input], axis=3)

    conv = layers.Conv2D(n_filters,   # Number of filters
                 3,     # Kernel size
                 activation="relu",
                 padding="same",
                 kernel_initializer="he_normal")(merge)

    conv = layers.Conv2D(n_filters,  # Number of filters
                 3,   # Kernel size
                 activation="relu",
                 padding="same",
                 kernel_initializer="he_normal")(conv)

    return conv