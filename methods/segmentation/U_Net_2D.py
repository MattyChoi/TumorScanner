import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model

# use keras's functional api to build convolutional model for segmentation
def unet_2D_model(input_size=(240, 240, 3), n_filters=64, n_classes=10):
    # encode blocks have two outputs: first one is the output 
    # and the second is for skip connection

    inputs = layers.Input(input_size)
    encode1 = encode_block(inputs, n_filters)
    encode2 = encode_block(encode1[0], n_filters*2)
    encode3 = encode_block(encode2[0], n_filters*4)
    encode4 = encode_block(encode3[0], n_filters*8)

    encode5 = encode_block(encode4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    decode6 = decode_block(encode5[0], encode4[1], n_filters*8)
    decode7 = decode_block(decode6, encode3[1], n_filters*4)
    decode8 = decode_block(decode7, encode2[1], n_filters*2)
    decode9 = decode_block(decode8, encode1[1], n_filters)

    conv = layers.Conv2D(n_filters, 3, 
                    activation='relu',
                    padding='same', 
                    kernel_initializer='he_normal')(decode9)
    outputs = layers.Conv2D(n_classes, 1, padding='same')(conv)

    model = Model(inputs=inputs, outputs=outputs)

    return model


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