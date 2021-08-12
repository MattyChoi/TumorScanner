import numpy as np
import tensorflow as tf
import keras as layers
from blocks import encode_block, decode_block

def unet_model(input_size=(256, 256, 3), n_filters=64, n_classes=10):
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

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model