import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras import Input
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity, he_normal

# use keras's functional api to build convolutional model for segmentation
def unet_2D_model(input_shape=(240, 240, 2), n_filters=32, n_classes=4):
    # encode blocks have two outputs: first one is the output 
    # and the second is for skip connection

    inputs = Input(input_shape)
    encode1 = encode_block(inputs, n_filters)
    encode2 = encode_block(encode1[0], n_filters*2)
    encode3 = encode_block(encode2[0], n_filters*4)
    encode4 = encode_block(encode3[0], n_filters*8)

    encode5 = encode_block(encode4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    decode6 = decode_block(encode5[0], encode4[1], n_filters*8)
    decode7 = decode_block(decode6, encode3[1], n_filters*4)
    decode8 = decode_block(decode7, encode2[1], n_filters*2)
    decode9 = decode_block(decode8, encode1[1], n_filters)

    conv = Conv2D(n_filters, 3, 
                    activation='relu',
                    padding='same', 
                    kernel_initializer=he_normal)(decode9)
    outputs = Conv2D(n_classes, 1, activation = 'softmax')(conv)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss="categorical_crossentropy", 
                optimizer=Adam(learning_rate=1e-3), 
                metrics = ['accuracy', 
                            tf.keras.metrics.MeanIoU(num_classes=4), 
                            dice_coef, 
                            precision, 
                            sensitivity, 
                            specificity, 
                            dice_coef_necrotic, 
                            dice_coef_edema,
                            dice_coef_enhancing
                        ])

    return model


def encode_block(inputs, n_filters, dropout_prob=0, max_pooling=True):

    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size   
                  activation="relu",
                  padding="same",
                  kernel_initializer=he_normal)(inputs)
    conv = BatchNormalization()(conv)
    conv = Conv2D(n_filters, # Number of filters
                  3,   # Kernel size
                  activation="relu",
                  padding="same",
                  kernel_initializer=he_normal)(conv)
    conv = BatchNormalization()(conv)
    
    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
        
    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D((2, 2))(conv)
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection


def decode_block(expansive_input, contractive_input, n_filters):
    
    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 3,    # Kernel size
                 strides=2,
                 padding="same")(expansive_input)
                 
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)

    conv = Conv2D(n_filters,   # Number of filters
                 3,     # Kernel size
                 activation="relu",
                 padding="same",
                 kernel_initializer=he_normal)(merge)

    conv = Conv2D(n_filters,  # Number of filters
                 3,   # Kernel size
                 activation="relu",
                 padding="same",
                 kernel_initializer=he_normal)(conv)

    return conv


# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        # K.print_tensor(loss, message='loss value for class {} : '.format(SEGMENT_CLASSES[i]))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    # K.print_tensor(total_loss, message=' total dice coef: ')
    return total_loss


 
# define per class evaluation of dice coef
# inspired by https://github.com/keras-team/keras/issues/9395
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)



# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    
# Computing Sensitivity      
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())