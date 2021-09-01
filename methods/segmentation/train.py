import os
import sys
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint
import cv2
# import gif_your_nifti.core as gif2nif

sys.path.insert(1, os.path.join(os.getcwd(), 'models'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from u_net_2D import *
from u_net_3D import *
from gif_maker import *


# how to create custom data generator in link: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# tensorflow docs for keras.utils.Sequence: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
class DataGenerator(Sequence):
    def __init__(self, data_ids, dim=(128, 128), batch_size = 1, n_channels = 2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.data_ids = data_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()


    # Denotes the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.data_ids) / self.batch_size))


    # generates a batch of data
    def __getitem__(self, index):
        # get indexes of batch data
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]

        # find list of IDs
        batch_ids = [self.data_ids[k] for k in indexes]

        # generate data
        X, y = self.__data_generation(batch_ids)

        return X, y


    # updates indexes after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    # generates data containing batch_size samples
    def __data_generation(self, batch_ids):
        numVolumes = 100
        startVolume = 22
        X = np.zeros((self.batch_size * numVolumes, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * numVolumes, 240, 240))

        # generate data
        for c, id in enumerate(batch_ids):
            # load all the data
            case_path = os.path.join(os.getcwd(), 'BRATS_data', 'brats_2020', 'Training', id)
            flair = nb.load(os.path.join(case_path, f'{id}_flair.nii')).get_fdata()
            ce = nb.load(os.path.join(case_path, f'{id}_t1ce.nii')).get_fdata()
            # t1 = nb.load(os.path.join(case_path, f'{id}_t1.nii')).get_fdata()
            # t2 = nb.load(os.path.join(case_path, f'{id}_t2.nii')).get_fdata()

            seg = nb.load(os.path.join(case_path, f'{id}_seg.nii')).get_fdata()
            for j in range(numVolumes):
                X[j + numVolumes * c,:,:,0] = cv2.resize(flair[:,:,j + startVolume], self.dim)
                X[j + numVolumes * c,:,:,1] = cv2.resize(ce[:,:,j + startVolume], self.dim)
                # X[j + numVolumes * c,:,:,2] = cv2.resize(t1[:,:,j + startVolume], self.dim)
                # X[j + numVolumes * c,:,:,3] = cv2.resize(t2[:,:,j + startVolume], self.dim)
                y[j + numVolumes * c] = seg[:,:,j + startVolume]
        X /= np.max(X)            

        # Generate masks
        y[y==4] = 3
        mask = tf.one_hot(y, 4)
        y = tf.image.resize(mask, self.dim)
        return X, y


def train(model, name, BATCH_SIZE=32, EPOCHS=50, IMG_SIZE=(128,128), NUM_CHANNELS = 2):
    current = os.getcwd()

    # set constants
    EPOCHS = EPOCHS
    BATCH_SIZE = BATCH_SIZE
    IMG_SIZE = IMG_SIZE
    NUM_CHANNELS = NUM_CHANNELS

    # get a list of all the paths to the directories containing the training data
    PATH = os.path.join(current, 'BRATS_data', 'brats_2020', 'Training')
    train_and_val_ids = [id for id in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, id))]
    
    # split list between training ids (80%) and validation ids (20%)
    train_ids, val_ids = train_test_split(train_and_val_ids, test_size=0.2)

    # create data generators for each id 
    training_generator = DataGenerator(train_ids, dim=IMG_SIZE, batch_size=BATCH_SIZE, n_channels=NUM_CHANNELS)
    validation_generator = DataGenerator(val_ids, dim=IMG_SIZE, batch_size=BATCH_SIZE, n_channels=NUM_CHANNELS)

    history = model.fit(
        training_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        # callbacks=[stop,checkpoint]
    )

    # serialize weights to HDF5
    model.save("../../trained_models/{}_model.h5".format(name))
    print("Saved model to disk")
    

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    input_shape = (240, 240)
    n_channels = 2

    # create the model
    model = unet_2D_model(input_shape=(*input_shape, n_channels))
    # print(model.summary())

    # train the model
    train(model, "u_net_2D", BATCH_SIZE=16, EPOCHS=30, IMG_SIZE=input_shape, NUM_CHANNELS=n_channels)