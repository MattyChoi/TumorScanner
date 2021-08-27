import os
import sys
import numpy as np
import nibabel as nb
import nilearn as nl
import nilearn.plotting as nlplt
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage 
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint
# import gif_your_nifti.core as gif2nif

sys.path.insert(1, os.path.join(os.getcwd(), 'models'))
from U_Net_2D import *
from U_Net_3D import *
from preprocess import *


def train(model, name):
    current = os.getcwd()
    img_path = os.path.join(current, 'BRATS_data', 'medicathlon', 'imagesTr')
    label_path = os.path.join(current, 'BRATS_data', 'medicathlon', 'labelsTr')

    img_list = os.listdir(img_path)
    label_list = os.listdir(label_path)
    img_list = [os.path.join(img_path, img) for img in img_list]
    label_list =  [os.path.join(label_path, label) for label in label_list]

    # pick which image to choose from
    index = 65

    img = nb.load(img_list[index]).get_fdata()
    label = nb.load(label_list[index]).get_fdata()

    whichImg = img.shape[2]//2 + 10

    # (FLAIR, T1w, T1gd,T2w)
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize = (12, 6))
    ax1.imshow(img[:,:,whichImg,0], cmap = 'gray')
    ax1.set_title('FLAIR')
    ax2.imshow(img[:,:,whichImg, 1], cmap = 'gray')
    ax2.set_title('T1w')
    ax3.imshow(img[:,:,whichImg, 2], cmap = 'gray')
    ax3.set_title('T1gd')
    ax4.imshow(img[:,:,whichImg, 3], cmap = 'gray')
    ax4.set_title('T2w')
    ax5.imshow(label[:,:,whichImg] == 0)
    ax6.imshow(label[:,:,whichImg] == 1)
    ax7.imshow(label[:,:,whichImg] == 2)
    ax8.imshow(label[:,:,whichImg] == 3)
    ax5.set_title('background')
    ax6.set_title('edema')
    ax7.set_title('non-enhancing tumor')
    ax8.set_title('enhancing tumour')
    plt.show()
        
    fig, ax1 = plt.subplots(1, 1, figsize = (10,10))
    ax1.imshow(rotate(montage(img[50:-50,:,:,0]), 90, resize=True), cmap ='gray')
    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize = (10,10))
    ax1.imshow(rotate(montage(label[50:-50,:,:]), 90, resize=True), cmap ='gray')
    plt.show()

    img = os.path.join(current, 'BRATS_data', 'brats_2020', 'Training', 'BraTS20_Training_016', 'BraTS20_Training_016_flair.nii')
    data = nb.load(img)
    write_gif_normal(data)
    

if __name__ == "__main__":
    train("yes", "no")