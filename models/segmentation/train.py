import os
import platform
import numpy as np
import nibabel as nib
import nilearn as nl
import nilearn.plotting as nlplt
import matplotlib.pyplot as plt

def train(model, data):

    return 1

if __name__ == "__main__":
    # check which system you're running on to get path
    if (platform.system() == "Windows"):
        os.chdir('..\\..\\')
        current = os.getcwd()
        img_path = os.path.join(current, 'BRATS_data\imagesTr\\')
        label_path = os.path.join(current, 'BRATS_data\labelsTr\\')
    else:
        os.chdir('..\\..\\')
        current = os.getcwd()
        img_path = os.path.join(current, 'BRATS_data/imagesTr/')
        label_path = os.path.join(current, 'BRATS_data/labelsTr/')

    img_list = os.listdir(img_path)
    label_list = os.listdir(label_path)
    img_list = [img_path + img for img in img_list]
    label_list =  [label_path + label for label in label_list]

    # pick which image to choose from
    index = 60

    img = nib.load(img_list[index]).get_fdata()
    label = nib.load(label_list[index]).get_fdata()

    # (FLAIR, T1w, T1gd,T2w)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
    ax1.imshow(img[:,:,img.shape[2]//2, 0], cmap = 'gray')
    ax1.set_title('Image t1')
    ax2.imshow(img[:,:,img.shape[2]//2, 1], cmap = 'gray')
    ax2.set_title('Image t1Gd')
    ax3.imshow(img[:,:,img.shape[2]//2, 2], cmap = 'gray')
    ax3.set_title('Image t2')
    ax4.imshow(img[:,:,img.shape[2]//2, 3], cmap = 'gray')
    ax4.set_title('Image flair')
    ax5.imshow(label[:,:,img.shape[2]//2])
    ax5.set_title('Mask')
    
    plt.show()