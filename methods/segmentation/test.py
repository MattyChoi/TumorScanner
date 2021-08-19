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
    os.chdir('..\\')
    current = os.getcwd()
    
    # check which system you're running on to get path
    if (platform.system() == "Windows"):
        img_path = os.path.join(current, 'BRATS_data\imagesTr\\')
        label_path = os.path.join(current, 'BRATS_data\labelsTr\\')
    else:
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

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
    slice_w = 25
    ax1.imshow(test_image_flair[:,:,test_image_flair.shape[0]//2-slice_w], cmap = 'gray')
    ax1.set_title('Image flair')
    ax2.imshow(test_image_t1[:,:,test_image_t1.shape[0]//2-slice_w], cmap = 'gray')
    ax2.set_title('Image t1')
    ax3.imshow(test_image_t1ce[:,:,test_image_t1ce.shape[0]//2-slice_w], cmap = 'gray')
    ax3.set_title('Image t1ce')
    ax4.imshow(test_image_t2[:,:,test_image_t2.shape[0]//2-slice_w], cmap = 'gray')
    ax4.set_title('Image t2')
    ax5.imshow(test_mask[:,:,test_mask.shape[0]//2-slice_w])
    ax5.set_title('Mask')
    
    plt.show()