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
    current = os.getcwd()

    # check which system you're running on to get path
    if (platform.system() == "Windows"):
        img_path = os.path.join(current, 'classify_data\imagesTr\\')
        label_path = os.path.join(current, 'classify_data\labelsTr\\')
    else:
        img_path = os.path.join(current, 'classify_data/imagesTr/')
        label_path = os.path.join(current, 'classify_data/labelsTr/')

    img_list = os.listdir(img_path)
    label_list = os.listdir(label_path)
    img_list = [img_path + img for img in img_list]
    label_list =  [label_path + label for label in label_list]
