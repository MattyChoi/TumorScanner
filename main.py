import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from io import BytesIO
from imageio import mimwrite
import os
import sys
import nibabel as nb
from nibabel import FileHolder, Nifti1Image, Nifti2Image
import tensorflow as tf
import cv2

# get functions from ./methods/segmentation/preprocess.py
sys.path.insert(1, os.path.join(os.getcwd(), 'methods', 'segmentation'))
from gif_maker import *
sys.path.insert(1, os.path.join(os.getcwd(), 'methods', 'segmentation', 'models'))
from u_net_2D import *

# checks if uploaded files are all nifti files
def checkNifti(images):
    for image in images:
        if image.type != "application/octet-stream":
            return False
    return True

# create web application with user interface using streamlit
st.title("Tumor Scanner")

uploaded_files = st.file_uploader("Choose Brain Tumor File", accept_multiple_files=True)

if uploaded_files:
    # use classifier if file is jpeg
    if len(uploaded_files) == 1 and uploaded_files[0].type == "image/jpeg":

        # load the image and convert to rgb
        img = Image.open(uploaded_files[0])
        img = img.convert('RGB')

        # show the image
        st.image(img, caption=uploaded_files[0].name, use_column_width=True)
        
        # choose classifier model
        classify = os.path.join(os.getcwd(), "methods", "classification", "models")
        choices = np.array([choice[:-3] for choice in os.listdir(classify) if choice[-3:] == ".py"])
        model_choice = st.selectbox("Choose model", choices, key='classify_models')
        text = "Classifying with " + model_choice + "..."
        st.write(text)

        # load model based on model choice
        model = load_model(os.path.join(os.getcwd(), "trained_models", model_choice + "_model.h5"))

        # resize image and make it into an array
        img = img.resize((224, 224))
        input = img_to_array(img)

        # have to expand dims to get shape (None, 224, 224, 3)
        input = np.expand_dims(input, axis=0)
        
        pred = model.predict(input)

        text = "Classified with " + model_choice
        st.write(text)
        
        if pred < 0.5:
            st.write("Tumor not found")
        else:
            st.write("Tumor detected")

    # use segmenter if file is nifti file
    elif len(uploaded_files) > 1 and checkNifti(uploaded_files):
        dim = (144, 144)
        input = np.zeros((155, *dim, 2))
        reqs = [0, 0]

        for uploaded_file in uploaded_files:
            # get name
            name = uploaded_file.name[uploaded_file.name.rfind('_') + 1:]

            # hold the file
            file = FileHolder(fileobj=uploaded_file)

            # read the file 
            img = Nifti1Image.from_file_map({'header': file, 'image': file})
            data = img.get_fdata()

            # put data in right shape
            out_img, maximum = prepare_image(data, 1)

            # create output mosaic
            new_img = create_mosaic_normal(out_img, maximum)
            
            # create gif
            gif = BytesIO()
            gif.name = "temp.gif"
            mimwrite(gif, new_img, format='gif', fps=int(18))

            # show the image
            st.image(gif, caption=uploaded_file.name, use_column_width=True)

            if name == "flair.nii":
                for i in range(data.shape[2]):
                    input[i,:,:,0] = cv2.resize(data[:,:,i], dim)
                reqs[0] += 1
            elif name == "t1ce.nii":
                for i in range(data.shape[2]):
                    input[i,:,:,1] = cv2.resize(data[:,:,i], dim)
                reqs[1] += 1

        if reqs != [1, 1]:
            st.write("Need correct input files")
        else:
            # choose segmenter model
            segment = os.path.join(os.getcwd(), "methods", "segmentation", "models")
            choices = np.array([choice[:-3] for choice in os.listdir(segment) if choice[-3:] == ".py"])
            model_choice = st.selectbox("Choose model", choices, key='segment_models')
            text = "Segmenting with " + model_choice + "..."
            st.write(text)

            # load model based on model choice
            model = load_model(os.path.join(os.getcwd(), "trained_models", model_choice + "_model.h5"), 
                                custom_objects = { 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                    "dice_coef": dice_coef,
                                                    "precision": precision,
                                                    "sensitivity": sensitivity,
                                                    "specificity": specificity,
                                                    "dice_coef_necrotic": dice_coef_necrotic,
                                                    "dice_coef_edema": dice_coef_edema,
                                                    "dice_coef_enhancing": dice_coef_enhancing}, 
                                compile=False
            )
            
            pred = model.predict(input/np.max(input))

            # fetch the most likely labels
            pred = tf.argmax(pred, axis=-1)
            # pred[pred==3] = 4
            pred = tf.where(pred==3, 4, pred)
            img = np.zeros((pred.shape[1], pred.shape[2], pred.shape[0]))
            for i in range(pred.shape[0]):
                img[:,:,i] = pred[i,:,:]

            # resize prediction to match the format of the original files
            # lossy because resizing from a smaller fram to a bigger frame
            img = tf.image.resize(img, (240, 240))
            
            # put data in right shape
            out_img, maximum = prepare_image(img, 1)

            # create output mosaic
            new_img = create_mosaic_normal(out_img, maximum)
            
            # create gif
            pred_gif = BytesIO()
            pred_gif.name = "pred.gif"
            mimwrite(pred_gif, new_img, format='gif', fps=int(18))

            text = "Segmented with " + model_choice
            st.write(text)

            # show the image
            st.image(pred_gif, caption=pred_gif.name, use_column_width=True)
    else:
        st.write("Incorrect file formats")
