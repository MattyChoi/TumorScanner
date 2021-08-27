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

# get functions from ./methods/segmentation/preprocess.py
sys.path.insert(1, os.path.join(os.getcwd(), 'methods', 'segmentation'))
from preprocess import *

# create web application with user interface using streamlit
st.title("Tumor Scanner")

uploaded_file = st.file_uploader("Choose Brain Tumor File")

if uploaded_file:
    # use classifier if file is jpeg
    if uploaded_file.type == "image/jpeg":

        # load the image and convert to rgb
        img = Image.open(uploaded_file)
        img = img.convert('RGB')

        # show the image
        st.image(img, caption=uploaded_file.name, use_column_width=True)
        
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
    # elif uploaded_file.type == 'application/octet-stream':
    else:
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
        
        # choose segmenter model
        segment = os.path.join(os.getcwd(), "methods", "segmentation", "models")
        choices = np.array([choice[:-3] for choice in os.listdir(segment) if choice[-3:] == ".py"])
        model_choice = st.selectbox("Choose model", choices, key='segment_models')
        text = "Segmenting with " + model_choice + "..."
        st.write(text)

        # load model based on model choice
        # model = load_model(os.path.join(os.getcwd(), "trained_models", model_choice + "_model.h5"))
        
        st.write("work in progress")