import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import os

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
        st.image(img, caption=uploaded_file, use_column_width=True)
        
        # choose classifier model
        classify = os.path.join(os.getcwd(), "methods", "classification", "models")
        choices = np.array([choice[:-3] for choice in os.listdir(classify) if choice[-3:] == ".py"])
        model_choice = st.selectbox("Choose model", choices, key='models')
        st.write("Classifying with " + model_choice + "...")

        # load model based on model choice
        model = load_model(os.path.join(os.getcwd(), "trained_models", model_choice + "_model.h5"))

        # resize image and make it into an array
        img = img.resize((224, 224))
        input = img_to_array(img)

        # have to expand dims to get shape (None, 224, 224, 3)
        input = np.expand_dims(input, axis=0)
        
        pred = model.predict(input)
        
        if pred < 0.5:
            st.write('Classified: Tumor not found')
        else:
            st.write("Classified: Tumor detected")

    # use segmenter if file is nifti file
    else:
        st.write("work in progress")

