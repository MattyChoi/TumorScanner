import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# create web application with user interface using streamlit
st.title("Tumor Scanner")

uploaded_file = st.file_uploader("Pick a file")
model = load_model(os.path.join(os.getcwd(), "models", "cnn_model.h5"))

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(150, 150), color_mode='rgb')
    img = img.convert('RGB')
    st.write(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = img.resize((150, 150))
    input = img_to_array(img)
    input = np.expand_dims(input, axis=0)
    
    pred = model.predict(input)

    st.write("Classified")
    
    if pred < 0.5:
        st.write('Tumor not found')
    else:
        st.write("Tumor detected")