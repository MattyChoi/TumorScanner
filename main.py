import streamlit as st
from PIL import Image
from keras.models import load_model
import keras.preprocessing
import numpy as np

# create web application with user interface using streamlit
st.title("Tumor Scanner")

uploaded_file = st.file_uploader("Pick a file")
model = load_model("model.h5")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image = image.resize((150, 150))
    img_tensor = keras.preprocessing.image.img_to_array(image)
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.
